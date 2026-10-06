"""Dixon-Coles (1997) model: per-team attack/defense strength fit by MLE.

lambda_home = attack_home * defense_away * home_advantage
lambda_away = attack_away * defense_home

`tau(x, y, rho)` nudges the independent-Poisson joint probability for the
low-scoring cells (0-0, 1-0, 0-1, 1-1), correcting the well-known empirical
fact that plain independent Poisson under-predicts low-scoring draws.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from soccer_predictor.model.time_weighting import DEFAULT_XI, match_weights


@dataclass
class DixonColesParams:
    league_code: str
    attack: dict[int, float]
    defense: dict[int, float]
    home_advantage: float
    rho: float
    xi: float
    fitted_at: str
    n_matches: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "league_code": self.league_code,
                "attack": self.attack,
                "defense": self.defense,
                "home_advantage": self.home_advantage,
                "rho": self.rho,
                "xi": self.xi,
                "fitted_at": self.fitted_at,
                "n_matches": self.n_matches,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "DixonColesParams":
        data = json.loads(raw)
        return cls(
            league_code=data["league_code"],
            attack={int(k): v for k, v in data["attack"].items()},
            defense={int(k): v for k, v in data["defense"].items()},
            home_advantage=data["home_advantage"],
            rho=data["rho"],
            xi=data["xi"],
            fitted_at=data["fitted_at"],
            n_matches=data["n_matches"],
        )


def tau(home_goals: np.ndarray, away_goals: np.ndarray, rho: float) -> np.ndarray:
    """Low-score correlation adjustment; 1.0 outside the four special cells."""
    result = np.ones_like(home_goals, dtype=float)
    result = np.where((home_goals == 0) & (away_goals == 0), 1.0 - rho, result)
    result = np.where((home_goals == 0) & (away_goals == 1), 1.0 + rho, result)
    result = np.where((home_goals == 1) & (away_goals == 0), 1.0 + rho, result)
    result = np.where((home_goals == 1) & (away_goals == 1), 1.0 - rho, result)
    # Guard against the optimizer wandering into a region where this would
    # go non-positive (log-likelihood is undefined there).
    return np.clip(result, 1e-6, None)


def _neg_log_likelihood(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    log_attack = params[:n_teams]
    log_defense = params[n_teams : 2 * n_teams]
    log_gamma = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    attack = np.exp(log_attack)
    defense = np.exp(log_defense)
    gamma = np.exp(log_gamma)

    lambda_home = attack[home_idx] * defense[away_idx] * gamma
    lambda_away = attack[away_idx] * defense[home_idx]

    poisson_ll = poisson.logpmf(home_goals, lambda_home) + poisson.logpmf(
        away_goals, lambda_away
    )
    correction = np.log(tau(home_goals, away_goals, rho))
    ll = weights * (poisson_ll + correction)
    return -np.sum(ll)


def _fit_objective(
    n_teams: int,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    ridge: float,
    group_of_team: np.ndarray | None,
    n_groups: int,
):
    """The same likelihood as `_neg_log_likelihood`, plus an optional penalty,
    returned as one function giving (value, gradient).

    Giving the optimiser the exact gradient (derived by hand below) instead of
    letting it estimate one numerically makes a fit ~70x faster -- 0.13 s
    instead of ~10 s for a top-flight league's history -- with the same ratings
    (see tests/test_dixon_coles_fit.py, which checks the gradient against the
    numerical one and the fit against the old implementation).

    Parameter vector: [log_attack (n), log_defense (n), log_gamma, rho] and, with
    `group_of_team`, [group mean log-attack (g), group mean log-defense (g)].
    `ridge` penalises each team's log rating's distance from its group's mean
    (from 0 without groups): a "shrink toward your group" prior, with the
    group means themselves free.
    """
    n = n_teams
    is_00_or_11 = ((home_goals == 0) & (away_goals == 0)) | ((home_goals == 1) & (away_goals == 1))
    is_01_or_10 = ((home_goals == 0) & (away_goals == 1)) | ((home_goals == 1) & (away_goals == 0))
    log_factorials = gammaln(home_goals + 1) + gammaln(away_goals + 1)
    n_extra = 2 * n_groups if group_of_team is not None else 0

    def objective(x: np.ndarray):
        log_attack, log_defense = x[:n].copy(), x[n : 2 * n].copy()
        log_gamma, rho = x[2 * n], x[2 * n + 1]
        if group_of_team is not None:
            mean_attack = x[2 * n + 2 : 2 * n + 2 + n_groups]
            mean_defense = x[2 * n + 2 + n_groups :]
            team_attack = log_attack + mean_attack[group_of_team]
            team_defense = log_defense + mean_defense[group_of_team]
        else:
            team_attack, team_defense = log_attack, log_defense

        lambda_home = np.exp(team_attack[home_idx] + team_defense[away_idx] + log_gamma)
        lambda_away = np.exp(team_attack[away_idx] + team_defense[home_idx])

        tau_value = np.ones_like(lambda_home)
        tau_value = np.where(is_00_or_11, 1.0 - rho, tau_value)
        tau_value = np.where(is_01_or_10, 1.0 + rho, tau_value)
        tau_value = np.clip(tau_value, 1e-6, None)  # same guard as tau()

        log_likelihood = weights * (
            home_goals * np.log(lambda_home)
            - lambda_home
            + away_goals * np.log(lambda_away)
            - lambda_away
            - log_factorials
            + np.log(tau_value)
        )
        value = -log_likelihood.sum()

        # d log-lik / d log(lambda) for a Poisson count is (goals - lambda).
        home_residual = weights * (home_goals - lambda_home)
        away_residual = weights * (away_goals - lambda_away)
        grad_team_attack = np.bincount(home_idx, home_residual, n) + np.bincount(away_idx, away_residual, n)
        grad_team_defense = np.bincount(away_idx, home_residual, n) + np.bincount(home_idx, away_residual, n)
        d_log_tau = np.where(is_00_or_11, -1.0 / (1.0 - rho), 0.0) + np.where(is_01_or_10, 1.0 / (1.0 + rho), 0.0)

        grad = np.zeros_like(x)
        grad[:n] = -grad_team_attack
        grad[n : 2 * n] = -grad_team_defense
        grad[2 * n] = -home_residual.sum()
        grad[2 * n + 1] = -(weights * d_log_tau).sum()
        if group_of_team is not None:
            grad[2 * n + 2 : 2 * n + 2 + n_groups] = -np.bincount(group_of_team, grad_team_attack, n_groups)
            grad[2 * n + 2 + n_groups :] = -np.bincount(group_of_team, grad_team_defense, n_groups)
        if ridge:
            value += ridge * (np.sum(log_attack**2) + np.sum(log_defense**2))
            grad[:n] += 2 * ridge * log_attack
            grad[n : 2 * n] += 2 * ridge * log_defense
        return value, grad

    return objective, n_extra


def fit_league(
    matches: pd.DataFrame,
    league_code: str,
    xi: float = DEFAULT_XI,
    as_of: dt.date | None = None,
    ridge: float = 0.0,
    team_groups: dict[int, int] | None = None,
    default_group: int = 0,
) -> DixonColesParams:
    """Fits Dixon-Coles parameters from a matches DataFrame.

    `matches` needs columns: date, home_team_id, away_team_id, home_goals,
    away_goals (see storage.repository.matches_for_league).

    `ridge` (default 0: plain maximum likelihood, as before) shrinks every
    team's rating toward its group's average. With no `team_groups` that
    is one group -- everyone shrinks toward the league average, which helps
    when each team has few matches (tournaments played every few years).
    With `team_groups` ({team_id: group index}, teams missing from it go to
    `default_group`) each group gets its own free average first -- e.g.
    Nations League divisions A-D, whose teams differ hugely in strength --
    so a team with little data is pulled toward its division's level
    rather than toward a meaningless league-wide mean.
    """
    if matches.empty:
        raise ValueError(f"No matches to fit for league {league_code}")

    team_ids = sorted(set(matches.home_team_id) | set(matches.away_team_id))
    id_to_idx = {team_id: idx for idx, team_id in enumerate(team_ids)}
    n_teams = len(team_ids)

    home_idx = matches.home_team_id.map(id_to_idx).to_numpy()
    away_idx = matches.away_team_id.map(id_to_idx).to_numpy()
    home_goals = matches.home_goals.to_numpy()
    away_goals = matches.away_goals.to_numpy()
    weights = match_weights(matches.date, as_of=as_of, xi=xi)

    group_of_team = None
    n_groups = 0
    if team_groups is not None:
        group_of_team = np.array([team_groups.get(team_id, default_group) for team_id in team_ids])
        n_groups = int(group_of_team.max()) + 1

    objective, n_extra = _fit_objective(
        n_teams, home_idx, away_idx, home_goals, away_goals, weights, ridge, group_of_team, n_groups
    )

    init = np.zeros(2 * n_teams + 2 + n_extra)
    init[2 * n_teams] = np.log(1.3)  # home_advantage starting guess
    init[2 * n_teams + 1] = 0.0  # rho starting guess

    bounds = (
        [(-3.0, 3.0)] * n_teams  # log_attack
        + [(-3.0, 3.0)] * n_teams  # log_defense
        + [(-2.0, 2.0)]  # log_gamma
        + [(-0.9, 0.9)]  # rho
        + [(-3.0, 3.0)] * n_extra  # group means
    )

    result = minimize(objective, init, jac=True, method="L-BFGS-B", bounds=bounds)

    log_attack = result.x[:n_teams]
    log_defense = result.x[n_teams : 2 * n_teams]
    if group_of_team is not None:
        log_attack = log_attack + result.x[2 * n_teams + 2 : 2 * n_teams + 2 + n_groups][group_of_team]
        log_defense = log_defense + result.x[2 * n_teams + 2 + n_groups :][group_of_team]
    gamma = float(np.exp(result.x[2 * n_teams]))
    rho = float(result.x[2 * n_teams + 1])

    attack = np.exp(log_attack)
    defense = np.exp(log_defense)

    # attack_i*defense_j*gamma is invariant to attack *= c, defense /= c for
    # all teams at once, so this normalization doesn't change the fit -- it
    # just fixes the arbitrary scale so ratings are comparable/interpretable.
    scale = 1.0 / attack.mean()
    attack = attack * scale
    defense = defense / scale

    return DixonColesParams(
        league_code=league_code,
        attack={team_id: float(attack[idx]) for team_id, idx in id_to_idx.items()},
        defense={team_id: float(defense[idx]) for team_id, idx in id_to_idx.items()},
        home_advantage=gamma,
        rho=rho,
        xi=xi,
        fitted_at=dt.datetime.now(dt.UTC).isoformat(),
        n_matches=len(matches),
    )
