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


def fit_league(
    matches: pd.DataFrame,
    league_code: str,
    xi: float = DEFAULT_XI,
    as_of: dt.date | None = None,
) -> DixonColesParams:
    """Fits Dixon-Coles parameters from a matches DataFrame.

    `matches` needs columns: date, home_team_id, away_team_id, home_goals,
    away_goals (see storage.repository.matches_for_league).
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

    init = np.zeros(2 * n_teams + 2)
    init[2 * n_teams] = np.log(1.3)  # home_advantage starting guess
    init[2 * n_teams + 1] = 0.0  # rho starting guess

    bounds = (
        [(-3.0, 3.0)] * n_teams  # log_attack
        + [(-3.0, 3.0)] * n_teams  # log_defense
        + [(-2.0, 2.0)]  # log_gamma
        + [(-0.9, 0.9)]  # rho
    )

    result = minimize(
        _neg_log_likelihood,
        init,
        args=(home_idx, away_idx, home_goals, away_goals, weights, n_teams),
        method="L-BFGS-B",
        bounds=bounds,
    )

    log_attack = result.x[:n_teams]
    log_defense = result.x[n_teams : 2 * n_teams]
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
