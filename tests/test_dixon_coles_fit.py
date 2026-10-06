"""The fast (analytic-gradient) Dixon-Coles fit and its optional smoothing."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.optimize import check_grad, minimize

from soccer_predictor.model import dixon_coles as dc
from soccer_predictor.model.dixon_coles import _fit_objective, _neg_log_likelihood, fit_league
from soccer_predictor.model.time_weighting import match_weights


def _simulate(n_teams=8, n_matches=400, seed=0, strengths=None):
    rng = np.random.default_rng(seed)
    attack = np.exp(rng.normal(0, 0.3, n_teams)) if strengths is None else strengths
    defense = np.exp(rng.normal(0, 0.3, n_teams))
    rows = []
    start = dt.date(2024, 1, 1)
    for k in range(n_matches):
        h, a = rng.choice(n_teams, 2, replace=False)
        lam_h, lam_a = attack[h] * defense[a] * 1.3, attack[a] * defense[h]
        rows.append((start + dt.timedelta(days=k), h, a, rng.poisson(lam_h), rng.poisson(lam_a)))
    return pd.DataFrame(rows, columns=["date", "home_team_id", "away_team_id", "home_goals", "away_goals"])


def _arrays(df):
    ids = sorted(set(df.home_team_id) | set(df.away_team_id))
    idx = {t: i for i, t in enumerate(ids)}
    return (
        len(ids),
        df.home_team_id.map(idx).to_numpy(),
        df.away_team_id.map(idx).to_numpy(),
        df.home_goals.to_numpy(),
        df.away_goals.to_numpy(),
        match_weights(df.date, as_of=df.date.max(), xi=0.002),
    )


def test_the_analytic_gradient_matches_the_numerical_one():
    n, h, a, hg, ag, w = _arrays(_simulate())
    objective, _ = _fit_objective(n, h, a, hg, ag, w, 0.0, None, 0)
    x = np.random.default_rng(1).normal(0, 0.2, 2 * n + 2)
    x[2 * n + 1] = 0.05  # rho
    err = check_grad(lambda v: objective(v)[0], lambda v: objective(v)[1], x)
    assert err < 1e-3 * np.linalg.norm(objective(x)[1])


def test_the_gradient_is_also_right_with_smoothing_and_groups():
    n, h, a, hg, ag, w = _arrays(_simulate())
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    objective, n_extra = _fit_objective(n, h, a, hg, ag, w, 0.7, groups, 4)
    assert n_extra == 8
    x = np.random.default_rng(2).normal(0, 0.2, 2 * n + 2 + n_extra)
    x[2 * n + 1] = -0.03
    err = check_grad(lambda v: objective(v)[0], lambda v: objective(v)[1], x)
    assert err < 1e-3 * np.linalg.norm(objective(x)[1])


def test_the_objective_equals_the_original_likelihood():
    n, h, a, hg, ag, w = _arrays(_simulate())
    objective, _ = _fit_objective(n, h, a, hg, ag, w, 0.0, None, 0)
    x = np.random.default_rng(3).normal(0, 0.2, 2 * n + 2)
    x[2 * n + 1] = 0.04
    assert abs(objective(x)[0] - _neg_log_likelihood(x, h, a, hg, ag, w, n)) < 1e-8


def test_the_fit_matches_the_old_numerical_fit():
    df = _simulate(seed=5)
    fast = fit_league(df, "T", xi=0.002)

    n, h, a, hg, ag, w = _arrays(df)
    init = np.zeros(2 * n + 2)
    init[2 * n] = np.log(1.3)
    bounds = [(-3, 3)] * (2 * n) + [(-2, 2), (-0.9, 0.9)]
    slow = minimize(_neg_log_likelihood, init, args=(h, a, hg, ag, w, n), method="L-BFGS-B", bounds=bounds)
    attack = np.exp(slow.x[:n])
    scale = 1.0 / attack.mean()
    ids = sorted(set(df.home_team_id) | set(df.away_team_id))
    for i, team in enumerate(ids):
        assert abs(fast.attack[team] - attack[i] * scale) < 0.02
        assert abs(fast.defense[team] - np.exp(slow.x[n + i]) / scale) < 0.02
    assert abs(fast.home_advantage - np.exp(slow.x[2 * n])) < 0.01
    assert abs(fast.rho - slow.x[2 * n + 1]) < 0.01


def test_ridge_pulls_ratings_toward_the_league_average():
    df = _simulate(seed=7, n_matches=120)
    plain = fit_league(df, "T", xi=0.002)
    shrunk = fit_league(df, "T", xi=0.002, ridge=3.0)

    def spread(p):
        return float(np.std(np.log(list(p.attack.values()))))

    assert spread(shrunk) < spread(plain)


def test_default_arguments_still_mean_plain_maximum_likelihood():
    df = _simulate(seed=9)
    a = fit_league(df, "T", xi=0.002)
    b = fit_league(df, "T", xi=0.002, ridge=0.0, team_groups=None)
    assert a.attack == b.attack and a.defense == b.defense


def test_group_priors_shrink_toward_the_teams_own_group_not_the_league_mean():
    # Teams 0-3 are strong (a "League A"), 4-7 are weak ("League D"). Team 3
    # has only a couple of matches, so with heavy smoothing its rating is
    # dominated by the prior: its own group's level vs one league-wide level.
    rng = np.random.default_rng(11)
    attack = np.array([1.6, 1.5, 1.7, 1.55, 0.5, 0.45, 0.55, 0.5])
    defense = np.array([0.6, 0.65, 0.55, 0.6, 1.5, 1.6, 1.55, 1.5])
    rows, day = [], dt.date(2024, 1, 1)
    for _ in range(500):
        pool = [t for t in range(8) if t != 3] if rng.random() > 0.01 else list(range(8))
        h, a = rng.choice(pool, 2, replace=False)
        rows.append((day, h, a, rng.poisson(attack[h] * defense[a] * 1.3), rng.poisson(attack[a] * defense[h])))
        day += dt.timedelta(days=1)
    for _ in range(3):  # team 3 plays just three times
        a = int(rng.choice([0, 1, 2]))
        rows.append((day, 3, a, rng.poisson(attack[3] * defense[a] * 1.3), rng.poisson(attack[a] * defense[3])))
        day += dt.timedelta(days=1)
    df = pd.DataFrame(rows, columns=["date", "home_team_id", "away_team_id", "home_goals", "away_goals"])
    groups = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}

    league_mean = fit_league(df, "T", xi=0.001, ridge=8.0)
    group_mean = fit_league(df, "T", xi=0.001, ridge=8.0, team_groups=groups)

    # Net strength of team 3 (high attack, low defence is good).
    strength = lambda p: p.attack[3] / p.defense[3]
    assert strength(group_mean) > strength(league_mean)


def test_teams_missing_from_the_group_map_use_the_default_group():
    df = _simulate(seed=13)
    only_some = fit_league(df, "T", xi=0.002, ridge=1.0, team_groups={0: 1, 1: 1}, default_group=0)
    assert set(only_some.attack) == set(sorted(set(df.home_team_id) | set(df.away_team_id)))


def test_a_fit_is_fast(monkeypatch):
    import time

    df = _simulate(n_teams=20, n_matches=3000, seed=17)
    start = time.perf_counter()
    fit_league(df, "T")
    assert time.perf_counter() - start < 5  # the numerical-gradient version took ~10s at this size
