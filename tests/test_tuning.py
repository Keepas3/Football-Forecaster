"""Walk-forward scoring of a league's rating settings (prediction/tuning.py)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from soccer_predictor.model.dixon_coles import fit_league
from soccer_predictor.prediction import tuning


def _league_matches(seasons=4, seed=3):
    """Eight clubs of very different strength, each pair meeting home and away once a season."""
    rng = np.random.default_rng(seed)
    strength = [2.4, 1.8, 1.4, 1.1, 0.9, 0.7, 0.55, 0.4]
    rows = []
    for season in range(seasons):
        day = dt.date(2020 + season, 9, 1)
        for home in range(8):
            for away in range(8):
                if home == away:
                    continue
                day += dt.timedelta(days=2)
                rows.append(
                    {
                        "date": day,
                        "season": f"{20 + season}{21 + season}",
                        "home_team_id": home + 1,
                        "away_team_id": away + 1,
                        "home_goals": int(rng.poisson(1.3 * strength[home] / strength[away] ** 0.5)),
                        "away_goals": int(rng.poisson(1.0 * strength[away] / strength[home] ** 0.5)),
                    }
                )
    return pd.DataFrame(rows)


def test_a_forecast_never_sees_its_own_month(monkeypatch):
    seen = []

    def spying_fit(train, league_code, **kwargs):
        seen.append((train["date"].max(), kwargs["as_of"]))
        return fit_league(train, league_code, **kwargs)

    monkeypatch.setattr(tuning, "fit_league", spying_fit)
    matches = _league_matches()
    scored = tuning.walk_forward_rps(matches, "X", xi=0.001, ridge=0.5, eval_start=dt.date(2022, 1, 1), min_train=100)

    assert not scored.empty and seen
    for newest_training_match, forecast_from in seen:
        assert pd.Timestamp(newest_training_match) < pd.Timestamp(forecast_from)
    # Refit once per calendar month, and every scored match lies in the window.
    assert len(seen) == scored["date"].map(lambda d: (d.year, d.month)).nunique()
    assert (pd.to_datetime(scored["date"]) >= pd.Timestamp(2022, 1, 1)).all()


def test_a_month_with_too_little_history_is_not_scored():
    matches = _league_matches(seasons=1)
    scored = tuning.walk_forward_rps(matches, "X", xi=0.001, ridge=0.5, eval_start=dt.date(2020, 9, 1), min_train=10_000)
    assert scored.empty and list(scored.columns) == ["date", "rps", "base_rps"]


def test_the_model_beats_guessing_when_the_teams_really_differ():
    matches = _league_matches()
    scored = tuning.walk_forward_rps(matches, "X", xi=0.0005, ridge=0.5, eval_start=dt.date(2022, 9, 1))
    assert len(scored) > 100
    assert scored["rps"].mean() < scored["base_rps"].mean() - 0.02
    assert scored["rps"].between(0, 1).all()


def test_the_grid_is_scored_pair_by_pair_and_ordered_best_first(monkeypatch):
    def fake_scores(matches, league_code, xi, ridge, eval_start, eval_end, min_train):
        # Pretend a longer memory (smaller xi) always scores better, and more smoothing too, but
        # memory matters more.
        return pd.DataFrame(
            {"date": [eval_start] * 3, "rps": [xi * 10 + 0.01 / (1 + ridge)] * 3, "base_rps": [0.25] * 3}
        )

    monkeypatch.setattr(tuning, "walk_forward_rps", fake_scores)
    results = tuning.grid_search(
        pd.DataFrame(), "X", dt.date(2022, 1, 1), xi_grid=(0.002, 0.0005), ridge_grid=(0.0, 2.0)
    )

    assert [(r["xi"], r["ridge"]) for r in results] == [(0.0005, 2.0), (0.0005, 0.0), (0.002, 2.0), (0.002, 0.0)]
    assert all(r["n"] == 3 and r["base_rps"] == 0.25 for r in results)


def test_a_pair_with_nothing_to_score_is_left_out(monkeypatch):
    monkeypatch.setattr(
        tuning, "walk_forward_rps", lambda *a, **k: pd.DataFrame(columns=["date", "rps", "base_rps"])
    )
    assert tuning.grid_search(pd.DataFrame(), "X", dt.date(2022, 1, 1)) == []


def test_the_bootstrap_interval_separates_a_real_gap_from_noise():
    rng = np.random.default_rng(0)
    worse = rng.uniform(0.1, 0.4, 400)
    mean, low, high = tuning.paired_bootstrap(worse - 0.03, worse)
    assert mean == pytest.approx(-0.03) and high < 0  # clearly better

    noise = rng.normal(0, 0.05, 400)
    _, low, high = tuning.paired_bootstrap(worse + noise, worse)
    assert low < 0 < high  # could be chance

    assert tuning.paired_bootstrap(worse, worse) == (0.0, 0.0, 0.0)
