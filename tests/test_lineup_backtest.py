"""prediction/lineup_backtest.py: scoring the lineup adjustment against results."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from soccer_predictor.model.lineup_adjustment import LineupAbsence
from soccer_predictor.prediction import lineup_backtest as lb

STAR = LineupAbsence("Star", "attack", 0.9)


def _obs(actual, home_absences=(), away_absences=()):
    return lb.Observation(
        "EPL", dt.date(2026, 10, 1), 1.2, 1.0, 1.0, 1.0, 1.3, -0.1, home_absences, away_absences, actual
    )


def test_with_no_factors_the_probabilities_ignore_the_absences():
    with_star = lb.probabilities(_obs(1, home_absences=[STAR]), 0.0, 0.0)
    without = lb.probabilities(_obs(1), 0.0, 0.0)
    assert with_star == pytest.approx(without)
    assert sum(without) == pytest.approx(1.0, abs=1e-6)


def test_a_missing_home_star_moves_probability_from_the_home_side_to_the_away_side():
    plain = lb.probabilities(_obs(1), 0.5, 0.5)
    missing = lb.probabilities(_obs(1, home_absences=[STAR]), 0.5, 0.5)
    assert missing[0] < plain[0] and missing[2] > plain[2]


def test_scoring_a_grid_gives_one_score_per_match_per_factor_pair():
    observations = [_obs(1), _obs(0, home_absences=[STAR]), _obs(-1)]
    scores = lb.score_grid(observations, grid=[(0.0, 0.0), (0.5, 0.5)])
    assert set(scores) == {(0.0, 0.0), (0.5, 0.5)}
    assert all(len(rps) == 3 and len(ll) == 3 for rps, ll in scores.values())


def test_the_lineup_wins_when_the_missing_star_really_does_cost_the_team():
    # Home stars missing and the home team keeps losing: pricing the absence in should score better.
    observations = [_obs(-1, home_absences=[STAR]) for _ in range(30)] + [_obs(1) for _ in range(30)]
    scores = lb.score_grid(observations, grid=[(0.0, 0.0), (0.5, 0.0)])
    assert scores[(0.5, 0.0)][0].mean() < scores[(0.0, 0.0)][0].mean()
    assert lb.best_factors(scores) == (0.5, 0.0)


def test_cross_validation_scores_every_match_once_with_factors_chosen_on_the_other_half():
    observations = [_obs(-1, home_absences=[STAR]) for _ in range(20)] + [_obs(1) for _ in range(20)]
    scores = lb.score_grid(observations, grid=[(0.0, 0.0), (0.5, 0.0)])
    rps, log_loss, chosen = lb.cross_validated_scores(scores)
    assert len(rps) == len(log_loss) == 40 and len(chosen) == 2
    assert rps.mean() == pytest.approx(scores[(0.5, 0.0)][0].mean())  # both halves agree on the better pair


def test_the_bootstrap_interval_brackets_a_consistent_gain():
    baseline = np.full(200, 0.30)
    candidate = baseline - 0.01
    mean, low, high = lb.paired_bootstrap(baseline, candidate)
    assert mean == pytest.approx(-0.01) and low <= mean <= high < 0
