"""Scores the lineup adjustment (model/lineup_adjustment.py) against what really
happened, so its strength is chosen from results instead of guessed.

Each observation is one finished match: the base ratings the model had before
it, who was missing from each side's usual XI, and the result. Every
(attack factor, defense factor) pair is scored by the ranked probability score
(RPS) and log loss of the forecast it would have made; the (0, 0) pair is the
model with no lineup information, so any gain is measured against it.

Picking the best pair from the same matches it is scored on flatters it, so
`cross_validated_scores` also scores each half of the matches with the pair that
was best on the other half -- the number to believe.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from soccer_predictor.model.lineup_adjustment import LineupAbsence, adjust_for_lineup
from soccer_predictor.model.markets import predict_markets
from soccer_predictor.model.scoreline_matrix import build_matrix
from soccer_predictor.prediction.tracking import _log_loss, _rps

ATTACK_FACTORS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
DEFENSE_FACTORS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75)
GRID = tuple((a, d) for a in ATTACK_FACTORS for d in DEFENSE_FACTORS)
NO_LINEUP = (0.0, 0.0)


@dataclass
class Observation:
    league: str
    date: dt.date
    base_home_attack: float
    base_home_defense: float
    base_away_attack: float
    base_away_defense: float
    home_advantage: float
    rho: float
    home_absences: Sequence[LineupAbsence]
    away_absences: Sequence[LineupAbsence]
    actual: int  # 1 home win, 0 draw, -1 away win


def probabilities(obs: Observation, attack_factor: float, defense_factor: float) -> tuple[float, float, float]:
    home_attack, home_defense = adjust_for_lineup(
        obs.base_home_attack, obs.base_home_defense, obs.home_absences, attack_factor, defense_factor
    )
    away_attack, away_defense = adjust_for_lineup(
        obs.base_away_attack, obs.base_away_defense, obs.away_absences, attack_factor, defense_factor
    )
    matrix = build_matrix(home_attack * away_defense * obs.home_advantage, away_attack * home_defense, obs.rho)
    prediction = predict_markets(matrix)
    return prediction.home_win, prediction.draw, prediction.away_win


def score_grid(
    observations: Sequence[Observation], grid: Sequence[tuple[float, float]] = GRID
) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    """{(attack factor, defense factor): (RPS per match, log loss per match)}."""
    scores = {}
    for factors in grid:
        rps, log_loss = [], []
        for obs in observations:
            probs = probabilities(obs, *factors)
            rps.append(_rps(probs, obs.actual))
            log_loss.append(_log_loss(probs, obs.actual))
        scores[factors] = (np.array(rps), np.array(log_loss))
    return scores


def best_factors(scores: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]], rows=None) -> tuple[float, float]:
    """The pair with the lowest mean RPS (over `rows`, an index array, when given)."""
    def mean_rps(item):
        rps = item[1][0]
        return float(rps.mean() if rows is None else rps[rows].mean())

    return min(scores.items(), key=mean_rps)[0]


def cross_validated_scores(
    scores: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]], folds: int = 2
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Per-match (RPS, log loss) when each interleaved fold is scored with the
    factors that were best on all the other folds, and the factors chosen."""
    n = len(next(iter(scores.values()))[0])
    index = np.arange(n)
    rps_out, log_out = np.zeros(n), np.zeros(n)
    chosen = []
    for fold in range(folds):
        held_out = index[index % folds == fold]
        training = index[index % folds != fold]
        factors = best_factors(scores, training)
        chosen.append(factors)
        rps_out[held_out] = scores[factors][0][held_out]
        log_out[held_out] = scores[factors][1][held_out]
    return rps_out, log_out, chosen


def paired_bootstrap(
    baseline: np.ndarray, candidate: np.ndarray, resamples: int = 4000, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, 2.5th and 97.5th percentile) of mean(candidate - baseline) over
    resampled matches; negative means the candidate scores better (lower)."""
    difference = candidate - baseline
    rng = np.random.default_rng(seed)
    means = difference[rng.integers(0, len(difference), size=(resamples, len(difference)))].mean(axis=1)
    return float(difference.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else math.nan
