"""Walk-forward check of how a league's ratings are fit.

Which memory length (`League.rating_decay_xi`) and smoothing
(`League.rating_ridge`) a competition needs depends on how often its teams play
each other: the default one-year memory suits a league whose clubs meet twice a
season, but forgets almost everything about a team that plays a handful of
matches a year. So each competition's numbers are chosen the way a forecast
would be judged -- predict each stretch of matches using only what was known
before it -- and written into config/leagues.yaml with the scores that justified
them. scripts/tune_ratings.py runs this for a league.

The score is the ranked probability score (RPS) of the predicted
home/draw/away probabilities, lower being better, set against the same score for
always guessing the usual win/draw/loss rates (NO_SKILL_PROBS): a setting that
cannot beat that guess is worse than no model.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from soccer_predictor.model.dixon_coles import fit_league
from soccer_predictor.model.markets import predict_markets
from soccer_predictor.model.scoreline_matrix import build_matrix
from soccer_predictor.prediction.tracking import NO_SKILL_PROBS, _outcome_sign, _rps

DEFAULT_XI_GRID = (0.0018, 0.001, 0.0005, 0.0003, 0.0002)
DEFAULT_RIDGE_GRID = (0.0, 0.15, 0.5, 1.5, 4.0)
# Fewer matches than this and a fit says little; those months are not scored.
DEFAULT_MIN_TRAIN = 100


def walk_forward_rps(
    matches: pd.DataFrame,
    league_code: str,
    xi: float,
    ridge: float,
    eval_start: dt.date,
    eval_end: dt.date | None = None,
    min_train: int = DEFAULT_MIN_TRAIN,
) -> pd.DataFrame:
    """One row per match played in [eval_start, eval_end]: its `date`, the RPS of
    the model's forecast (`rps`) and of the no-skill guess (`base_rps`).

    The ratings are refit at the start of every calendar month on the matches
    played before it and used for that month's matches -- what the daily refresh
    does, only coarser -- so a forecast never sees its own result. `matches`
    needs the columns of storage.repository.matches_for_league.
    """
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    start = pd.Timestamp(eval_start)
    end = pd.Timestamp(eval_end) if eval_end is not None else frame["date"].max()
    window = frame[(frame["date"] >= start) & (frame["date"] <= end)]

    rows = []
    for month, block in window.groupby(window["date"].dt.to_period("M")):
        month_start = month.start_time
        train = frame[frame["date"] < month_start]
        if len(train) < min_train:
            continue
        params = fit_league(train, league_code, xi=xi, ridge=ridge, as_of=month_start)
        for match in block.itertuples():
            home_goals_expected = (
                params.attack[match.home_team_id] * params.defense[match.away_team_id] * params.home_advantage
            )
            away_goals_expected = params.attack[match.away_team_id] * params.defense[match.home_team_id]
            prediction = predict_markets(build_matrix(home_goals_expected, away_goals_expected, params.rho), top_n=1)
            actual = _outcome_sign(match.home_goals, match.away_goals)
            rows.append(
                {
                    "date": match.date,
                    "rps": _rps((prediction.home_win, prediction.draw, prediction.away_win), actual),
                    "base_rps": _rps(NO_SKILL_PROBS, actual),
                }
            )
    return pd.DataFrame(rows, columns=["date", "rps", "base_rps"])


def grid_search(
    matches: pd.DataFrame,
    league_code: str,
    eval_start: dt.date,
    eval_end: dt.date | None = None,
    xi_grid: tuple[float, ...] = DEFAULT_XI_GRID,
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID,
    min_train: int = DEFAULT_MIN_TRAIN,
) -> list[dict]:
    """Every (xi, ridge) pair's mean RPS over the window, best first. Each entry:
    xi, ridge, rps, base_rps (the no-skill guess, the same for every pair) and
    n, the number of matches scored."""
    results = []
    for xi in xi_grid:
        for ridge in ridge_grid:
            scored = walk_forward_rps(matches, league_code, xi, ridge, eval_start, eval_end, min_train)
            if scored.empty:
                continue
            results.append(
                {
                    "xi": xi,
                    "ridge": ridge,
                    "rps": float(scored["rps"].mean()),
                    "base_rps": float(scored["base_rps"].mean()),
                    "n": len(scored),
                }
            )
    return sorted(results, key=lambda entry: entry["rps"])


def paired_bootstrap(better: np.ndarray, worse: np.ndarray, resamples: int = 4000, seed: int = 0):
    """Mean of (better - worse) per-match scores and its 95% bootstrap interval.
    Negative means `better` really scores lower (better); an interval that
    straddles 0 means the gap could be chance."""
    difference = np.asarray(better) - np.asarray(worse)
    rng = np.random.default_rng(seed)
    means = rng.choice(difference, size=(resamples, len(difference))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(difference.mean()), float(low), float(high)
