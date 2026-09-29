"""Derives betting-style markets from a scoreline probability matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionBreakdown:
    """Every number that fed the Poisson/Dixon-Coles calculation for one
    fixture -- attached by prediction.service.predict_fixture so the
    dashboard can show the actual math, not just its output. `base_*` is the
    fitted rating straight from training; `*_attack`/`*_defense` is after
    this fixture's injury/form-note adjustments (identical to `base_*` if
    neither team has any active notes).
    """

    base_home_attack: float
    base_home_defense: float
    base_away_attack: float
    base_away_defense: float
    home_attack: float
    home_defense: float
    away_attack: float
    away_defense: float
    home_advantage: float  # gamma
    rho: float
    lambda_home: float
    lambda_away: float
    # Provenance of base_*/home_advantage/rho above -- attack/defense are
    # fit from TEAM-level historical match results (final scores) only, never
    # from individual player data, so this is what actually answers "where
    # does this number come from" (see dashboard.components.render_prediction_breakdown).
    n_matches: int  # matches used in the league-wide fit (every team, not just these two)
    fitted_at: str  # ISO timestamp of that fit
    xi: float  # time-decay rate -- higher means older results count for less
    # Each team's current-season attacking output (goals + xG-weighted
    # assists) relative to their league's current average, via
    # ingest/player_importance.py::resolve_current_attack_strength -- 1.0
    # means "right at the league average," None means no coverage/no
    # signal (see model/current_form_adjustment.py). This is what's
    # already folded into *_attack above; kept here too so the dashboard
    # can name the adjustment explicitly rather than leaving it opaque.
    home_xg_relative_strength: float | None = None
    away_xg_relative_strength: float | None = None


@dataclass
class MatchPrediction:
    home_win: float
    draw: float
    away_win: float
    over_2_5: float
    under_2_5: float
    both_teams_to_score: float
    top_scorelines: list[tuple[int, int, float]]  # (home_goals, away_goals, prob), sorted desc
    breakdown: PredictionBreakdown | None = None


def predict_markets(matrix: np.ndarray, top_n: int = 5) -> MatchPrediction:
    n = matrix.shape[0]
    goals = np.arange(n)
    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")

    home_win = float(matrix[home_grid > away_grid].sum())
    draw = float(matrix[home_grid == away_grid].sum())
    away_win = float(matrix[home_grid < away_grid].sum())

    total_goals = home_grid + away_grid
    over_2_5 = float(matrix[total_goals > 2.5].sum())
    under_2_5 = 1.0 - over_2_5

    btts = float(matrix[(home_grid > 0) & (away_grid > 0)].sum())

    flat_indices = np.argsort(matrix, axis=None)[::-1][:top_n]
    top_scorelines = [
        (int(idx // n), int(idx % n), float(matrix.flat[idx])) for idx in flat_indices
    ]

    return MatchPrediction(
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        over_2_5=over_2_5,
        under_2_5=under_2_5,
        both_teams_to_score=btts,
        top_scorelines=top_scorelines,
    )
