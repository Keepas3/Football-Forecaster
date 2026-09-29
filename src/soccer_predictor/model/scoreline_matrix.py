"""Builds the full P(home_goals=i, away_goals=j) probability matrix."""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson

from soccer_predictor.model.dixon_coles import tau

DEFAULT_MAX_GOALS = 10

# Real football scorelines are overdispersed relative to a plain Poisson
# (which pins variance == mean by construction) -- a pure-Poisson model
# systematically over-concentrates probability near the expected scoreline
# and under-predicts genuine blowout margins (e.g. 5-0). DEFAULT_DISPERSION
# switches the per-side goal distribution to a negative binomial (NB2 form)
# with the SAME mean (lambda) but variance lambda + lambda^2/dispersion --
# the extra lambda^2 term fattens the tail toward higher margins as lambda
# grows, without touching the fitted attack/defense/home_advantage/rho
# values or the training pipeline at all. A documented, tunable heuristic
# (like injury_adjustment.py's ATTACK_IMPACT_FACTOR), not empirically fit --
# lower means MORE overdispersion (fatter tail), None recovers plain Poisson
# exactly. Worth revisiting once prediction/tracking.py's own hit-rate data
# has accumulated enough graded predictions to check empirically.
DEFAULT_DISPERSION = 8.0


def marginal_pmf(goals: np.ndarray, lam: float, dispersion: float | None) -> np.ndarray:
    """P(goals=k) for one side (home or away) -- plain Poisson if
    `dispersion` is None, otherwise the overdispersed negative binomial
    described above. Public (not `_`-prefixed) so
    dashboard/components.py::render_prediction_breakdown can redo this
    exact same per-scoreline math for its worked example, rather than
    duplicating the formula and risking it silently drifting out of sync
    with what build_matrix actually computes.
    """
    if dispersion is None:
        return poisson.pmf(goals, lam)
    # NB2 parameterization: mean = lam, variance = lam + lam^2/dispersion.
    p = dispersion / (dispersion + lam)
    return nbinom.pmf(goals, dispersion, p)


def build_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float,
    max_goals: int = DEFAULT_MAX_GOALS,
    dispersion: float | None = DEFAULT_DISPERSION,
) -> np.ndarray:
    """Returns an (max_goals+1) x (max_goals+1) matrix, matrix[i, j] = P(home=i, away=j)."""
    goals = np.arange(max_goals + 1)
    home_pmf = marginal_pmf(goals, lambda_home, dispersion)
    away_pmf = marginal_pmf(goals, lambda_away, dispersion)
    matrix = np.outer(home_pmf, away_pmf)

    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    matrix *= tau(home_grid, away_grid, rho)

    return matrix / matrix.sum()
