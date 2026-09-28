"""Builds the full P(home_goals=i, away_goals=j) probability matrix."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from soccer_predictor.model.dixon_coles import tau

DEFAULT_MAX_GOALS = 10


def build_matrix(
    lambda_home: float, lambda_away: float, rho: float, max_goals: int = DEFAULT_MAX_GOALS
) -> np.ndarray:
    """Returns an (max_goals+1) x (max_goals+1) matrix, matrix[i, j] = P(home=i, away=j)."""
    goals = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(goals, lambda_home)
    away_pmf = poisson.pmf(goals, lambda_away)
    matrix = np.outer(home_pmf, away_pmf)

    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    matrix *= tau(home_grid, away_grid, rho)

    return matrix / matrix.sum()
