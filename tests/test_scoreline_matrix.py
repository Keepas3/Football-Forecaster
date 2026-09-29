from __future__ import annotations

import numpy as np
import pytest

from soccer_predictor.model.scoreline_matrix import DEFAULT_DISPERSION, build_matrix, marginal_pmf


def test_matrix_sums_to_one_with_default_dispersion():
    matrix = build_matrix(lambda_home=1.5, lambda_away=1.1, rho=-0.1)
    assert matrix.sum() == pytest.approx(1.0)


def test_dispersion_none_reproduces_exact_prior_pure_poisson_behavior():
    # Regression guard: dispersion=None must give the exact same matrix this
    # function always produced before overdispersion was added.
    from scipy.stats import poisson

    from soccer_predictor.model.dixon_coles import tau

    lambda_home, lambda_away, rho = 1.4, 1.0, -0.1
    matrix = build_matrix(lambda_home, lambda_away, rho, dispersion=None)

    max_goals = 10
    goals = np.arange(max_goals + 1)
    expected = np.outer(poisson.pmf(goals, lambda_home), poisson.pmf(goals, lambda_away))
    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    expected *= tau(home_grid, away_grid, rho)
    expected /= expected.sum()

    assert matrix == pytest.approx(expected)


def test_default_dispersion_increases_blowout_probability():
    lambda_home, lambda_away, rho = 2.2, 0.6, -0.1

    poisson_matrix = build_matrix(lambda_home, lambda_away, rho, dispersion=None)
    overdispersed_matrix = build_matrix(lambda_home, lambda_away, rho, dispersion=DEFAULT_DISPERSION)

    max_goals = poisson_matrix.shape[0] - 1
    goals = np.arange(max_goals + 1)
    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    blowout_mask = (home_grid - away_grid) >= 4  # e.g. 4-0, 5-0, 5-1, 6-0...

    # Total probability mass on a clear blowout margin (home wins by 4+)
    # should be MORE likely under the fatter-tailed overdispersed model
    # than under plain Poisson -- a single cell can wobble either way as
    # mass redistributes, but the aggregate tail is the real claim here.
    assert overdispersed_matrix[blowout_mask].sum() > poisson_matrix[blowout_mask].sum()
    assert overdispersed_matrix[5, 0] > poisson_matrix[5, 0]


def test_overdispersion_preserves_mean_goals_within_tolerance():
    lambda_home, lambda_away, rho = 1.8, 0.9, 0.0

    matrix = build_matrix(lambda_home, lambda_away, rho, dispersion=DEFAULT_DISPERSION)
    max_goals = matrix.shape[0] - 1
    goals = np.arange(max_goals + 1)

    home_marginal = matrix.sum(axis=1)
    away_marginal = matrix.sum(axis=0)
    mean_home = float((home_marginal * goals).sum())
    mean_away = float((away_marginal * goals).sum())

    # tau() nudges the mean slightly off lambda -- same as it already does
    # for the pure-Poisson case -- so this checks "still close," not exact.
    assert mean_home == pytest.approx(lambda_home, abs=0.05)
    assert mean_away == pytest.approx(lambda_away, abs=0.05)


def test_marginal_pmf_matches_poisson_when_dispersion_none():
    from scipy.stats import poisson

    goals = np.arange(11)
    assert marginal_pmf(goals, 1.7, dispersion=None) == pytest.approx(poisson.pmf(goals, 1.7))


def test_marginal_pmf_sums_to_one():
    goals = np.arange(200)  # generous range so the NB tail is fully captured
    pmf = marginal_pmf(goals, lam=1.5, dispersion=DEFAULT_DISPERSION)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-6)
