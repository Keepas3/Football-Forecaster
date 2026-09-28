"""Validates the optimizer/likelihood by fitting synthetic data with known params."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from soccer_predictor.model.dixon_coles import fit_league

RNG = np.random.default_rng(42)


def _simulate_matches(
    true_attack: dict[int, float],
    true_defense: dict[int, float],
    true_gamma: float,
    n_rounds: int,
) -> pd.DataFrame:
    """Round-robin simulation from known parameters (rho=0, i.e. plain Poisson)."""
    team_ids = list(true_attack.keys())
    rows = []
    base_date = dt.date(2024, 8, 1)
    day = 0
    for _ in range(n_rounds):
        for home in team_ids:
            for away in team_ids:
                if home == away:
                    continue
                lam_home = true_attack[home] * true_defense[away] * true_gamma
                lam_away = true_attack[away] * true_defense[home]
                rows.append(
                    {
                        "date": base_date + dt.timedelta(days=day),
                        "home_team_id": home,
                        "away_team_id": away,
                        "home_goals": RNG.poisson(lam_home),
                        "away_goals": RNG.poisson(lam_away),
                    }
                )
                day += 1
    return pd.DataFrame(rows)


def test_recovers_known_parameters_from_synthetic_data():
    true_attack = {1: 1.4, 2: 1.0, 3: 0.7, 4: 0.9}
    true_defense = {1: 0.8, 2: 1.0, 3: 1.3, 4: 1.1}
    true_gamma = 1.3

    matches = _simulate_matches(true_attack, true_defense, true_gamma, n_rounds=15)
    params = fit_league(matches, league_code="TEST")

    # Rank order of net strength should match, even if absolute values drift
    # a little due to sampling noise -- that's what actually matters for markets.
    true_rank = sorted(true_attack, key=lambda t: true_attack[t] - true_defense[t], reverse=True)
    fitted_rank = sorted(
        params.attack, key=lambda t: params.attack[t] - params.defense[t], reverse=True
    )
    assert fitted_rank == true_rank
    assert params.home_advantage == pytest.approx(true_gamma, rel=0.25)


def test_fitted_attack_mean_normalized_to_one():
    true_attack = {1: 1.4, 2: 1.0, 3: 0.7}
    true_defense = {1: 0.8, 2: 1.0, 3: 1.3}
    matches = _simulate_matches(true_attack, true_defense, true_gamma=1.2, n_rounds=10)

    params = fit_league(matches, league_code="TEST")
    assert np.mean(list(params.attack.values())) == pytest.approx(1.0, abs=1e-6)
