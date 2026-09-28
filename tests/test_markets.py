from __future__ import annotations

import pytest

from soccer_predictor.model.markets import predict_markets
from soccer_predictor.model.scoreline_matrix import build_matrix


def test_market_probabilities_sum_to_one():
    matrix = build_matrix(lambda_home=1.5, lambda_away=1.1, rho=-0.1)
    prediction = predict_markets(matrix)

    assert prediction.home_win + prediction.draw + prediction.away_win == pytest.approx(1.0)
    assert prediction.over_2_5 + prediction.under_2_5 == pytest.approx(1.0)


def test_stronger_home_side_is_favored():
    matrix = build_matrix(lambda_home=2.2, lambda_away=0.8, rho=-0.1)
    prediction = predict_markets(matrix)

    assert prediction.home_win > prediction.away_win
    assert prediction.home_win > prediction.draw


def test_negative_rho_increases_low_score_draw_probability():
    lambda_home, lambda_away = 1.2, 1.0

    independent = build_matrix(lambda_home, lambda_away, rho=0.0)
    correlated = build_matrix(lambda_home, lambda_away, rho=-0.15)

    # Dixon-Coles' whole point: negative rho pulls extra mass into 0-0/1-1
    # relative to plain independent Poisson.
    assert correlated[0, 0] > independent[0, 0]
    assert correlated[1, 1] > independent[1, 1]


def test_top_scorelines_sorted_descending():
    matrix = build_matrix(lambda_home=1.4, lambda_away=1.2, rho=-0.1)
    prediction = predict_markets(matrix, top_n=5)

    probs = [p for _, _, p in prediction.top_scorelines]
    assert probs == sorted(probs, reverse=True)
    assert len(probs) == 5
