"""The workings panel's note on each side's last six results."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from soccer_predictor.model.results_form_adjustment import ResultsForm


def _panel(home_form, away_form) -> AppTest:
    def app(home_form, away_form):
        from soccer_predictor.dashboard.components import render_prediction_breakdown
        from soccer_predictor.model.markets import MatchPrediction, PredictionBreakdown

        b = PredictionBreakdown(
            1.4, 1.0, 1.1, 1.2, 1.3, 1.0, 1.1, 1.2, 1.3, -0.1, 1.5, 1.2, 100, "2026-10-01T00:00:00", 0.0018,
            home_results_form=home_form, away_results_form=away_form,
        )
        prediction = MatchPrediction(0.5, 0.3, 0.2, 0.5, 0.5, 0.5, [(1, 0, 0.12)], b)
        render_prediction_breakdown(prediction, "Chelsea", "Manchester United")

    return AppTest.from_function(app, default_timeout=30, args=(home_form, away_form)).run()


def _text(at) -> str:
    return " ".join(m.value for m in [*at.markdown, *at.caption])


def test_each_sides_points_and_rating_change_are_shown():
    at = _panel(ResultsForm(points=11, matches=6, shift=0.023), ResultsForm(points=2, matches=6, shift=-0.052))
    assert not at.exception
    text = _text(at)
    assert "Chelsea: **11 points from the last 6 matches**" in text
    assert "Manchester United: **2 points from the last 6 matches**" in text
    assert "attack +2.3%" in text and "attack -5.2%" in text
    assert "goals conceded +5.5%" in text  # 1 / (1 - 0.052) - 1


def test_a_side_without_six_results_says_so():
    text = _text(_panel(ResultsForm(points=9, matches=6, shift=0.0), None))
    assert "Manchester United: fewer than six recent results, so no adjustment." in text


def test_a_league_that_does_not_use_results_has_no_such_note():
    at = _panel(None, None)
    assert not at.exception
    assert "last six results" not in _text(at)
