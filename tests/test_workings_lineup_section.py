"""The workings panel's "Starting lineup" step, and the fixture workings reusing a lock's saved XI."""

from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest

from soccer_predictor.dashboard import fixture_workings
from soccer_predictor.model.lineup_adjustment import LineupAbsence
from soccer_predictor.prediction import tracking


def _panel(home_absences, away_absences) -> AppTest:
    def app(home_absences, away_absences):
        from soccer_predictor.dashboard.components import render_prediction_breakdown
        from soccer_predictor.model.markets import MatchPrediction, PredictionBreakdown

        b = PredictionBreakdown(
            1.4, 1.0, 1.1, 1.2, 1.3, 1.0, 1.1, 1.2, 1.3, -0.1, 1.5, 1.2, 100, "2026-10-01T00:00:00", 0.0018,
            home_lineup_absences=home_absences, away_lineup_absences=away_absences,
        )
        prediction = MatchPrediction(0.5, 0.3, 0.2, 0.5, 0.5, 0.5, [(1, 0, 0.12)], b)
        render_prediction_breakdown(prediction, "Arsenal", "Chelsea")

    return AppTest.from_function(app, default_timeout=30, args=(home_absences, away_absences)).run()


def _text(at) -> str:
    return " ".join(m.value for m in [*at.markdown, *at.caption])


def test_the_panel_lists_who_is_missing_from_each_usual_xi():
    at = _panel([LineupAbsence("Bukayo Saka", "attack", 0.54)], [])
    assert not at.exception
    assert "Starting lineup" in _text(at)
    assert "The usual starting XI is playing" in " ".join(w.value for w in at.markdown)
    [table] = [t for t in at.table if "Missing from the usual XI" in t.value.columns]
    assert list(table.value["Missing from the usual XI"]) == ["Bukayo Saka"]
    assert list(table.value["Weight"]) == ["0.54"]


def test_a_side_that_could_not_be_rated_says_injuries_were_used():
    at = _panel([], None)
    assert "injuries used instead" in _text(at)


def test_there_is_no_lineup_step_without_a_lineup():
    at = _panel(None, None)
    assert not at.exception
    assert "Starting lineup" not in _text(at)


def test_the_old_claim_that_the_squad_has_no_effect_is_qualified():
    text = _text(_panel(None, None))
    assert "Once a team's starting XI is announced it takes over" in text


def test_the_saved_xi_comes_back_only_from_a_lineup_lock():
    class Record:
        def __init__(self, basis, lineup_json):
            self.basis, self.lineup_json = basis, lineup_json

    saved = json.dumps({"home": ["A"] * 11, "away": ["B"] * 11})
    assert tracking.stored_lineup(Record("lineup", saved)) == (["A"] * 11, ["B"] * 11)
    assert tracking.stored_lineup(Record("early", None)) == (None, None)
    assert tracking.stored_lineup(Record(None, None)) == (None, None)
    assert tracking.stored_lineup(Record("lineup", "not json")) == (None, None)
    assert tracking.stored_lineup(Record("lineup", "{}")) == (None, None)
    assert tracking.stored_lineup(None) == (None, None)
    assert fixture_workings.stored_lineup is tracking.stored_lineup
