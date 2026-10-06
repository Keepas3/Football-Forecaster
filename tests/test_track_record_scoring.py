"""Track Record: grading the most likely OUTCOME (not just the likeliest
scoreline), saved probabilities, RPS / log-loss scoring, and the backfill."""

from __future__ import annotations

import datetime as dt
import math

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.prediction import tracking
from soccer_predictor.prediction.tracking import (
    NO_SKILL_PROBS,
    _log_loss,
    _rps,
    backfill_prediction_probabilities,
    compute_prediction_accuracy,
    most_likely_outcome,
    snapshot_upcoming_predictions,
)
from soccer_predictor.storage import db
from soccer_predictor.storage.models import Base, PredictionRecord
from soccer_predictor.storage.repository import (
    all_prediction_records,
    get_or_create_team,
    insert_prediction_record,
    upsert_fixture,
    upsert_match,
)


class _League:
    def __init__(self, code):
        self.code = code


LEAGUE = _League("EPL")
PAST = dt.date(2026, 10, 1)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture(autouse=True)
def _no_external_stats(monkeypatch):
    monkeypatch.setattr("soccer_predictor.prediction.service.resolve_current_attack_strength", lambda *a, **k: None)


def _teams(session):
    a = get_or_create_team(session, "France", "EPL").id
    b = get_or_create_team(session, "Belgium", "EPL").id
    session.commit()
    return a, b


def _params(a, b, home_attack=1.4, away_attack=0.9):
    return DixonColesParams(
        league_code="EPL", attack={a: home_attack, b: away_attack}, defense={a: 1.0, b: 1.2},
        home_advantage=1.3, rho=-0.05, xi=0.0018, fitted_at="2026-09-27T00:00:00", n_matches=1000,
    )


def _record(session, a, b, scoreline, probs=None, date=PAST):
    insert_prediction_record(
        session, "EPL", date, a, b, scoreline[0], scoreline[1],
        p_home=probs[0] if probs else None, p_draw=probs[1] if probs else None, p_away=probs[2] if probs else None,
    )
    session.commit()


# --- most likely outcome ---------------------------------------------------------------------------


def test_the_most_likely_outcome_comes_from_the_probabilities_when_saved():
    record = PredictionRecord(predicted_home_goals=0, predicted_away_goals=0, p_home=0.45, p_draw=0.28, p_away=0.27)
    assert most_likely_outcome(record) == 1  # a home win, even though the top scoreline is a 0-0 draw


def test_older_records_fall_back_to_the_scoreline():
    old = PredictionRecord(predicted_home_goals=0, predicted_away_goals=2)
    assert most_likely_outcome(old) == -1
    assert most_likely_outcome(PredictionRecord(predicted_home_goals=1, predicted_away_goals=1)) == 0


def test_the_france_belgium_case_is_now_a_correct_outcome(session):
    # The model gave France 45% and the top scoreline was 0-0; France won 4-1.
    a, b = _teams(session)
    _record(session, a, b, (0, 0), probs=(0.45, 0.28, 0.27))
    upsert_match(session, "EPL", "2627", PAST, a, b, 4, 1)
    session.commit()

    result = compute_prediction_accuracy(session)

    assert (result.exact, result.correct_outcome, result.wrong) == (0, 1, 0)


def test_the_same_match_with_no_saved_probabilities_is_still_graded_on_the_scoreline(session):
    a, b = _teams(session)
    _record(session, a, b, (0, 0))  # legacy record
    upsert_match(session, "EPL", "2627", PAST, a, b, 4, 1)
    session.commit()

    result = compute_prediction_accuracy(session)

    assert (result.exact, result.correct_outcome, result.wrong) == (0, 0, 1)
    assert result.scored == 0 and result.mean_rps is None  # nothing to score without probabilities


def test_an_exact_scoreline_is_still_exact(session):
    a, b = _teams(session)
    _record(session, a, b, (2, 1), probs=(0.5, 0.25, 0.25))
    upsert_match(session, "EPL", "2627", PAST, a, b, 2, 1)
    session.commit()
    assert compute_prediction_accuracy(session).exact == 1


def test_a_wrong_favourite_is_wrong(session):
    a, b = _teams(session)
    _record(session, a, b, (1, 0), probs=(0.55, 0.25, 0.20))
    upsert_match(session, "EPL", "2627", PAST, a, b, 0, 3)
    session.commit()
    assert compute_prediction_accuracy(session).wrong == 1


def test_unplayed_records_stay_pending(session):
    a, b = _teams(session)
    _record(session, a, b, (1, 0), probs=(0.5, 0.25, 0.25), date=dt.date.today() + dt.timedelta(days=3))
    assert compute_prediction_accuracy(session).pending == 1


# --- RPS and log loss -----------------------------------------------------------------------------------


def test_rps_is_zero_for_a_perfect_forecast_and_worst_for_a_confident_miss():
    assert _rps((1.0, 0.0, 0.0), 1) == 0.0
    assert _rps((0.0, 0.0, 1.0), 1) == 1.0  # certain of an away win, home won


def test_rps_punishes_a_near_miss_less_than_a_far_miss():
    # Truth: home win. Predicting a draw is closer than predicting an away win.
    assert _rps((0.1, 0.8, 0.1), 1) < _rps((0.1, 0.1, 0.8), 1)


def test_rps_known_value():
    probs = (0.5, 0.3, 0.2)  # actual draw: cum pred .5/.8 vs cum obs 0/1 -> .5*(.25+.04)
    assert _rps(probs, 0) == pytest.approx(0.145)


def test_log_loss_known_values():
    assert _log_loss((0.5, 0.3, 0.2), 0) == pytest.approx(-math.log(0.3))
    assert _log_loss((0.0, 0.0, 1.0), 1) > 15  # never infinite, but enormous


def test_scores_are_averaged_and_compared_with_the_no_skill_guess(session):
    a, b = _teams(session)
    c = get_or_create_team(session, "Italy", "EPL").id
    d = get_or_create_team(session, "Spain", "EPL").id
    session.commit()
    _record(session, a, b, (2, 0), probs=(0.7, 0.2, 0.1))
    _record(session, c, d, (1, 1), probs=(0.3, 0.4, 0.3), date=PAST + dt.timedelta(days=1))
    upsert_match(session, "EPL", "2627", PAST, a, b, 2, 0)  # home win
    upsert_match(session, "EPL", "2627", PAST + dt.timedelta(days=1), c, d, 0, 2)  # away win
    session.commit()

    result = compute_prediction_accuracy(session)

    expected_rps = (_rps((0.7, 0.2, 0.1), 1) + _rps((0.3, 0.4, 0.3), -1)) / 2
    expected_base = (_rps(NO_SKILL_PROBS, 1) + _rps(NO_SKILL_PROBS, -1)) / 2
    assert result.scored == 2
    assert result.mean_rps == pytest.approx(expected_rps)
    assert result.baseline_rps == pytest.approx(expected_base)
    assert result.mean_log_loss == pytest.approx((-math.log(0.7) - math.log(0.3)) / 2)


def test_no_skill_guess_is_a_valid_distribution():
    assert sum(NO_SKILL_PROBS) == pytest.approx(1.0)


# --- snapshots store the probabilities ---------------------------------------------------------------------


def test_new_snapshots_save_the_probabilities(session):
    a, b = _teams(session)
    upsert_fixture(session, "EPL", dt.date.today() + dt.timedelta(days=5), a, b, "SCHEDULED")
    session.commit()

    snapshot_upcoming_predictions(session, LEAGUE, _params(a, b))
    session.commit()

    record = all_prediction_records(session)[0]
    assert record.p_home is not None
    assert record.p_home + record.p_draw + record.p_away == pytest.approx(1.0)
    assert record.p_home > record.p_away  # the stronger home side is the favourite


# --- backfill ----------------------------------------------------------------------------------------------------


def test_backfill_fills_future_records_without_touching_the_locked_scoreline(session):
    a, b = _teams(session)
    future = dt.date.today() + dt.timedelta(days=4)
    _record(session, a, b, (0, 0), date=future)  # legacy record: no probabilities

    filled = backfill_prediction_probabilities(session, LEAGUE, _params(a, b))
    session.commit()

    record = all_prediction_records(session)[0]
    assert filled == 1
    assert record.p_home + record.p_draw + record.p_away == pytest.approx(1.0)
    assert (record.predicted_home_goals, record.predicted_away_goals) == (0, 0)  # never rewritten


def test_backfill_never_touches_matches_already_played(session):
    # Computing a probability after the result is known would mean scoring a
    # model that has seen the answer -- those stay without probabilities.
    a, b = _teams(session)
    _record(session, a, b, (1, 0), date=dt.date.today() - dt.timedelta(days=2))

    assert backfill_prediction_probabilities(session, LEAGUE, _params(a, b)) == 0
    assert all_prediction_records(session)[0].p_home is None


def test_backfill_leaves_records_that_already_have_probabilities(session):
    a, b = _teams(session)
    _record(session, a, b, (1, 0), probs=(0.5, 0.3, 0.2), date=dt.date.today() + dt.timedelta(days=2))

    assert backfill_prediction_probabilities(session, LEAGUE, _params(a, b)) == 0
    assert all_prediction_records(session)[0].p_home == 0.5


# --- schema upgrade ---------------------------------------------------------------------------------------------------


def test_ensure_prediction_probability_columns_upgrades_an_old_database_and_is_idempotent():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE prediction_records (id INTEGER PRIMARY KEY, league_code VARCHAR)"))

    assert db.ensure_prediction_probability_columns(engine) == ["p_home", "p_draw", "p_away"]
    columns = {c["name"] for c in inspect(engine).get_columns("prediction_records")}
    assert {"p_home", "p_draw", "p_away"} <= columns
    assert db.ensure_prediction_probability_columns(engine) == []


def test_backfill_skips_a_match_dated_today_that_already_has_a_result(session):
    a, b = _teams(session)
    _record(session, a, b, (1, 0), date=dt.date.today())
    upsert_match(session, "EPL", "2627", dt.date.today(), a, b, 3, 0)  # already played this morning
    session.commit()

    assert backfill_prediction_probabilities(session, LEAGUE, _params(a, b)) == 0
    assert all_prediction_records(session)[0].p_home is None


# --- never after kickoff ---------------------------------------------------------------------------------------------


def _fixture_at(session, a, b, kickoff):
    upsert_fixture(session, "EPL", kickoff.date(), a, b, "SCHEDULED", kickoff_utc=kickoff)
    session.commit()


def test_a_match_already_under_way_is_not_snapshotted(session):
    a, b = _teams(session)
    _fixture_at(session, a, b, dt.datetime.now(dt.UTC).replace(tzinfo=None) - dt.timedelta(minutes=20))

    assert snapshot_upcoming_predictions(session, LEAGUE, _params(a, b)) == 0
    assert all_prediction_records(session) == []


def test_a_match_kicking_off_later_today_is_snapshotted(session):
    a, b = _teams(session)
    _fixture_at(session, a, b, dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(hours=3))

    assert snapshot_upcoming_predictions(session, LEAGUE, _params(a, b)) == 1


def test_backfill_skips_a_match_under_way_but_fills_one_that_has_not_started(session):
    a, b = _teams(session)
    c = get_or_create_team(session, "Italy", "EPL").id
    d = get_or_create_team(session, "Spain", "EPL").id
    session.commit()
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    _fixture_at(session, a, b, now - dt.timedelta(minutes=30))  # in progress
    _fixture_at(session, c, d, now + dt.timedelta(hours=2))  # not started
    _record(session, a, b, (1, 0), date=now.date())
    _record(session, c, d, (1, 0), date=now.date())
    params = _params(a, b)
    params.attack.update({c: 1.2, d: 0.8})
    params.defense.update({c: 1.0, d: 1.1})

    assert backfill_prediction_probabilities(session, LEAGUE, params) == 1
    by_home = {r.home_team_id: r for r in all_prediction_records(session)}
    assert by_home[a].p_home is None and by_home[c].p_home is not None


def test_backfill_skips_todays_match_when_its_kickoff_is_unknown(session):
    a, b = _teams(session)
    _record(session, a, b, (1, 0), date=dt.date.today())  # no fixture row, so no kickoff time

    assert backfill_prediction_probabilities(session, LEAGUE, _params(a, b)) == 0
