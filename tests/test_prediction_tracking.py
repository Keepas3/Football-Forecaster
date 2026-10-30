from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.prediction.tracking import LOCK_GROUPS, compute_prediction_accuracy
from soccer_predictor.storage.models import Base, PredictionRecord
from soccer_predictor.storage.repository import (
    all_prediction_records,
    get_or_create_team,
    upsert_match,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _record(league_code, date, home_id, away_id, predicted_home, predicted_away):
    return PredictionRecord(
        league_code=league_code,
        date=date,
        home_team_id=home_id,
        away_team_id=away_id,
        predicted_home_goals=predicted_home,
        predicted_away_goals=predicted_away,
        snapshotted_at=dt.datetime(2026, 1, 1),
    )


def test_compute_prediction_accuracy_buckets_exact_correct_wrong_and_pending(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    chelsea = get_or_create_team(session, "Chelsea", "EPL")
    leeds = get_or_create_team(session, "Leeds", "EPL")
    everton = get_or_create_team(session, "Everton", "EPL")
    fulham = get_or_create_team(session, "Fulham", "EPL")
    session.commit()

    d1, d2, d3, d4 = (dt.date(2026, 1, i) for i in range(1, 5))

    # Exact: predicted 2-0, actual 2-0.
    session.add(_record("EPL", d1, arsenal.id, villa.id, 2, 0))
    upsert_match(session, "EPL", "2526", d1, arsenal.id, villa.id, 2, 0)

    # Correct outcome, wrong score: predicted home win 2-0, actual home win 1-0.
    session.add(_record("EPL", d2, chelsea.id, leeds.id, 2, 0))
    upsert_match(session, "EPL", "2526", d2, chelsea.id, leeds.id, 1, 0)

    # Wrong: predicted home win, actual away win.
    session.add(_record("EPL", d3, everton.id, fulham.id, 2, 0))
    upsert_match(session, "EPL", "2526", d3, everton.id, fulham.id, 0, 1)

    # Pending: locked in, no result yet.
    session.add(_record("EPL", d4, arsenal.id, chelsea.id, 1, 1))

    session.commit()

    accuracy = compute_prediction_accuracy(session)

    assert accuracy.exact == 1
    assert accuracy.correct_outcome == 1
    assert accuracy.wrong == 1
    assert accuracy.pending == 1
    assert accuracy.graded_total == 3
    assert accuracy.hit_rate == pytest.approx(2 / 3)


def test_compute_prediction_accuracy_draw_counts_as_its_own_outcome(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    session.commit()

    d1 = dt.date(2026, 1, 1)
    session.add(_record("EPL", d1, arsenal.id, villa.id, 1, 1))
    upsert_match(session, "EPL", "2526", d1, arsenal.id, villa.id, 2, 2)
    session.commit()

    accuracy = compute_prediction_accuracy(session)
    assert accuracy.correct_outcome == 1
    assert accuracy.exact == 0
    assert accuracy.wrong == 0


def test_compute_prediction_accuracy_empty_gives_none_hit_rate(session):
    accuracy = compute_prediction_accuracy(session)
    assert accuracy.graded_total == 0
    assert accuracy.hit_rate is None


def test_compute_prediction_accuracy_all_pending_gives_none_hit_rate(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    session.commit()
    session.add(_record("EPL", dt.date(2026, 1, 1), arsenal.id, villa.id, 1, 0))
    session.commit()

    accuracy = compute_prediction_accuracy(session)
    assert accuracy.pending == 1
    assert accuracy.graded_total == 0
    assert accuracy.hit_rate is None


def _two_teams(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    session.commit()
    return arsenal.id, villa.id


def test_locked_predictions_are_looked_up_by_match(session):
    from soccer_predictor.storage.repository import insert_prediction_record, locked_predictions_for_league

    a, b = _two_teams(session)
    day = dt.date(2026, 10, 24)
    insert_prediction_record(session, "EPL", day, a, b, 2, 1, p_home=0.5, p_draw=0.3, p_away=0.2)
    session.commit()

    locked = locked_predictions_for_league(session, "EPL")
    assert locked[(day, a, b)].home_goals == 2 and locked[(day, a, b)].p_home == 0.5
    assert (day, b, a) not in locked and locked_predictions_for_league(session, "LALIGA") == {}


# --- the Track Record by kind of lock --------------------------------------------------------------


def test_accuracy_is_also_kept_per_kind_of_lock(session):
    teams = [get_or_create_team(session, f"Team {i}", "EPL") for i in range(8)]
    session.commit()
    day = dt.date(2026, 10, 3)

    def add(i, basis, predicted, actual):
        home, away = teams[2 * i], teams[2 * i + 1]
        record = _record("EPL", day, home.id, away.id, *predicted)
        record.basis = basis
        record.p_home, record.p_draw, record.p_away = (0.6, 0.25, 0.15) if predicted[0] > predicted[1] else (0.2, 0.3, 0.5)
        session.add(record)
        upsert_match(session, "EPL", "2627", day, home.id, away.id, *actual)

    add(0, "lineup", (2, 0), (1, 0))  # right outcome
    add(1, "lineup", (2, 0), (0, 1))  # wrong
    add(2, "early", (2, 0), (2, 0))  # exact
    add(3, None, (0, 1), (3, 0))  # an older 48-hour lock: wrong
    session.commit()

    accuracy = compute_prediction_accuracy(session)

    assert accuracy.graded_total == 4 and accuracy.scored == 4
    lineup, early, legacy = accuracy.groups["lineup"], accuracy.groups["early"], accuracy.groups["legacy"]
    assert (lineup.correct_outcome, lineup.wrong, lineup.graded_total) == (1, 1, 2)
    assert lineup.hit_rate == pytest.approx(0.5)
    assert (early.exact, early.graded_total) == (1, 1)
    assert (legacy.wrong, legacy.graded_total) == (1, 1)
    assert lineup.scored == 2 and lineup.mean_rps is not None
    assert LOCK_GROUPS == ("lineup", "early", "legacy")


def test_a_pending_lock_counts_in_its_group_too(session):
    a = get_or_create_team(session, "Arsenal", "EPL")
    b = get_or_create_team(session, "Aston Villa", "EPL")
    session.commit()
    record = _record("EPL", dt.date(2026, 11, 1), a.id, b.id, 1, 0)
    record.basis = "lineup"
    session.add(record)
    session.commit()

    accuracy = compute_prediction_accuracy(session)
    assert accuracy.pending == 1 and accuracy.groups["lineup"].pending == 1
    assert accuracy.groups["lineup"].hit_rate is None
