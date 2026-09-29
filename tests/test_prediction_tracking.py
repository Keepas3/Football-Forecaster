from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.prediction.tracking import compute_prediction_accuracy, snapshot_upcoming_predictions
from soccer_predictor.storage.models import Base, PredictionRecord
from soccer_predictor.storage.repository import (
    all_prediction_records,
    get_or_create_team,
    upsert_fixture,
    upsert_match,
)


class _League:
    """Minimal stand-in -- snapshot_upcoming_predictions only ever reads
    `.code`, never any other League field."""

    def __init__(self, code: str):
        self.code = code


LEAGUE = _League("EPL")


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _params(home_id: int, away_id: int, home_attack=1.4, home_defense=1.0, away_attack=0.9, away_defense=1.2):
    return DixonColesParams(
        league_code="EPL",
        attack={home_id: home_attack, away_id: away_attack},
        defense={home_id: home_defense, away_id: away_defense},
        home_advantage=1.3,
        rho=-0.05,
        xi=0.0018,
        fitted_at="2026-09-27T00:00:00",
        n_matches=1000,
    )


def test_snapshot_creates_one_record_per_upcoming_fixture(session, monkeypatch):
    monkeypatch.setattr(
        "soccer_predictor.prediction.service.resolve_current_attack_strength", lambda *a, **k: None
    )
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    session.commit()

    fixture_date = dt.date.today() + dt.timedelta(days=7)
    upsert_fixture(session, "EPL", fixture_date, arsenal.id, villa.id, "SCHEDULED")
    session.commit()

    created = snapshot_upcoming_predictions(session, LEAGUE, _params(arsenal.id, villa.id))
    session.commit()

    assert created == 1
    records = all_prediction_records(session)
    assert len(records) == 1
    record = records[0]
    assert record.league_code == "EPL"
    assert record.date == fixture_date
    assert record.home_team_id == arsenal.id
    assert record.away_team_id == villa.id
    assert isinstance(record.predicted_home_goals, int)
    assert isinstance(record.predicted_away_goals, int)


def test_snapshot_is_idempotent_and_never_overwrites(session, monkeypatch):
    monkeypatch.setattr(
        "soccer_predictor.prediction.service.resolve_current_attack_strength", lambda *a, **k: None
    )
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    session.commit()

    fixture_date = dt.date.today() + dt.timedelta(days=7)
    upsert_fixture(session, "EPL", fixture_date, arsenal.id, villa.id, "SCHEDULED")
    session.commit()

    first_params = _params(arsenal.id, villa.id, home_attack=1.4, away_attack=0.9)
    snapshot_upcoming_predictions(session, LEAGUE, first_params)
    session.commit()
    original = all_prediction_records(session)[0]
    original_prediction = (original.predicted_home_goals, original.predicted_away_goals)

    # A wildly different "retrained" model shouldn't touch the locked-in
    # snapshot -- that's the whole point of a forward snapshot.
    retrained_params = _params(arsenal.id, villa.id, home_attack=0.3, away_attack=2.5)
    created_second_pass = snapshot_upcoming_predictions(session, LEAGUE, retrained_params)
    session.commit()

    assert created_second_pass == 0
    records = all_prediction_records(session)
    assert len(records) == 1
    assert (records[0].predicted_home_goals, records[0].predicted_away_goals) == original_prediction


def test_snapshot_returns_zero_when_no_upcoming_fixtures(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    session.commit()

    created = snapshot_upcoming_predictions(session, LEAGUE, _params(arsenal.id, villa.id))
    assert created == 0
    assert all_prediction_records(session) == []


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
