"""Storage for what a locked prediction was made from (basis, lineup) and the
rule for replacing a lock made by the older 48-hour rule."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from soccer_predictor.storage import db
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import (
    all_prediction_records,
    get_or_create_team,
    insert_prediction_record,
    locked_predictions_for_league,
    prediction_record_for_match,
)

DAY = dt.date(2026, 10, 31)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture
def ids(session):
    a, b = get_or_create_team(session, "Arsenal", "EPL"), get_or_create_team(session, "Aston Villa", "EPL")
    session.commit()
    return a.id, b.id


def test_a_lock_remembers_what_it_was_made_from(session, ids):
    a, b = ids
    assert insert_prediction_record(session, "EPL", DAY, a, b, 2, 1, 0.5, 0.3, 0.2, basis="lineup", lineup_json='{"x": 1}')
    session.commit()

    record = prediction_record_for_match(session, "EPL", DAY, a, b)
    assert (record.basis, record.lineup_json) == ("lineup", '{"x": 1}')
    assert locked_predictions_for_league(session, "EPL")[(DAY, a, b)].basis == "lineup"


def test_an_existing_lock_is_never_overwritten_by_default(session, ids):
    a, b = ids
    insert_prediction_record(session, "EPL", DAY, a, b, 2, 1, 0.5, 0.3, 0.2)
    assert not insert_prediction_record(session, "EPL", DAY, a, b, 0, 3, 0.1, 0.1, 0.8, basis="lineup")
    assert not insert_prediction_record(session, "EPL", DAY, a, b, 0, 3, 0.1, 0.1, 0.8, basis="lineup", replace_older_rule=False)
    session.commit()

    [record] = all_prediction_records(session)
    assert (record.predicted_home_goals, record.basis) == (2, None)


def test_a_lock_from_the_older_rule_can_be_replaced_once(session, ids):
    a, b = ids
    insert_prediction_record(session, "EPL", DAY, a, b, 2, 1, 0.5, 0.3, 0.2)  # basis None: the 48-hour rule

    assert insert_prediction_record(session, "EPL", DAY, a, b, 0, 3, 0.1, 0.1, 0.8, basis="lineup", replace_older_rule=True)
    session.commit()
    [record] = all_prediction_records(session)
    assert (record.predicted_home_goals, record.p_away, record.basis) == (0, 0.8, "lineup")

    # now it is a lineup lock: not replaceable, even when asked
    assert not insert_prediction_record(session, "EPL", DAY, a, b, 1, 1, 0.3, 0.4, 0.3, basis="early", replace_older_rule=True)
    session.commit()
    assert prediction_record_for_match(session, "EPL", DAY, a, b).basis == "lineup"


def test_ensure_prediction_basis_columns_upgrades_an_old_database_and_is_idempotent():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE prediction_records (id INTEGER PRIMARY KEY, league_code VARCHAR)"))

    assert db.ensure_prediction_basis_columns(engine) == ["basis", "lineup_json"]
    assert {"basis", "lineup_json"} <= {c["name"] for c in inspect(engine).get_columns("prediction_records")}
    assert db.ensure_prediction_basis_columns(engine) == []
