from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import (
    fixtures_for_team,
    get_or_create_team,
    matches_for_team,
    next_fixture_per_team,
    team_by_id,
    update_team_crest,
    upsert_fixture,
    upsert_match,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_matches_for_team_includes_home_and_away_newest_first(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    chelsea = get_or_create_team(session, "Chelsea", "EPL")
    session.commit()

    upsert_match(session, "EPL", "2425", dt.date(2025, 1, 1), arsenal.id, villa.id, 2, 1)
    upsert_match(session, "EPL", "2425", dt.date(2025, 2, 1), chelsea.id, arsenal.id, 0, 0)
    upsert_match(session, "EPL", "2425", dt.date(2025, 3, 1), villa.id, chelsea.id, 1, 1)
    session.commit()

    df = matches_for_team(session, arsenal.id)
    assert len(df) == 2
    assert list(df["date"]) == [dt.date(2025, 2, 1), dt.date(2025, 1, 1)]


def test_fixtures_for_team_includes_home_and_away_within_range(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    chelsea = get_or_create_team(session, "Chelsea", "EPL")
    session.commit()

    upsert_fixture(session, "EPL", dt.date(2026, 10, 1), arsenal.id, villa.id, "SCHEDULED")
    upsert_fixture(session, "EPL", dt.date(2026, 10, 8), chelsea.id, arsenal.id, "SCHEDULED")
    upsert_fixture(session, "EPL", dt.date(2026, 10, 15), villa.id, chelsea.id, "SCHEDULED")
    session.commit()

    df = fixtures_for_team(session, arsenal.id, dt.date(2026, 9, 1), dt.date(2026, 12, 1))
    assert len(df) == 2
    assert set(df["date"]) == {dt.date(2026, 10, 1), dt.date(2026, 10, 8)}


def test_fixtures_for_team_respects_date_range(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    session.commit()
    upsert_fixture(session, "EPL", dt.date(2026, 10, 1), arsenal.id, villa.id, "SCHEDULED")
    session.commit()

    df = fixtures_for_team(session, arsenal.id, dt.date(2026, 11, 1), dt.date(2026, 12, 1))
    assert df.empty


def test_update_team_crest_sets_on_first_call(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    session.commit()

    update_team_crest(session, arsenal.id, "https://example.com/arsenal.png")
    session.commit()

    assert team_by_id(session, arsenal.id).crest_url == "https://example.com/arsenal.png"


def test_update_team_crest_never_clobbers_with_falsy_value(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    session.commit()
    update_team_crest(session, arsenal.id, "https://example.com/arsenal.png")
    session.commit()

    update_team_crest(session, arsenal.id, None)
    update_team_crest(session, arsenal.id, "")
    session.commit()

    assert team_by_id(session, arsenal.id).crest_url == "https://example.com/arsenal.png"


def test_team_by_id_returns_none_for_missing(session):
    assert team_by_id(session, 999999) is None


def test_next_fixture_per_team_maps_soonest_opponent(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    chelsea = get_or_create_team(session, "Chelsea", "EPL")
    session.commit()

    upsert_fixture(session, "EPL", dt.date(2026, 10, 8), arsenal.id, chelsea.id, "SCHEDULED")
    upsert_fixture(session, "EPL", dt.date(2026, 10, 1), arsenal.id, villa.id, "SCHEDULED")
    session.commit()

    mapping = next_fixture_per_team(session, "EPL", dt.date(2026, 9, 1))
    assert mapping[arsenal.id] == villa.id  # earlier fixture wins
    assert mapping[villa.id] == arsenal.id
    assert mapping[chelsea.id] == arsenal.id


def test_next_fixture_per_team_ignores_past_fixtures(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    session.commit()
    upsert_fixture(session, "EPL", dt.date(2026, 1, 1), arsenal.id, villa.id, "SCHEDULED")
    session.commit()

    mapping = next_fixture_per_team(session, "EPL", dt.date(2026, 9, 1))
    assert arsenal.id not in mapping
