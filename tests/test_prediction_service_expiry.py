from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.prediction.service import get_injuries_for_team
from soccer_predictor.storage.models import Base, Injury
from soccer_predictor.storage.repository import get_or_create_team


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _add_chat_injury(session, team_id, expected_return_date):
    session.add(
        Injury(
            team_id=team_id,
            player_name="Test Player",
            position="attack",
            importance_weight=0.8,
            source="chat",
            note="",
            fetched_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
            expected_return_date=expected_return_date,
        )
    )
    session.commit()


def test_chat_injury_active_before_return_date(session):
    team = get_or_create_team(session, "Arsenal", "EPL")
    session.commit()
    _add_chat_injury(session, team.id, expected_return_date=dt.date(2026, 10, 20))

    result = get_injuries_for_team(session, team.id, "Arsenal", as_of=dt.date(2026, 10, 1))
    assert len(result.entries) == 1
    assert result.sources == ["chat"]


def test_chat_injury_excluded_after_return_date(session):
    team = get_or_create_team(session, "Arsenal", "EPL")
    session.commit()
    _add_chat_injury(session, team.id, expected_return_date=dt.date(2026, 10, 20))

    result = get_injuries_for_team(session, team.id, "Arsenal", as_of=dt.date(2026, 11, 1))
    assert result.entries == []


def test_chat_injury_with_no_return_date_never_expires(session):
    team = get_or_create_team(session, "Arsenal", "EPL")
    session.commit()
    _add_chat_injury(session, team.id, expected_return_date=None)

    far_future = dt.date(2099, 1, 1)
    result = get_injuries_for_team(session, team.id, "Arsenal", as_of=far_future)
    assert len(result.entries) == 1
