from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import (
    add_form_note,
    add_or_update_chat_injury,
    delete_form_note,
    delete_injury,
    form_notes_for_team,
    get_or_create_team,
    injuries_for_team,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_add_or_update_chat_injury_upserts_not_duplicates(session):
    team = get_or_create_team(session, "Arsenal", "EPL")
    session.commit()

    add_or_update_chat_injury(
        session, team.id, "Bukayo Saka", "attack", 0.8, dt.date(2026, 10, 20)
    )
    session.commit()
    assert len(injuries_for_team(session, team.id)) == 1

    # Corrected estimate for the same player should update, not duplicate.
    add_or_update_chat_injury(
        session, team.id, "bukayo saka", "attack", 0.9, dt.date(2026, 10, 27)
    )
    session.commit()

    rows = injuries_for_team(session, team.id)
    assert len(rows) == 1
    assert rows[0].importance_weight == 0.9
    assert rows[0].expected_return_date == dt.date(2026, 10, 27)


def test_delete_injury_removes_row(session):
    team = get_or_create_team(session, "Arsenal", "EPL")
    session.commit()
    injury = add_or_update_chat_injury(
        session, team.id, "Bukayo Saka", "attack", 0.8, None
    )
    session.commit()

    delete_injury(session, injury.id)
    session.commit()
    assert injuries_for_team(session, team.id) == []


def test_add_and_delete_form_note(session):
    team = get_or_create_team(session, "Arsenal", "EPL")
    session.commit()

    note = add_form_note(
        session,
        team.id,
        raw_text="team's been flat since the manager change",
        summary="Poor recent form",
        magnitude=-0.4,
        affects="both",
        expires_on=dt.date(2026, 11, 1),
    )
    session.commit()
    assert len(form_notes_for_team(session, team.id)) == 1

    delete_form_note(session, note.id)
    session.commit()
    assert form_notes_for_team(session, team.id) == []
