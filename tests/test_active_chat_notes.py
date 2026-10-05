"""The Notes tab lists saved chat notes across every league: a note is saved
against a team in one league, so a per-league listing made it disappear
whenever the league picker was on another league."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import (
    active_chat_injuries,
    active_form_notes,
    add_form_note,
    add_or_update_chat_injury,
    get_or_create_team,
)

TODAY = dt.date(2026, 10, 5)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _form(session, team_id, expires_on):
    add_form_note(session, team_id, raw_text="r", summary="flat", magnitude=-0.4, affects="both", expires_on=expires_on)


def test_notes_for_teams_in_different_leagues_are_all_listed(session):
    poland_nl = get_or_create_team(session, "Poland", "NL")
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    add_or_update_chat_injury(session, poland_nl.id, "Lewandowski", "attack", 0.7, TODAY + dt.timedelta(weeks=3))
    add_or_update_chat_injury(session, arsenal.id, "Saka", "attack", 0.8, None)
    _form(session, poland_nl.id, TODAY + dt.timedelta(weeks=4))
    session.commit()

    injuries = active_chat_injuries(session, TODAY)
    forms = active_form_notes(session, TODAY)

    assert {(i.player_name, t.league_code, t.canonical_name) for i, t in injuries} == {
        ("Lewandowski", "NL", "Poland"),
        ("Saka", "EPL", "Arsenal"),
    }
    assert [(t.league_code, t.canonical_name) for _, t in forms] == [("NL", "Poland")]


def test_expired_notes_are_left_out_and_the_boundary_day_still_counts(session):
    team = get_or_create_team(session, "Poland", "NL")
    add_or_update_chat_injury(session, team.id, "Back today", "attack", 0.5, TODAY)
    add_or_update_chat_injury(session, team.id, "Back yesterday", "attack", 0.5, TODAY - dt.timedelta(days=1))
    _form(session, team.id, TODAY)
    _form(session, team.id, TODAY - dt.timedelta(days=1))
    session.commit()

    assert [i.player_name for i, _ in active_chat_injuries(session, TODAY)] == ["Back today"]
    assert len(active_form_notes(session, TODAY)) == 1


def test_only_chat_sourced_injuries_are_listed(session):
    from soccer_predictor.storage.models import Injury

    team = get_or_create_team(session, "Poland", "NL")
    session.add(
        Injury(
            team_id=team.id,
            player_name="API injury",
            position="attack",
            importance_weight=0.5,
            source="api",
            note="",
            fetched_at=dt.datetime(2026, 10, 1),
        )
    )
    add_or_update_chat_injury(session, team.id, "Chat injury", "attack", 0.5, None)
    session.commit()

    assert [i.player_name for i, _ in active_chat_injuries(session, TODAY)] == ["Chat injury"]


def test_no_notes_gives_empty_lists(session):
    assert active_chat_injuries(session, TODAY) == []
    assert active_form_notes(session, TODAY) == []


def test_notes_tab_lists_every_leagues_notes_with_league_headings(monkeypatch):
    """Renders _render_active_notes for real (Streamlit's AppTest) against a
    stubbed store, with the league picker conceptually on a different league."""
    from streamlit.testing.v1 import AppTest

    def app():
        import datetime as dt
        from types import SimpleNamespace

        from soccer_predictor.config import League
        from soccer_predictor.dashboard.views import predictor

        team_nl = SimpleNamespace(league_code="NL", canonical_name="Poland")
        team_epl = SimpleNamespace(league_code="EPL", canonical_name="Arsenal")
        injury = SimpleNamespace(
            id=1, player_name="Lewandowski", position="attack", importance_weight=0.7,
            expected_return_date=dt.date(2026, 10, 26),
        )
        form = SimpleNamespace(id=2, summary="Arsenal flat", magnitude=-0.4, affects="both", expires_on=dt.date(2026, 11, 2))

        class FakeScope:
            def __enter__(self):
                return object()

            def __exit__(self, *a):
                return False

        predictor.session_scope = lambda: FakeScope()
        predictor.active_chat_injuries = lambda s, d: [(injury, team_nl)]
        predictor.active_form_notes = lambda s, d: [(form, team_epl)]
        leagues = {
            "EPL": League(code="EPL", name="English Premier League", seasons=["2627"]),
            "NL": League(code="NL", name="UEFA Nations League", seasons=["2627"]),
        }
        predictor._render_active_notes(leagues, can_delete=False)

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)
    assert "English Premier League" in text and "UEFA Nations League" in text
    assert "Poland" in text and "Arsenal" in text
    assert any("Lewandowski" in m.value for m in at.markdown)
    assert not at.button  # read-only: no Delete buttons
