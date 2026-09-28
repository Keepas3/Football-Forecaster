from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import espn_client, squad
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, update_espn_team_id

LEAGUE = League(
    code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1"
)

ROSTER = [
    espn_client.EspnRosterPlayer(name="Evan Bush", position="Goalkeeper", position_bucket="defense"),
    espn_client.EspnRosterPlayer(name="Bukayo Saka", position="Offence", position_bucket="attack"),
]


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        crew = get_or_create_team(s, "Columbus Crew", "MLS")
        update_espn_team_id(s, crew.id, 183)
        s.commit()
        yield s


def test_finds_squad_for_team_with_espn_id(session, monkeypatch):
    monkeypatch.setattr(espn_client, "fetch_team_roster", lambda *a, **k: (ROSTER, []))
    crew_id = get_or_create_team(session, "Columbus Crew", "MLS").id

    players = squad.fetch_squad_for_team(session, LEAGUE, crew_id)

    assert players is not None
    assert [p.name for p in players] == ["Evan Bush", "Bukayo Saka"]
    assert players[0].position == "Goalkeeper"


def test_returns_none_for_team_without_espn_id(session, monkeypatch):
    monkeypatch.setattr(espn_client, "fetch_team_roster", lambda *a, **k: (ROSTER, []))
    unknown_id = get_or_create_team(session, "Some Other Club", "MLS").id

    assert squad.fetch_squad_for_team(session, LEAGUE, unknown_id) is None


def test_returns_none_when_roster_fetch_has_nothing(session, monkeypatch):
    # espn_client.fetch_team_roster already degrades a RequestException (or
    # any other failure) to ([], []) itself (see test_espn_client.py) --
    # this is what fetch_squad_for_team actually sees at its own boundary.
    monkeypatch.setattr(espn_client, "fetch_team_roster", lambda *a, **k: ([], []))
    crew_id = get_or_create_team(session, "Columbus Crew", "MLS").id

    assert squad.fetch_squad_for_team(session, LEAGUE, crew_id) is None


def test_fetch_team_info_returns_none_for_espn_teams(session):
    crew_id = get_or_create_team(session, "Columbus Crew", "MLS").id
    assert squad.fetch_team_info(session, LEAGUE, crew_id) is None
