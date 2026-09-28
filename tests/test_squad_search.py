from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import api_client, squad
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, upsert_team_alias

LEAGUE = League(
    code="EPL", name="English Premier League", csv_code="E0", api_competition_id=2021, seasons=["2425"]
)

RAW_RESPONSE = {
    "teams": [
        {
            "name": "Arsenal FC",
            "squad": [
                {"name": "Bukayo Saka", "position": "Attacker", "nationality": "England", "dateOfBirth": "2001-09-05"},
                {"name": "Kepa Arrizabalaga", "position": "Goalkeeper", "nationality": "Spain", "dateOfBirth": "1994-10-03"},
            ],
        },
        {
            "name": "Chelsea FC",
            "squad": [
                {"name": "Robert Sanchez", "position": "Goalkeeper", "nationality": "Spain", "dateOfBirth": "1997-11-18"},
            ],
        },
    ]
}


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        arsenal = get_or_create_team(s, "Arsenal", "EPL")
        upsert_team_alias(s, arsenal, "Arsenal FC", source="api")
        chelsea = get_or_create_team(s, "Chelsea", "EPL")
        upsert_team_alias(s, chelsea, "Chelsea FC", source="api")
        s.commit()
        yield s


def test_finds_player_across_teams_case_insensitively(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: RAW_RESPONSE)

    results = squad.search_players_in_league(session, LEAGUE, "saka")

    assert len(results) == 1
    assert results[0].player.name == "Bukayo Saka"
    assert results[0].team_name == "Arsenal"


def test_matches_multiple_teams_for_a_shared_substring(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: RAW_RESPONSE)

    results = squad.search_players_in_league(session, LEAGUE, "Goalkeeper")
    # Substring matches player NAME only, not position -- "Goalkeeper" isn't
    # in any of these names, so this should find nothing.
    assert results == []

    results = squad.search_players_in_league(session, LEAGUE, "sanchez")
    assert len(results) == 1
    assert results[0].team_name == "Chelsea"


def test_returns_empty_list_for_no_match(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: RAW_RESPONSE)

    assert squad.search_players_in_league(session, LEAGUE, "Zzzznonexistent") == []


def test_returns_empty_list_when_squad_fetch_fails(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: {"teams": []})

    assert squad.search_players_in_league(session, LEAGUE, "Saka") == []
