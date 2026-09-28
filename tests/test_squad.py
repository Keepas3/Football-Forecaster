from __future__ import annotations

import pytest
import requests
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
            "founded": 1886,
            "venue": "Emirates Stadium",
            "clubColors": "Red / White",
            "address": "75 Drayton Park London N5 1BU",
            "website": "http://www.arsenal.com",
            "coach": {"name": "Mikel Arteta", "nationality": "Spain"},
            "squad": [
                {"name": "Kepa Arrizabalaga", "position": "Goalkeeper", "nationality": "Spain", "dateOfBirth": "1994-10-03"},
                {"name": "Bukayo Saka", "position": "Attacker", "nationality": "England", "dateOfBirth": "2001-09-05"},
            ],
        },
        {
            "name": "Chelsea FC",
            "founded": None,
            "venue": None,
            "clubColors": None,
            "address": None,
            "website": None,
            "coach": {},
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
        s.commit()
        yield s


def test_finds_squad_for_matching_team(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *args, **kwargs: RAW_RESPONSE)
    arsenal_id = get_or_create_team(session, "Arsenal", "EPL").id

    players = squad.fetch_squad_for_team(session, LEAGUE, arsenal_id)

    assert players is not None
    assert [p.name for p in players] == ["Kepa Arrizabalaga", "Bukayo Saka"]
    assert players[1].position == "Attacker"
    assert players[1].nationality == "England"


def test_returns_none_for_team_not_in_response(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *args, **kwargs: RAW_RESPONSE)
    unknown_id = get_or_create_team(session, "Some Other Team", "EPL").id

    assert squad.fetch_squad_for_team(session, LEAGUE, unknown_id) is None


def test_returns_none_when_api_key_missing(session, monkeypatch):
    def raise_missing_key(*args, **kwargs):
        raise api_client.MissingApiKey("no key")

    monkeypatch.setattr(api_client, "get", raise_missing_key)
    arsenal_id = get_or_create_team(session, "Arsenal", "EPL").id

    assert squad.fetch_squad_for_team(session, LEAGUE, arsenal_id) is None


def test_returns_none_on_http_error(session, monkeypatch):
    def raise_http_error(*args, **kwargs):
        raise requests.HTTPError("500 server error")

    monkeypatch.setattr(api_client, "get", raise_http_error)
    arsenal_id = get_or_create_team(session, "Arsenal", "EPL").id

    assert squad.fetch_squad_for_team(session, LEAGUE, arsenal_id) is None


def test_fetch_team_info_returns_club_background(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *args, **kwargs: RAW_RESPONSE)
    arsenal_id = get_or_create_team(session, "Arsenal", "EPL").id

    info = squad.fetch_team_info(session, LEAGUE, arsenal_id)

    assert info is not None
    assert info.founded == 1886
    assert info.venue == "Emirates Stadium"
    assert info.club_colors == "Red / White"
    assert info.coach_name == "Mikel Arteta"
    assert info.coach_nationality == "Spain"


def test_fetch_team_info_nulls_stay_none_not_missing_key(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *args, **kwargs: RAW_RESPONSE)
    chelsea = get_or_create_team(session, "Chelsea", "EPL")
    upsert_team_alias(session, chelsea, "Chelsea FC", source="api")
    session.commit()

    info = squad.fetch_team_info(session, LEAGUE, chelsea.id)

    assert info is not None
    assert info.founded is None
    assert info.venue is None
    assert info.coach_name is None


def test_fetch_team_info_returns_none_for_team_not_in_response(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *args, **kwargs: RAW_RESPONSE)
    unknown_id = get_or_create_team(session, "Some Other Team", "EPL").id

    assert squad.fetch_team_info(session, LEAGUE, unknown_id) is None
