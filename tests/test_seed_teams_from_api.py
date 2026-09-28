from __future__ import annotations

import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import api_client, team_mapper
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import all_alias_texts, teams_for_league

LEAGUE = League(code="UCL", name="UEFA Champions League", api_competition_id=2001, seasons=["2425"])

RAW_RESPONSE = {
    "teams": [
        {"name": "Manchester City FC", "crest": "https://example.com/city.png"},
        {"name": "Inter FC", "crest": None},
    ]
}


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def test_seeds_teams_and_api_aliases_from_response(monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *args, **kwargs: RAW_RESPONSE)
    with _session() as s:
        count = team_mapper.seed_teams_from_api(s, LEAGUE)
        s.commit()

        assert count == 2
        assert set(teams_for_league(s, "UCL").values()) == {"Manchester City FC", "Inter FC"}
        aliases = {text for text, _ in all_alias_texts(s, "api", "UCL")}
        assert aliases == {"Manchester City FC", "Inter FC"}


def test_seeding_twice_is_idempotent(monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *args, **kwargs: RAW_RESPONSE)
    with _session() as s:
        team_mapper.seed_teams_from_api(s, LEAGUE)
        team_mapper.seed_teams_from_api(s, LEAGUE)
        s.commit()

        assert len(teams_for_league(s, "UCL")) == 2


def test_returns_zero_when_api_key_missing(monkeypatch):
    def raise_missing_key(*args, **kwargs):
        raise api_client.MissingApiKey("no key")

    monkeypatch.setattr(api_client, "get", raise_missing_key)
    with _session() as s:
        assert team_mapper.seed_teams_from_api(s, LEAGUE) == 0


def test_returns_zero_on_http_error(monkeypatch):
    def raise_http_error(*args, **kwargs):
        raise requests.HTTPError("500 server error")

    monkeypatch.setattr(api_client, "get", raise_http_error)
    with _session() as s:
        assert team_mapper.seed_teams_from_api(s, LEAGUE) == 0
