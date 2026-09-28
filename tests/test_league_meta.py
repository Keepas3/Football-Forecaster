from __future__ import annotations

import requests

from soccer_predictor.config import League
from soccer_predictor.ingest import api_client, league_meta

LEAGUE = League(
    code="EPL", name="English Premier League", csv_code="E0", api_competition_id=2021, seasons=["2425"]
)


def test_returns_none_when_api_key_missing(monkeypatch):
    def raise_missing_key(*args, **kwargs):
        raise api_client.MissingApiKey("no key")

    monkeypatch.setattr(api_client, "get", raise_missing_key)
    assert league_meta.fetch_competition_emblem(LEAGUE) is None


def test_extracts_emblem_from_response(monkeypatch):
    monkeypatch.setattr(
        api_client, "get", lambda *args, **kwargs: {"emblem": "https://example.com/epl.png"}
    )
    assert league_meta.fetch_competition_emblem(LEAGUE) == "https://example.com/epl.png"


def test_returns_none_on_http_error(monkeypatch):
    def raise_http_error(*args, **kwargs):
        raise requests.HTTPError("500 server error")

    monkeypatch.setattr(api_client, "get", raise_http_error)
    assert league_meta.fetch_competition_emblem(LEAGUE) is None


def test_returns_none_when_emblem_field_missing(monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *args, **kwargs: {"name": "Premier League"})
    assert league_meta.fetch_competition_emblem(LEAGUE) is None
