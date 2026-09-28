"""Regression tests for injuries.sync_injuries_to_db: caching
Team.api_football_team_id, resolving a real position via the squad list
(instead of trusting the injuries endpoint's likely-absent position field),
and computing a real per-player importance_weight from historical stats
(instead of a flat default for every API-sourced injury).
"""

from __future__ import annotations

import pytest
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import api_football_client, injuries, player_stats, squad, understat_client
from soccer_predictor.ingest.player_importance import DEFAULT_API_IMPORTANCE_WEIGHT
from soccer_predictor.ingest.player_stats import HistoricalPlayerStats
from soccer_predictor.ingest.squad import SquadPlayer
from soccer_predictor.ingest.understat_client import UnderstatPlayerStats
from soccer_predictor.storage.models import Base, Team
from soccer_predictor.storage.repository import get_or_create_team, injuries_for_team

LEAGUE = League(code="EPL", name="English Premier League", csv_code="E0", api_competition_id=2021, seasons=["2425"])

TEAM_SEARCH_RESPONSE = {"response": [{"team": {"id": 42}}]}
INJURIES_RESPONSE = {
    "response": [
        {"player": {"name": "Bukayo Saka", "position": "Attacker", "reason": "Hamstring"}},
    ]
}


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        get_or_create_team(s, "Arsenal", "EPL")
        s.commit()
        yield s


@pytest.fixture(autouse=True)
def no_squad_or_player_stats_by_default(monkeypatch):
    # Isolates these tests from squad.py's/player_stats.py's/
    # understat_client.py's own internals (each has its own dedicated test
    # file) -- defaults to "no data found", individual tests override this
    # where that specific behavior matters. Without this, these tests would
    # hit the real, live Understat/football-data.org network.
    monkeypatch.setattr(squad, "fetch_squad_for_team", lambda *a, **k: None)
    monkeypatch.setattr(player_stats, "fetch_team_player_stats", lambda *a, **k: [])
    monkeypatch.setattr(understat_client, "fetch_team_season", lambda *a, **k: [])


def _fake_get(calls):
    def get(path, params=None, cache_ttl_seconds=0):
        calls.append(path)
        if path == "/teams":
            return TEAM_SEARCH_RESPONSE
        if path == "/injuries":
            return INJURIES_RESPONSE
        raise AssertionError(f"unexpected path {path}")

    return get


def _arsenal_injury(session):
    team = session.query(Team).filter_by(canonical_name="Arsenal").one()
    return injuries_for_team(session, team.id)[0]


def test_first_sync_searches_and_caches_the_team_id(session, monkeypatch):
    calls = []
    monkeypatch.setattr(api_football_client, "get", _fake_get(calls))

    synced, skipped = injuries.sync_injuries_to_db(session, LEAGUE, 2026)

    assert synced == 1
    assert skipped == 0
    assert calls.count("/teams") == 1
    assert calls.count("/injuries") == 1

    team = session.query(Team).filter_by(canonical_name="Arsenal").one()
    assert team.api_football_team_id == 42


def test_second_sync_reuses_cached_id_and_skips_the_search(session, monkeypatch):
    calls = []
    monkeypatch.setattr(api_football_client, "get", _fake_get(calls))

    injuries.sync_injuries_to_db(session, LEAGUE, 2026)
    session.commit()
    calls.clear()

    synced, skipped = injuries.sync_injuries_to_db(session, LEAGUE, 2026)

    assert synced == 1
    assert skipped == 0
    assert "/teams" not in calls  # the whole point of the cache
    assert calls.count("/injuries") == 1


def test_skips_team_when_search_finds_nothing(session, monkeypatch):
    def get(path, params=None, cache_ttl_seconds=0):
        if path == "/teams":
            return {"response": []}
        raise AssertionError("injuries endpoint should never be reached")

    monkeypatch.setattr(api_football_client, "get", get)

    synced, skipped = injuries.sync_injuries_to_db(session, LEAGUE, 2026)
    assert synced == 0
    assert skipped == 1


def test_skips_team_on_http_error_during_search(session, monkeypatch):
    def get(path, params=None, cache_ttl_seconds=0):
        raise requests.HTTPError("500 server error")

    monkeypatch.setattr(api_football_client, "get", get)

    synced, skipped = injuries.sync_injuries_to_db(session, LEAGUE, 2026)
    assert synced == 0
    assert skipped == 1


def test_position_resolved_via_squad_cross_reference(session, monkeypatch):
    monkeypatch.setattr(api_football_client, "get", _fake_get([]))
    monkeypatch.setattr(
        squad,
        "fetch_squad_for_team",
        lambda *a, **k: [SquadPlayer(name="Bukayo Saka", position="Defender", nationality="England", date_of_birth=None)],
    )

    injuries.sync_injuries_to_db(session, LEAGUE, 2026)
    session.commit()

    # Squad says "Defender" -- must win over the injuries endpoint's own
    # (fake, in this test) "Attacker" position field.
    assert _arsenal_injury(session).position == "defense"


def test_position_falls_back_to_injuries_endpoint_when_no_squad_match(session, monkeypatch):
    monkeypatch.setattr(api_football_client, "get", _fake_get([]))
    # squad fixture (autouse) returns None -> falls back to the injuries
    # response's own position field, "Attacker".

    injuries.sync_injuries_to_db(session, LEAGUE, 2026)
    session.commit()

    assert _arsenal_injury(session).position == "attack"


def test_importance_weight_computed_from_real_stats_when_available(session, monkeypatch):
    monkeypatch.setattr(api_football_client, "get", _fake_get([]))
    monkeypatch.setattr(
        player_stats,
        "fetch_team_player_stats",
        lambda *a, **k: [
            HistoricalPlayerStats(
                name="Bukayo Saka",
                position="Attacker",
                nationality=None,
                appearances=30,
                minutes=2700,
                goals=15,
                assists=10,
                saves=None,
                tackles=None,
                yellow_cards=None,
                red_cards=None,
                rating=7.5,
            ),
        ],
    )

    injuries.sync_injuries_to_db(session, LEAGUE, 2026)
    session.commit()

    assert _arsenal_injury(session).importance_weight != DEFAULT_API_IMPORTANCE_WEIGHT


def test_importance_weight_prefers_understat_xg_over_api_football_goals(session, monkeypatch):
    monkeypatch.setattr(api_football_client, "get", _fake_get([]))
    # No API-Football stats at all (autouse fixture) -- Understat alone
    # should be enough to compute a real (non-default) weight.
    monkeypatch.setattr(
        understat_client,
        "fetch_team_season",
        lambda *a, **k: [
            UnderstatPlayerStats(
                name="Bukayo Saka",
                team_title="Arsenal",
                position="F M",
                minutes=2200,
                goals=12,
                assists=8,
                xg=10.5,
                xa=6.2,
            ),
            UnderstatPlayerStats(
                name="Backup Winger",
                team_title="Arsenal",
                position="M",
                minutes=300,
                goals=1,
                assists=0,
                xg=0.8,
                xa=0.1,
            ),
        ],
    )

    injuries.sync_injuries_to_db(session, LEAGUE, 2026)
    session.commit()

    assert _arsenal_injury(session).importance_weight != DEFAULT_API_IMPORTANCE_WEIGHT


def test_importance_weight_falls_back_to_default_when_no_stats(session, monkeypatch):
    monkeypatch.setattr(api_football_client, "get", _fake_get([]))
    # player_stats fixture (autouse) returns [] -> no data to compute from.

    injuries.sync_injuries_to_db(session, LEAGUE, 2026)
    session.commit()

    assert _arsenal_injury(session).importance_weight == DEFAULT_API_IMPORTANCE_WEIGHT


def test_player_stats_fetch_skipped_when_team_has_no_injuries(session, monkeypatch):
    def get(path, params=None, cache_ttl_seconds=0):
        if path == "/teams":
            return TEAM_SEARCH_RESPONSE
        if path == "/injuries":
            return {"response": []}
        raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr(api_football_client, "get", get)
    fetch_calls = []
    understat_calls = []
    monkeypatch.setattr(
        player_stats, "fetch_team_player_stats", lambda *a, **k: fetch_calls.append(1) or []
    )
    monkeypatch.setattr(
        understat_client, "fetch_team_season", lambda *a, **k: understat_calls.append(1) or []
    )

    synced, skipped = injuries.sync_injuries_to_db(session, LEAGUE, 2026)

    assert synced == 1
    assert fetch_calls == []  # never called -- nothing to price for this team
    assert understat_calls == []  # same gate applies to the Understat lookup
