from __future__ import annotations

import requests

from soccer_predictor.ingest import asa_client

TEAMS = [
    {"team_id": "ATL1", "team_name": "Atlanta United FC"},
    {"team_id": "LAFC1", "team_name": "Los Angeles FC"},
]

PLAYERS = [
    {"player_id": "p1", "player_name": "Star Striker", "primary_general_position": "ST"},
    {"player_id": "p2", "player_name": "Backup Forward", "primary_general_position": "ST"},
]

XGOALS_ROWS = [
    {
        "player_id": "p1",
        "team_id": "ATL1",
        "general_position": "ST",
        "minutes_played": 2000,
        "shots": 60,
        "goals": 15,
        "xgoals": 12.5,
        "key_passes": 20,
        "primary_assists": 5,
        "xassists": 4.2,
        "points_added": 3.1,
    },
    {
        "player_id": "p2",
        "team_id": "ATL1",
        "general_position": "ST",
        "minutes_played": 300,
        "shots": 5,
        "goals": 1,
        "xgoals": 0.8,
        "key_passes": 1,
        "primary_assists": 0,
        "xassists": 0.1,
        "points_added": 0.2,
    },
]


def _stub_fetches(monkeypatch, teams=TEAMS, players=PLAYERS, xgoals=XGOALS_ROWS):
    def fake_get_cached_json(cache_key, url, params):
        if url.endswith("/mls/teams"):
            return teams
        if url.endswith("/mls/players"):
            return players
        if url.endswith("/mls/players/xgoals"):
            return xgoals
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(asa_client, "_get_cached_json", fake_get_cached_json)


def test_fetch_team_season_joins_bio_and_stats(monkeypatch):
    _stub_fetches(monkeypatch)

    players = asa_client.fetch_team_season("Atlanta United FC", 2026)

    names = {p.name for p in players}
    assert names == {"Star Striker", "Backup Forward"}
    star = next(p for p in players if p.name == "Star Striker")
    assert star.goals == 15
    assert star.assists == 5
    assert star.xg == 12.5
    assert star.xa == 4.2
    assert star.shots == 60
    assert star.key_passes == 20
    assert star.points_added == 3.1


def test_fetch_team_season_skips_stats_rows_with_no_bio_match(monkeypatch):
    xgoals = XGOALS_ROWS + [{"player_id": "unknown", "team_id": "ATL1", "goals": 99}]
    _stub_fetches(monkeypatch, xgoals=xgoals)

    players = asa_client.fetch_team_season("Atlanta United FC", 2026)

    assert len(players) == 2  # the unmatched "unknown" row is dropped, not fabricated


def test_fetch_team_season_lafc_override_resolves_to_los_angeles_fc(monkeypatch):
    xgoals_lafc = [
        {
            "player_id": "p1",
            "team_id": "LAFC1",
            "goals": 10,
            "minutes_played": 1000,
            "shots": 30,
            "key_passes": 5,
            "primary_assists": 2,
            "xassists": 1.5,
            "xgoals": 8.0,
            "points_added": 1.0,
        }
    ]
    calls = []

    def fake_get_cached_json(cache_key, url, params):
        if url.endswith("/mls/teams"):
            return TEAMS
        if url.endswith("/mls/players"):
            return PLAYERS
        if url.endswith("/mls/players/xgoals"):
            calls.append(params)
            return xgoals_lafc
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(asa_client, "_get_cached_json", fake_get_cached_json)

    players = asa_client.fetch_team_season("LAFC", 2026)

    assert len(players) == 1
    assert calls[0]["team_id"] == "LAFC1"  # confirms the override, not a failed fuzzy match


def test_fetch_team_season_no_team_match_returns_empty(monkeypatch):
    _stub_fetches(monkeypatch)

    assert asa_client.fetch_team_season("Some Team Nobody Has Heard Of", 2026) == []


def test_fetch_team_season_no_teams_returns_empty(monkeypatch):
    _stub_fetches(monkeypatch, teams=[])

    assert asa_client.fetch_team_season("Atlanta United FC", 2026) == []


def test_fetch_team_season_no_stat_rows_returns_empty(monkeypatch):
    _stub_fetches(monkeypatch, xgoals=[])

    assert asa_client.fetch_team_season("Atlanta United FC", 2026) == []


def test_get_cached_json_returns_empty_on_non_list_response(tmp_path, monkeypatch):
    monkeypatch.setattr(asa_client, "CACHE_DIR", tmp_path)

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"error": "nope"}

    monkeypatch.setattr(asa_client.requests, "get", lambda *a, **k: FakeResponse())

    assert asa_client._get_cached_json("x", "https://example.test/x", {}) == []


def test_get_cached_json_caches_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(asa_client, "CACHE_DIR", tmp_path)
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            calls.append(1)
            return [{"a": 1}]

    monkeypatch.setattr(asa_client.requests, "get", lambda *a, **k: FakeResponse())

    first = asa_client._get_cached_json("x", "https://example.test/x", {})
    second = asa_client._get_cached_json("x", "https://example.test/x", {})

    assert first == [{"a": 1}]
    assert second == [{"a": 1}]
    assert len(calls) == 1  # second call served from disk cache


def test_asa_available_seasons_starts_at_2013_and_includes_current_year():
    import datetime as dt

    seasons = asa_client.ASA_AVAILABLE_SEASONS

    assert seasons[0] == 2013
    assert seasons == sorted(seasons)
    assert dt.date.today().year in seasons


def test_fetch_team_season_network_error_returns_empty(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network down")

    monkeypatch.setattr(asa_client, "_fetch_all_teams", raise_error)

    assert asa_client.fetch_team_season("Atlanta United FC", 2026) == []
