from __future__ import annotations

import pytest
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

    monkeypatch.setattr(asa_client.http_session, "get", lambda *a, **k: FakeResponse())

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

    monkeypatch.setattr(asa_client.http_session, "get", lambda *a, **k: FakeResponse())

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


def test_fetch_all_teams_totals_makes_one_xgoals_call_per_team(monkeypatch):
    calls = []
    atl_rows = XGOALS_ROWS
    lafc_rows = [
        {
            "player_id": "p1",
            "team_id": "LAFC1",
            "minutes_played": 1000,
            "shots": 30,
            "goals": 10,
            "xgoals": 8.0,
            "key_passes": 5,
            "primary_assists": 2,
            "xassists": 1.5,
            "points_added": 1.0,
        }
    ]

    def fake_get_cached_json(cache_key, url, params):
        if url.endswith("/mls/teams"):
            return TEAMS
        if url.endswith("/mls/players"):
            return PLAYERS
        if url.endswith("/mls/players/xgoals"):
            calls.append(params["team_id"])
            return atl_rows if params["team_id"] == "ATL1" else lafc_rows
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(asa_client, "_get_cached_json", fake_get_cached_json)

    result = asa_client.fetch_all_teams_totals(2026)

    assert set(result) == {"Atlanta United FC", "Los Angeles FC"}
    assert {p.name for p in result["Atlanta United FC"]} == {"Star Striker", "Backup Forward"}
    assert {p.name for p in result["Los Angeles FC"]} == {"Star Striker"}
    assert sorted(calls) == ["ATL1", "LAFC1"]


def test_fetch_all_teams_totals_no_teams_returns_empty(monkeypatch):
    _stub_fetches(monkeypatch, teams=[])

    assert asa_client.fetch_all_teams_totals(2026) == {}


def test_fetch_all_teams_totals_network_error_returns_empty(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network down")

    monkeypatch.setattr(asa_client, "_fetch_all_teams", raise_error)

    assert asa_client.fetch_all_teams_totals(2026) == {}


def test_fetch_team_season_network_error_returns_empty(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network down")

    monkeypatch.setattr(asa_client, "_fetch_all_teams", raise_error)

    assert asa_client.fetch_team_season("Atlanta United FC", 2026) == []


# --- the whole-league goalkeeper and defender lists ------------------------------------------

BIOS = [
    {"player_id": "gk1", "player_name": "Safe Hands"},
    {"player_id": "gk2", "player_name": "Traded Keeper"},
    {"player_id": "cb1", "player_name": "Rock Solid"},
    {"player_id": "fb1", "player_name": "Overlapping Back"},
    {"player_id": "w1", "player_name": "Quick Winger"},
]


def _action(kind, above_avg, raw=9.9):
    return {"action_type": kind, "goals_added_raw": raw, "goals_added_above_avg": above_avg, "count_actions": 10}


def _stub_league_lists(monkeypatch, keepers=(), goals_added=()):
    from soccer_predictor.ingest import memo

    memo.clear_all()
    asked = []

    def fake_get_cached_json(cache_key, url, params):
        asked.append((url.rsplit("/mls/", 1)[1], params.get("season_name")))
        if url.endswith("/mls/teams"):
            return TEAMS
        if url.endswith("/mls/players"):
            return BIOS
        if url.endswith("/mls/goalkeepers/xgoals"):
            return list(keepers)
        if url.endswith("/mls/players/goals-added"):
            return list(goals_added)
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(asa_client, "_get_cached_json", fake_get_cached_json)
    return asked


def test_goalkeepers_come_from_one_league_wide_request_with_goals_prevented_as_expected_minus_conceded(monkeypatch):
    keepers = [
        {"player_id": "gk1", "team_id": "ATL1", "minutes_played": 3000, "shots_faced": 150, "saves": 110, "goals_conceded": 40,
         "xgoals_gk_faced": 46.5, "goals_minus_xgoals_gk": -6.5},
        {"player_id": "gk2", "team_id": ["ATL1", "LAFC1"], "minutes_played": 900, "shots_faced": 40, "saves": 25, "goals_conceded": 15,
         "xgoals_gk_faced": 12.0},
        {"player_id": "ghost", "team_id": "ATL1", "minutes_played": 10},  # no bio: skipped
    ]
    asked = _stub_league_lists(monkeypatch, keepers=keepers)

    result = asa_client.fetch_league_goalkeepers(2026)

    assert asked.count(("goalkeepers/xgoals", 2026)) == 1
    safe, traded = result
    assert (safe.name, safe.team, safe.minutes, safe.shots_faced, safe.saves, safe.goals_conceded) == (
        "Safe Hands", "Atlanta United FC", 3000, 150, 110, 40,
    )
    assert safe.goals_prevented == 6.5  # the shots he faced should have gone in 46.5 times; 40 did
    assert traded.team == "Los Angeles FC" and traded.goals_prevented == -3.0  # the club he is at now; let in more than expected


def test_defenders_are_the_centre_backs_and_full_backs_with_goals_added_above_average(monkeypatch):
    goals_added = [
        {"player_id": "cb1", "team_id": "ATL1", "general_position": "CB", "minutes_played": 2000,
         "data": [_action("Interrupting", 0.50), _action("Fouling", 0.10), _action("Passing", -0.20), _action("Receiving", 0.05)]},
        {"player_id": "fb1", "team_id": ["LAFC1", "ATL1"], "general_position": "FB", "minutes_played": 1200, "data": []},
        {"player_id": "w1", "team_id": "ATL1", "general_position": "W", "minutes_played": 1500, "data": [_action("Interrupting", 2.0)]},
        {"player_id": "ghost", "team_id": "ATL1", "general_position": "CB", "minutes_played": 100, "data": []},
    ]
    asked = _stub_league_lists(monkeypatch, goals_added=goals_added)

    result = asa_client.fetch_league_defenders(2026)

    assert asked.count(("players/goals-added", 2026)) == 1
    assert [d.name for d in result] == ["Rock Solid", "Overlapping Back"]  # not the winger, nor a row with no bio
    rock, back = result
    assert (rock.team, rock.position, rock.minutes) == ("Atlanta United FC", "CB", 2000)
    assert rock.interrupting == 0.5 and rock.fouling == 0.1
    assert rock.total == pytest.approx(0.45)  # above-average values summed over every kind of action, not the raw ones
    assert (back.team, back.position, back.interrupting, back.total) == ("Atlanta United FC", "FB", 0.0, 0.0)


def test_a_failing_request_gives_no_goalkeepers_or_defenders(monkeypatch):
    from soccer_predictor.ingest import memo

    memo.clear_all()

    def down(*args, **kwargs):
        raise requests.RequestException("down")

    monkeypatch.setattr(asa_client, "_get_cached_json", down)
    assert asa_client.fetch_league_goalkeepers(2026) == []
    assert asa_client.fetch_league_defenders(2026) == []
    memo.clear_all()
