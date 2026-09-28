from __future__ import annotations

import requests

from soccer_predictor.ingest import api_football_client, player_stats


def _stat_row(league_name: str, appearances: int, goals: int = 0, saves=None, position="Defender") -> dict:
    return {
        "league": {"name": league_name},
        "games": {
            "appearences": appearances,
            "minutes": appearances * 90,
            "rating": "7.10",
            "position": position,
        },
        "goals": {"total": goals, "assists": 1, "saves": saves},
        "tackles": {"total": 10},
        "cards": {"yellow": 2, "red": 0},
    }


def _player_entry(name: str, nationality: str, statistics: list[dict]) -> dict:
    return {"player": {"name": name, "nationality": nationality}, "statistics": statistics}


TEAMS_RESPONSE = {
    "response": [
        {"team": {"id": 50, "name": "Manchester City"}},
        {"team": {"id": 40, "name": "Liverpool"}},
    ]
}


def test_select_primary_row_matches_competition_name():
    rows = [_stat_row("UEFA Champions League", 5), _stat_row("Premier League", 30)]
    selected = player_stats._select_primary_row(rows, "EPL")
    assert selected["league"]["name"] == "Premier League"


def test_select_primary_row_falls_back_to_most_appearances():
    rows = [_stat_row("Some Cup", 2), _stat_row("Another Cup", 8)]
    selected = player_stats._select_primary_row(rows, "EPL")
    assert selected["league"]["name"] == "Another Cup"


def test_parse_player_extracts_fields():
    entry = _player_entry(
        "M. Akanji", "Switzerland", [_stat_row("Premier League", 20, goals=5, saves=None)]
    )
    player = player_stats._parse_player(entry, "EPL")
    assert player.name == "M. Akanji"
    assert player.nationality == "Switzerland"
    assert player.position == "Defender"
    assert player.appearances == 20
    assert player.goals == 5
    assert player.assists == 1
    assert player.saves is None
    assert player.tackles == 10
    assert player.rating == 7.10


def test_parse_player_returns_none_when_no_statistics():
    entry = _player_entry("Bench Warmer", "England", [])
    assert player_stats._parse_player(entry, "EPL") is None


def test_find_team_id_fuzzy_matches_abbreviated_name(monkeypatch):
    monkeypatch.setattr(api_football_client, "get", lambda *a, **k: TEAMS_RESPONSE)
    assert player_stats._find_team_id("Man City", "EPL", 2024) == 50


def test_find_team_id_returns_none_for_unrelated_name(monkeypatch):
    monkeypatch.setattr(api_football_client, "get", lambda *a, **k: TEAMS_RESPONSE)
    assert player_stats._find_team_id("Totally Unrelated FC", "EPL", 2024) is None


def test_find_team_id_returns_none_for_unmapped_league(monkeypatch):
    monkeypatch.setattr(api_football_client, "get", lambda *a, **k: TEAMS_RESPONSE)
    assert player_stats._find_team_id("Man City", "UNKNOWN_LEAGUE", 2024) is None


def test_fetch_team_player_stats_paginates_and_filters_by_league(monkeypatch):
    page1 = {
        "response": [
            _player_entry("M. Akanji", "Switzerland", [_stat_row("Premier League", 30, goals=2)]),
        ],
        "paging": {"current": 1, "total": 2},
    }
    page2 = {
        "response": [
            _player_entry(
                "E. Haaland", "Norway", [_stat_row("Premier League", 35, goals=27, position="Attacker")]
            ),
        ],
        "paging": {"current": 2, "total": 2},
    }

    def fake_get(path, params, cache_ttl_seconds=0):
        if path == "/teams":
            return TEAMS_RESPONSE
        return page1 if params["page"] == 1 else page2

    monkeypatch.setattr(api_football_client, "get", fake_get)

    players = player_stats.fetch_team_player_stats("Man City", "EPL", 2024)
    names = {p.name for p in players}
    assert names == {"M. Akanji", "E. Haaland"}
    assert next(p for p in players if p.name == "E. Haaland").goals == 27


def test_fetch_team_player_stats_returns_empty_when_team_not_matched(monkeypatch):
    monkeypatch.setattr(api_football_client, "get", lambda *a, **k: {"response": []})
    assert player_stats.fetch_team_player_stats("Unknown Team", "EPL", 2024) == []


def test_fetch_team_player_stats_returns_empty_on_missing_key(monkeypatch):
    def raise_missing_key(*args, **kwargs):
        raise api_football_client.MissingApiKey("no key")

    monkeypatch.setattr(api_football_client, "get", raise_missing_key)
    assert player_stats.fetch_team_player_stats("Man City", "EPL", 2024) == []


def test_fetch_team_player_stats_returns_empty_on_http_error(monkeypatch):
    def fake_get(path, params, cache_ttl_seconds=0):
        if path == "/teams":
            return TEAMS_RESPONSE
        raise requests.HTTPError("500")

    monkeypatch.setattr(api_football_client, "get", fake_get)
    assert player_stats.fetch_team_player_stats("Man City", "EPL", 2024) == []


def test_fetch_team_player_stats_uses_requested_season(monkeypatch):
    seen_seasons = []

    def fake_get(path, params, cache_ttl_seconds=0):
        seen_seasons.append(params.get("season"))
        if path == "/teams":
            return TEAMS_RESPONSE
        return {"response": [], "paging": {"current": 1, "total": 1}}

    monkeypatch.setattr(api_football_client, "get", fake_get)
    player_stats.fetch_team_player_stats("Man City", "EPL", 2022)
    assert seen_seasons == [2022, 2022]
