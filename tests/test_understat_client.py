from __future__ import annotations

import requests

from soccer_predictor.ingest import understat_client


def _row(name, team, goals=0, assists=0, xg=0.0, xa=0.0, minutes=0, position="F"):
    return {
        "player_name": name,
        "team_title": team,
        "position": position,
        "time": str(minutes),
        "goals": str(goals),
        "assists": str(assists),
        "xG": str(xg),
        "xA": str(xa),
    }


LEAGUE_ROWS = [
    _row("Bukayo Saka", "Arsenal", goals=12, assists=8, xg=10.5, xa=6.2, minutes=2200),
    _row("Backup Winger", "Arsenal", goals=1, assists=0, xg=0.8, xa=0.1, minutes=300),
    _row("Erling Haaland", "Manchester City", goals=25, assists=3, xg=22.1, xa=2.4, minutes=2900),
]


def test_fetch_team_season_returns_only_matching_team_players(monkeypatch):
    monkeypatch.setattr(understat_client, "_fetch_league_players", lambda *a, **k: LEAGUE_ROWS)

    players = understat_client.fetch_team_season("Arsenal", "EPL", 2026)

    names = {p.name for p in players}
    assert names == {"Bukayo Saka", "Backup Winger"}


def test_fetch_team_season_fuzzy_matches_team_name(monkeypatch):
    monkeypatch.setattr(understat_client, "_fetch_league_players", lambda *a, **k: LEAGUE_ROWS)

    players = understat_client.fetch_team_season("Man City", "EPL", 2026)

    assert {p.name for p in players} == {"Erling Haaland"}


def test_fetch_team_season_includes_mid_season_transfers_not_just_pure_title(monkeypatch):
    # Regression test: a player who transferred mid-season gets a combined
    # Understat team_title like "Chelsea,Manchester City" -- confirmed live
    # this used to make a short query like "Man City" tie-score against
    # that combined title over the real "Manchester City", returning ONLY
    # the transferred player instead of the whole real squad.
    rows = [
        _row("Erling Haaland", "Manchester City", goals=25, assists=3, xg=22.1, xa=2.4, minutes=2900),
        _row("Phil Foden", "Manchester City", goals=10, assists=5, xg=9.0, xa=4.0, minutes=2500),
        _row("Enzo Fernandez", "Chelsea,Manchester City", goals=1, assists=0, xg=0.5, xa=0.1, minutes=270),
        _row("Someone Else", "Chelsea", goals=8, assists=2, xg=7.0, xa=1.5, minutes=2000),
    ]
    monkeypatch.setattr(understat_client, "_fetch_league_players", lambda *a, **k: rows)

    players = understat_client.fetch_team_season("Man City", "EPL", 2026)

    names = {p.name for p in players}
    assert names == {"Erling Haaland", "Phil Foden", "Enzo Fernandez"}
    assert "Someone Else" not in names


def test_fetch_team_season_parses_numeric_fields(monkeypatch):
    monkeypatch.setattr(understat_client, "_fetch_league_players", lambda *a, **k: LEAGUE_ROWS)

    saka = next(p for p in understat_client.fetch_team_season("Arsenal", "EPL", 2026) if p.name == "Bukayo Saka")

    assert saka.goals == 12
    assert saka.assists == 8
    assert saka.xg == 10.5
    assert saka.xa == 6.2
    assert saka.minutes == 2200


def test_fetch_team_season_unmapped_league_returns_empty_without_fetching(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("should never fetch for an unmapped league")

    monkeypatch.setattr(understat_client, "_fetch_league_players", fail)

    assert understat_client.fetch_team_season("Arsenal", "UCL", 2026) == []


def test_fetch_team_season_returns_empty_when_no_team_match(monkeypatch):
    monkeypatch.setattr(understat_client, "_fetch_league_players", lambda *a, **k: LEAGUE_ROWS)

    assert understat_client.fetch_team_season("Totally Unrelated FC", "EPL", 2026) == []


def test_fetch_team_season_returns_empty_on_empty_league_response(monkeypatch):
    monkeypatch.setattr(understat_client, "_fetch_league_players", lambda *a, **k: [])

    assert understat_client.fetch_team_season("Arsenal", "EPL", 2026) == []


def test_fetch_team_season_degrades_on_request_exception(monkeypatch):
    def raise_request_error(*args, **kwargs):
        raise requests.RequestException("network error")

    monkeypatch.setattr(understat_client, "_fetch_league_players", raise_request_error)

    assert understat_client.fetch_team_season("Arsenal", "EPL", 2026) == []


def test_fetch_team_season_degrades_on_malformed_rows(monkeypatch):
    # Missing "team_title" -- simulates Understat changing its response shape.
    monkeypatch.setattr(understat_client, "_fetch_league_players", lambda *a, **k: [{"player_name": "X"}])

    assert understat_client.fetch_team_season("Arsenal", "EPL", 2026) == []


def test_fetch_league_players_serves_from_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(understat_client, "CACHE_DIR", tmp_path)

    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True, "players": LEAGUE_ROWS}

    def fake_post(*args, **kwargs):
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(understat_client.requests, "post", fake_post)

    first = understat_client._fetch_league_players("EPL", 2026)
    second = understat_client._fetch_league_players("EPL", 2026)

    assert first == LEAGUE_ROWS
    assert second == LEAGUE_ROWS
    assert len(calls) == 1  # second call served from disk cache


def test_fetch_league_players_returns_empty_when_success_false(tmp_path, monkeypatch):
    monkeypatch.setattr(understat_client, "CACHE_DIR", tmp_path)

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": False}

    monkeypatch.setattr(understat_client.requests, "post", lambda *a, **k: FakeResponse())

    assert understat_client._fetch_league_players("EPL", 2026) == []


def test_understat_available_seasons_starts_at_2014_and_includes_current_year():
    import datetime as dt

    seasons = understat_client.UNDERSTAT_AVAILABLE_SEASONS

    assert seasons[0] == 2014
    assert seasons == sorted(seasons)
    assert dt.date.today().year in seasons
