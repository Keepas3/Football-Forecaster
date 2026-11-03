"""The committed copy of Understat's current-season data: what is used when Understat is down and there is
no disk cache (a GitHub Actions runner, a freshly rebooted Streamlit app), and how the daily refresh keeps it
up to date."""

from __future__ import annotations

import datetime as dt
import json
import os

import pytest
import requests

from soccer_predictor.ingest import understat_client as u
from soccer_predictor.ingest.memo import clear_all

ROWS = [
    {"player_name": "Bukayo Saka", "team_title": "Arsenal", "position": "F", "time": "2200", "goals": "12",
     "assists": "8", "xG": "10.5", "xA": "6.2", "games": "28"},
    {"player_name": "Declan Rice", "team_title": "Arsenal", "position": "M", "time": "2500", "goals": "4",
     "assists": "5", "xG": "3.1", "xA": "4.0", "games": "30"},
]
NEWER_ROWS = [{**ROWS[0], "goals": "13"}, ROWS[1]]


@pytest.fixture(autouse=True)
def folders(tmp_path, monkeypatch):
    monkeypatch.setattr(u, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(u, "SNAPSHOT_DIR", tmp_path / "snapshot")
    monkeypatch.setattr(u, "_unreachable_until", 0.0)
    clear_all()
    yield tmp_path
    clear_all()


def _down(monkeypatch, error=None):
    """Understat is down: its data request fails the way it did on 2026-11-03 (HTTP 405)."""
    def request(slug, season):
        raise error or requests.HTTPError("405 Client Error: Not Allowed")

    monkeypatch.setattr(u, "_request_league_players", request)


def _up(monkeypatch, rows):
    calls = []
    monkeypatch.setattr(u, "_request_league_players", lambda slug, season: calls.append((slug, season)) or rows)
    return calls


def _snapshot(slug, season, rows, fetched_at="2026-11-01T10:00:00"):
    u.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    u._snapshot_path(slug).write_text(
        json.dumps({"season": season, "fetched_at": fetched_at, "players": rows}), encoding="utf-8"
    )


# --- reading it ------------------------------------------------------------------------------


def test_the_snapshot_is_used_when_understat_is_down_and_there_is_no_disk_cache(monkeypatch):
    _snapshot("EPL", 2026, ROWS)
    _down(monkeypatch)

    assert u._fetch_league_players("EPL", 2026) == ROWS
    # ...and so every function built on it still works
    assert [p.name for p in u.fetch_team_season("Arsenal", "EPL", 2026)] == ["Bukayo Saka", "Declan Rice"]
    assert list(u.fetch_all_teams_totals("EPL", 2026)) == ["Arsenal"]


def test_a_snapshot_of_another_season_is_not_used(monkeypatch):
    _snapshot("EPL", 2025, ROWS)
    _down(monkeypatch)

    with pytest.raises(requests.RequestException):
        u._fetch_league_players("EPL", 2026)
    assert u.fetch_team_season("Arsenal", "EPL", 2026) == []  # nothing to fall back on: no stats, as before


def test_no_snapshot_and_no_cache_means_no_stats_as_before(monkeypatch):
    _down(monkeypatch)
    assert u.fetch_team_season("Arsenal", "EPL", 2026) == []
    assert u.fetch_league_players("EPL", 2026) == []


def test_a_maintenance_page_instead_of_json_also_falls_back(monkeypatch):
    _snapshot("EPL", 2026, ROWS)
    _down(monkeypatch, ValueError("Expecting value: line 1 column 1 (char 0)"))  # response.json() on an HTML page

    assert u._fetch_league_players("EPL", 2026) == ROWS


def test_the_newer_of_the_disk_cache_and_the_snapshot_wins(monkeypatch):
    _snapshot("EPL", 2026, ROWS, fetched_at="2026-11-01T10:00:00")
    cache = u._cache_path("EPL", 2026)
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(NEWER_ROWS), encoding="utf-8")
    two_days = dt.datetime.now().timestamp() - 2 * 24 * 3600
    os.utime(cache, (two_days, two_days))  # stale (over a day), but newer than the snapshot
    _down(monkeypatch)

    assert u._fetch_league_players("EPL", 2026) == NEWER_ROWS

    old = dt.datetime(2026, 10, 1).timestamp()
    os.utime(cache, (old, old))  # now older than the snapshot
    assert u._fetch_league_players("EPL", 2026) == ROWS


def test_while_understat_answers_its_own_data_is_used_not_the_snapshot(monkeypatch):
    _snapshot("EPL", 2026, ROWS)
    _up(monkeypatch, NEWER_ROWS)

    assert u._fetch_league_players("EPL", 2026) == NEWER_ROWS


def test_each_leagues_snapshot_has_its_own_file():
    names = {u._snapshot_path(slug).name for slug in u.UNDERSTAT_LEAGUE_SLUG.values()}
    assert names == {"epl.json", "la-liga.json", "bundesliga.json", "serie-a.json", "ligue-1.json"}


# --- writing it (the daily refresh) ------------------------------------------------------------


def test_save_snapshot_writes_what_understat_just_gave(monkeypatch):
    calls = _up(monkeypatch, ROWS)

    assert u.save_snapshot("EPL", 2026) is True

    saved = json.loads(u._snapshot_path("EPL").read_text(encoding="utf-8"))
    assert saved["season"] == 2026 and saved["players"] == ROWS and saved["fetched_at"].startswith("20")
    assert calls == [("EPL", 2026)]
    assert u._read_snapshot("EPL", 2026) is not None


def test_save_snapshot_leaves_an_unchanged_file_alone(monkeypatch):
    _up(monkeypatch, ROWS)
    assert u.save_snapshot("EPL", 2026) is True
    before = u._snapshot_path("EPL").read_text(encoding="utf-8")

    clear_all()
    cache = u._cache_path("EPL", 2026)
    later = dt.datetime.now().timestamp() + 60
    os.utime(cache, (later, later))  # fetched a minute later, same numbers
    assert u.save_snapshot("EPL", 2026) is False
    assert u._snapshot_path("EPL").read_text(encoding="utf-8") == before  # not even the fetch time moved


def test_save_snapshot_replaces_old_numbers_with_new_ones(monkeypatch):
    _snapshot("EPL", 2026, ROWS)
    _up(monkeypatch, NEWER_ROWS)

    assert u.save_snapshot("EPL", 2026) is True
    assert u._read_snapshot("EPL", 2026)[1] == NEWER_ROWS


def test_save_snapshot_does_nothing_while_understat_is_down(monkeypatch):
    _snapshot("EPL", 2026, ROWS)
    before = u._snapshot_path("EPL").read_text(encoding="utf-8")
    _down(monkeypatch)

    assert u.save_snapshot("EPL", 2026) is False
    assert u._snapshot_path("EPL").read_text(encoding="utf-8") == before


def test_save_snapshot_never_saves_an_old_disk_cache_over_the_snapshot(monkeypatch):
    """Understat is down and only a stale cache exists: that is not 'what Understat just gave'."""
    cache = u._cache_path("EPL", 2026)
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(NEWER_ROWS), encoding="utf-8")
    old = dt.datetime(2026, 10, 1).timestamp()
    os.utime(cache, (old, old))
    _down(monkeypatch)

    assert u.save_snapshot("EPL", 2026) is False
    assert not u._snapshot_path("EPL").exists()


def test_save_snapshot_ignores_an_empty_answer(monkeypatch):
    _up(monkeypatch, [])
    assert u.save_snapshot("EPL", 2026) is False
    assert not u._snapshot_path("EPL").exists()


def test_an_older_season_never_replaces_a_newer_snapshot():
    _snapshot("EPL", 2027, NEWER_ROWS)
    assert u._write_snapshot("EPL", 2026, ROWS, dt.datetime(2026, 11, 3)) is False
    assert u._read_snapshot("EPL", 2027)[1] == NEWER_ROWS

    assert u._write_snapshot("EPL", 2028, ROWS, dt.datetime(2027, 8, 20)) is True  # a new season replaces it
    assert u._read_snapshot("EPL", 2027) is None and u._read_snapshot("EPL", 2028) is not None


def test_a_league_understat_does_not_cover_has_no_snapshot():
    assert u.save_snapshot("UCL", 2026) is False
    assert u.save_snapshot("MLS", 2026) is False


# --- the script ----------------------------------------------------------------------------------


def test_the_script_snapshots_every_understat_league_for_its_current_season(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "save_understat_snapshot", Path(__file__).resolve().parents[1] / "scripts" / "save_understat_snapshot.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    seen = []
    monkeypatch.setattr(u, "save_snapshot", lambda code, season: seen.append((code, season)) or code == "EPL")

    script.main()

    assert [code for code, _ in seen] == list(u.UNDERSTAT_LEAGUE_SLUG)  # the five, no UCL/MLS/cups
    assert all(isinstance(season, int) and season >= 2026 for _, season in seen)
    out = capsys.readouterr().out
    assert "1 snapshot file(s) changed" in out and "EPL" in out
