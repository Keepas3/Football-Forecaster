"""The frequent "quick" refresh: cheap ESPN result sync, quick refresh_league,
the request-rate override, and the change fingerprint that gates commits."""

from __future__ import annotations

import datetime as dt
import importlib.util
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import api_client, espn_client, refresh
from soccer_predictor.ingest.fixtures import sync_recent_results_to_db_espn, sync_results_to_db_espn
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, matches_for_league, update_espn_team_id

NL = League(
    code="NL", name="UEFA Nations League", seasons=["2627"], data_source="espn",
    espn_league_slug="uefa.nations", has_groups=True,
)
MLS = League(code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1")
CSV_LEAGUE = League(code="EPL", name="English Premier League", csv_code="E0", seasons=["2627"])


# --- request rate override ---------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(None, 10), ("5", 5), ("1", 1), ("0", 1), ("-3", 1), ("50", 10), ("lots", 10), ("", 10)],
)
def test_rate_override_can_only_lower_the_free_tier_ceiling(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("FOOTBALL_DATA_MAX_REQUESTS_PER_MINUTE", raising=False)
    else:
        monkeypatch.setenv("FOOTBALL_DATA_MAX_REQUESTS_PER_MINUTE", value)
    assert api_client._configured_max_requests_per_minute() == expected


# --- recent results from the scoreboard ----------------------------------------------


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _seed(session, code, ids):
    for espn_id in ids:
        team = get_or_create_team(session, f"Team {espn_id}", code)
        update_espn_team_id(session, team.id, int(espn_id))
    session.commit()


def _espn_match(home, away, completed, scores, state, day=None, group=None):
    day = day or dt.datetime.now(dt.UTC).date()
    return espn_client.EspnMatch(
        espn_home_id=home, espn_away_id=away, home_name=f"Team {home}", away_name=f"Team {away}",
        date=day, kickoff_utc=dt.datetime.combine(day, dt.time(18, 45)), completed=completed,
        home_score=scores[0] if scores else None, away_score=scores[1] if scores else None,
        state=state, group_name=group,
    )


def test_recent_sync_stores_only_completed_matches_with_a_score(session, monkeypatch):
    _seed(session, "NL", [1, 2, 3, 4, 5, 6])
    matches = [
        _espn_match("1", "2", True, (2, 1), "post", group="Group A1"),
        _espn_match("3", "4", False, (0, 0), "in"),  # live, not final
        _espn_match("5", "6", False, None, "pre"),  # not started
    ]
    monkeypatch.setattr(espn_client, "fetch_group_membership", lambda *a, **k: {})
    monkeypatch.setattr(espn_client, "fetch_day_matches", lambda slug, day, cache_ttl_seconds=None: matches if day == dt.datetime.now(dt.UTC).date() else [])

    synced, skipped = sync_recent_results_to_db_espn(session, NL, "2627")
    session.commit()

    stored = matches_for_league(session, "NL")
    assert (synced, skipped) == (1, 0)
    assert len(stored) == 1
    assert (stored.iloc[0]["home_goals"], stored.iloc[0]["away_goals"]) == (2, 1)
    assert stored.iloc[0]["group_name"] == "Group A1"


def test_recent_sync_uses_a_handful_of_requests_not_one_per_team(session, monkeypatch):
    _seed(session, "NL", range(1, 56))  # 55 teams, like the Nations League
    calls = []
    monkeypatch.setattr(espn_client, "fetch_group_membership", lambda *a, **k: calls.append("standings") or {})
    monkeypatch.setattr(
        espn_client, "fetch_day_matches", lambda slug, day, cache_ttl_seconds=None: calls.append("scoreboard") or []
    )
    monkeypatch.setattr(
        espn_client, "fetch_team_results", lambda *a, **k: calls.append("team schedule") or []
    )

    sync_recent_results_to_db_espn(session, NL, "2627")

    assert calls.count("team schedule") == 0
    assert len(calls) <= 5  # 3 scoreboard days + 1 standings


def test_recent_sync_and_full_sync_write_the_same_row_not_a_duplicate(session, monkeypatch):
    _seed(session, "MLS", [183, 20232])
    final = _espn_match("183", "20232", True, (2, 1), "post")
    monkeypatch.setattr(espn_client, "fetch_day_matches", lambda slug, day, cache_ttl_seconds=None: [final] if day == final.date else [])
    monkeypatch.setattr(espn_client, "fetch_team_results", lambda *a, **k: [final])

    sync_recent_results_to_db_espn(session, MLS, "2026")
    sync_results_to_db_espn(session, MLS, "2026")
    session.commit()

    assert len(matches_for_league(session, "MLS")) == 1


def test_recent_sync_never_looks_up_groups_for_a_league_without_them(session, monkeypatch):
    _seed(session, "MLS", [183, 20232])

    def fail(*a, **k):
        raise AssertionError("standings must not be requested for a league without groups")

    monkeypatch.setattr(espn_client, "fetch_group_membership", fail)
    monkeypatch.setattr(espn_client, "fetch_day_matches", lambda *a, **k: [])
    sync_recent_results_to_db_espn(session, MLS, "2026")


# --- quick refresh_league ---------------------------------------------------------------


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _record_calls(monkeypatch):
    calls = []

    def fake(name, result=None):
        def inner(*a, **k):
            calls.append((name, k))
            return result

        return inner

    monkeypatch.setattr(refresh, "session_scope", lambda: _FakeSession())
    monkeypatch.setattr(refresh, "seed_teams_from_espn", fake("seed_espn", 0))
    monkeypatch.setattr(refresh, "seed_teams_from_api", fake("seed_api", 0))
    monkeypatch.setattr(refresh, "sync_fixtures_to_db_espn", fake("fixtures_espn", (4, 0)))
    monkeypatch.setattr(refresh, "sync_results_to_db_espn", fake("results_espn_full", (99, 0)))
    monkeypatch.setattr(refresh, "sync_recent_results_to_db_espn", fake("results_espn_recent", (6, 0)))
    monkeypatch.setattr(refresh, "sync_fixtures_to_db", fake("fixtures_fd", (7, 0)))
    monkeypatch.setattr(refresh, "load_latest_params", lambda *a, **k: None)
    monkeypatch.setattr(refresh.injuries_module, "sync_injuries_to_db", fake("injuries", (5, 0)))
    return calls


def test_quick_refresh_of_an_espn_league_is_cheap(monkeypatch):
    calls = _record_calls(monkeypatch)

    result = refresh.refresh_league(NL, quick=True)

    names = [n for n, _ in calls]
    assert names == ["fixtures_espn", "results_espn_recent"]  # no seeding, no full sync, no injuries
    assert dict(calls)["fixtures_espn"]["window_days"] == refresh.QUICK_FIXTURE_WINDOW_DAYS
    assert (result.fixtures_synced, result.results_synced, result.injuries_synced) == (4, 6, 0)


def test_full_refresh_of_an_espn_league_is_unchanged(monkeypatch):
    calls = _record_calls(monkeypatch)

    result = refresh.refresh_league(NL)

    assert [n for n, _ in calls] == ["seed_espn", "fixtures_espn", "results_espn_full", "injuries"]
    assert (result.results_synced, result.injuries_synced) == (99, 5)


def test_quick_refresh_of_a_football_data_league_skips_only_injuries(monkeypatch):
    calls = _record_calls(monkeypatch)

    result = refresh.refresh_league(CSV_LEAGUE, quick=True)

    assert [n for n, _ in calls] == ["fixtures_fd"]
    assert result.fixtures_synced == 7 and result.injuries_synced == 0


# --- the change fingerprint -------------------------------------------------------------


def _fingerprint_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "data_fingerprint.py"
    spec = importlib.util.spec_from_file_location("data_fingerprint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "soccer.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE matches (id INTEGER PRIMARY KEY, home_goals INT, away_goals INT, fetched_at TEXT)")
    con.execute("CREATE TABLE prediction_records (id INTEGER PRIMARY KEY)")
    con.execute("CREATE TABLE fixtures (id INTEGER PRIMARY KEY, fetched_at TEXT)")
    con.commit()
    con.close()
    return path


def _run(db, sql):
    con = sqlite3.connect(db)
    con.execute(sql)
    con.commit()
    con.close()


def test_fingerprint_changes_for_a_new_result_a_corrected_score_or_a_new_prediction(db):
    fp = _fingerprint_module().fingerprint
    start = fp(db)

    _run(db, "INSERT INTO matches (home_goals, away_goals) VALUES (2, 1)")
    after_result = fp(db)
    assert after_result != start

    _run(db, "UPDATE matches SET home_goals = 3")  # a corrected score, same row count
    after_correction = fp(db)
    assert after_correction != after_result

    _run(db, "INSERT INTO prediction_records DEFAULT VALUES")
    assert fp(db) != after_correction


def test_fingerprint_ignores_churn_that_does_not_matter(db):
    fp = _fingerprint_module().fingerprint
    _run(db, "INSERT INTO matches (home_goals, away_goals) VALUES (2, 1)")
    before = fp(db)

    _run(db, "INSERT INTO fixtures (fetched_at) VALUES ('2026-10-05')")  # timestamps/fixture refreshes
    _run(db, "UPDATE matches SET fetched_at = '2026-10-05 12:00'")
    _run(db, "VACUUM")

    assert fp(db) == before


def test_fingerprint_of_the_real_schema_runs(tmp_path):
    from soccer_predictor.storage.models import Base as RealBase

    path = tmp_path / "real.db"
    RealBase.metadata.create_all(create_engine(f"sqlite:///{path}"))
    assert _fingerprint_module().fingerprint(path) == "matches=0:0 predictions=0"
