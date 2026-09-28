from __future__ import annotations

import datetime as dt

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import espn_client
from soccer_predictor.ingest.fixtures import sync_fixtures_to_db_espn, sync_results_to_db_espn
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import (
    fixtures_for_league,
    get_or_create_team,
    matches_for_league,
    update_espn_team_id,
)

LEAGUE = League(
    code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1"
)


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def _match(home_id, away_id, home_name, away_name, completed, home_score=None, away_score=None):
    today = dt.date.today()
    return espn_client.EspnMatch(
        espn_home_id=home_id,
        espn_away_id=away_id,
        home_name=home_name,
        away_name=away_name,
        date=today,
        kickoff_utc=dt.datetime(today.year, today.month, today.day, 23, 0),
        completed=completed,
        home_score=home_score,
        away_score=away_score,
    )


def test_sync_results_to_db_espn_upserts_finished_matches(monkeypatch):
    with _session() as s:
        crew = get_or_create_team(s, "Columbus Crew", "MLS")
        update_espn_team_id(s, crew.id, 183)
        miami = get_or_create_team(s, "Inter Miami CF", "MLS")
        update_espn_team_id(s, miami.id, 20232)
        s.commit()

        match = _match("183", "20232", "Columbus Crew", "Inter Miami CF", completed=True, home_score=2, away_score=1)
        monkeypatch.setattr(espn_client, "fetch_team_results", lambda *a, **k: [match])

        synced, skipped = sync_results_to_db_espn(s, LEAGUE, "2026")
        s.commit()

        # A match appears in both teams' schedule fetches -- upsert_match's
        # existing (league, date, home, away) keying makes the second write
        # a no-op, not a duplicate row.
        assert synced == 2
        assert skipped == 0
        df = matches_for_league(s, "MLS")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["home_goals"] == 2
        assert row["away_goals"] == 1


def test_sync_results_to_db_espn_skips_non_seeded_opponent(monkeypatch):
    with _session() as s:
        crew = get_or_create_team(s, "Columbus Crew", "MLS")
        update_espn_team_id(s, crew.id, 183)
        s.commit()

        # Opponent id 999 was never seeded as an MLS team (e.g. a cross-
        # competition opponent) -- must be skipped, not upserted.
        match = _match("183", "999", "Columbus Crew", "Some Other Club", completed=True, home_score=3, away_score=0)
        monkeypatch.setattr(espn_client, "fetch_team_results", lambda *a, **k: [match])

        synced, skipped = sync_results_to_db_espn(s, LEAGUE, "2026")

        assert synced == 0
        assert skipped == 1
        assert len(matches_for_league(s, "MLS")) == 0


def test_sync_fixtures_to_db_espn_upserts_upcoming_matches(monkeypatch):
    with _session() as s:
        crew = get_or_create_team(s, "Columbus Crew", "MLS")
        update_espn_team_id(s, crew.id, 183)
        miami = get_or_create_team(s, "Inter Miami CF", "MLS")
        update_espn_team_id(s, miami.id, 20232)
        s.commit()

        match = _match("183", "20232", "Columbus Crew", "Inter Miami CF", completed=False)
        # Only the first scanned day returns a fixture -- every other day in
        # the 14-day window returns nothing, same as a real quiet MLS week.
        calls = {"count": 0}

        def fake_fetch_day(league_slug, date):
            calls["count"] += 1
            return [match] if calls["count"] == 1 else []

        monkeypatch.setattr(espn_client, "fetch_day_fixtures", fake_fetch_day)

        synced, skipped = sync_fixtures_to_db_espn(s, LEAGUE)
        s.commit()

        assert synced == 1
        assert skipped == 0
        start = dt.date.today()
        df = fixtures_for_league(s, "MLS", start, start + dt.timedelta(days=30))
        assert len(df) == 1
