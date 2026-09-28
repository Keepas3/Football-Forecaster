from __future__ import annotations

import datetime as dt

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import api_client
from soccer_predictor.ingest.fixtures import sync_results_to_db
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, matches_for_league, upsert_team_alias

LEAGUE = League(code="UCL", name="UEFA Champions League", api_competition_id=2001, seasons=["2425"])

RAW_RESPONSE = {
    "matches": [
        {
            "utcDate": "2024-09-17T19:00:00Z",
            "status": "FINISHED",
            "homeTeam": {"name": "Manchester City FC", "crest": None},
            "awayTeam": {"name": "Inter FC", "crest": None},
            "score": {"fullTime": {"home": 2, "away": 0}},
        },
        {
            # No score -- e.g. postponed/abandoned -- must be skipped rather
            # than upserted as a 0-0 or crashing on a missing key.
            "utcDate": "2024-09-18T19:00:00Z",
            "status": "FINISHED",
            "homeTeam": {"name": "Some Team"},
            "awayTeam": {"name": "Other Team"},
            "score": {"fullTime": {"home": None, "away": None}},
        },
    ]
}


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def test_sync_results_to_db_upserts_finished_matches(monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *args, **kwargs: RAW_RESPONSE)
    with _session() as s:
        city = get_or_create_team(s, "Manchester City", "UCL")
        upsert_team_alias(s, city, "Manchester City FC", source="api")
        inter = get_or_create_team(s, "Inter", "UCL")
        upsert_team_alias(s, inter, "Inter FC", source="api")
        s.commit()

        synced, skipped = sync_results_to_db(s, LEAGUE, "2425")
        s.commit()

        assert synced == 1
        assert skipped == 0
        df = matches_for_league(s, "UCL")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["home_goals"] == 2
        assert row["away_goals"] == 0
        assert row["date"] == dt.date(2024, 9, 17)


def test_sync_results_to_db_skips_unresolved_teams(monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *args, **kwargs: RAW_RESPONSE)
    with _session() as s:
        synced, skipped = sync_results_to_db(s, LEAGUE, "2425")

        # The no-score match is filtered out before resolution is even
        # attempted, so only the first (unresolved, no teams seeded) match
        # counts toward `skipped`.
        assert synced == 0
        assert skipped == 1
