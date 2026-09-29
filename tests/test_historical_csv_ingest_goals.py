from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.ingest.historical_csv import ingest_goals_into_db
from soccer_predictor.ingest.team_mapper import seed_teams_from_names
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, goals_for_team_season


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_ingest_goals_into_db_resolves_team_and_persists(session):
    seed_teams_from_names(session, "WC", {"England", "West Germany"})

    ingested, skipped = ingest_goals_into_db(
        session,
        "WC",
        "1966",
        [
            {"team_name": "England", "player_name": "Geoff Hurst", "minute": 18},
            {"team_name": "England", "player_name": "Geoff Hurst", "minute": 101},
        ],
        source="worldcup_archive",
    )

    assert ingested == 2
    assert skipped == 0
    england = get_or_create_team(session, "England", "WC")
    rows = goals_for_team_season(session, england.id, "WC", "1966")
    assert len(rows) == 2
    assert all(r.source == "worldcup_archive" for r in rows)


def test_ingest_goals_into_db_skips_unresolvable_team_name(session, tmp_path, monkeypatch):
    import soccer_predictor.ingest.team_mapper as team_mapper

    monkeypatch.setattr(team_mapper, "UNMATCHED_LOG", tmp_path / "unmatched.log")
    seed_teams_from_names(session, "WC", {"England"})

    ingested, skipped = ingest_goals_into_db(
        session,
        "WC",
        "1966",
        [{"team_name": "Nonexistent Nation FC", "player_name": "Nobody", "minute": 1}],
        source="worldcup_archive",
    )

    assert ingested == 0
    assert skipped == 1


def test_ingest_goals_into_db_re_ingest_replaces_not_appends(session):
    seed_teams_from_names(session, "WC", {"England"})
    row = {"team_name": "England", "player_name": "Hurst", "minute": 18}

    ingest_goals_into_db(session, "WC", "1966", [row], source="worldcup_archive")
    ingest_goals_into_db(session, "WC", "1966", [row], source="worldcup_archive")

    england = get_or_create_team(session, "England", "WC")
    rows = goals_for_team_season(session, england.id, "WC", "1966")
    assert len(rows) == 1
