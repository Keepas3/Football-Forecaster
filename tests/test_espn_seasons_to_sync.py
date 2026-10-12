"""A full ESPN historical sync skips past seasons that are already stored."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest.fixtures import espn_seasons_to_sync
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, match_count_for_season, upsert_match

LEAGUE = League(
    code="ASIA",
    name="AFC Asian Cup",
    seasons=["2015", "2019", "2023", "2027"],
    data_source="espn",
    espn_league_slug="afc.asian.cup",
)


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def _store_match(s, season: str, day: dt.date) -> None:
    home, away = get_or_create_team(s, "Japan", "ASIA"), get_or_create_team(s, "Iran", "ASIA")
    upsert_match(s, "ASIA", season, day, home.id, away.id, 1, 0, source="espn")
    s.commit()


def test_first_run_syncs_every_season():
    with _session() as s:
        assert espn_seasons_to_sync(s, LEAGUE) == ["2015", "2019", "2023", "2027"]


def test_stored_past_seasons_are_skipped_but_the_latest_never_is():
    with _session() as s:
        _store_match(s, "2015", dt.date(2015, 1, 10))
        _store_match(s, "2023", dt.date(2024, 1, 14))
        _store_match(s, "2027", dt.date(2027, 1, 12))  # latest: still synced
        assert match_count_for_season(s, "ASIA", "2015") == 1
        # 2019 has nothing stored (e.g. an earlier attempt failed), so it is retried.
        assert espn_seasons_to_sync(s, LEAGUE) == ["2019", "2027"]


def test_all_seasons_forces_the_full_rewalk():
    with _session() as s:
        _store_match(s, "2015", dt.date(2015, 1, 10))
        assert espn_seasons_to_sync(s, LEAGUE, all_seasons=True) == ["2015", "2019", "2023", "2027"]
