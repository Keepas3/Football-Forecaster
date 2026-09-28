from __future__ import annotations

import datetime as dt

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import (
    get_or_create_team,
    live_fixtures_across_leagues,
    upsert_fixture,
)

WINDOW = dt.timedelta(hours=2, minutes=15)


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def test_includes_live_games_from_every_league():
    with _session() as s:
        now = dt.datetime(2026, 10, 10, 15, 0)

        epl_home = get_or_create_team(s, "Man City", "EPL")
        epl_away = get_or_create_team(s, "Man United", "EPL")
        laliga_home = get_or_create_team(s, "Real Madrid", "LALIGA")
        laliga_away = get_or_create_team(s, "Barcelona", "LALIGA")
        s.commit()

        # EPL game kicked off 30 min ago -- live.
        upsert_fixture(
            s, "EPL", now.date(), epl_home.id, epl_away.id, "SCHEDULED",
            kickoff_utc=now - dt.timedelta(minutes=30),
        )
        # La Liga game kicked off 3 hours ago -- already over, outside window.
        upsert_fixture(
            s, "LALIGA", now.date(), laliga_home.id, laliga_away.id, "SCHEDULED",
            kickoff_utc=now - dt.timedelta(hours=3),
        )
        s.commit()

        df = live_fixtures_across_leagues(s, now, WINDOW)

        assert len(df) == 1
        assert df.iloc[0]["league_code"] == "EPL"
        assert df.iloc[0]["home_team_id"] == epl_home.id
        assert df.iloc[0]["away_team_id"] == epl_away.id


def test_excludes_future_and_null_kickoff_fixtures():
    with _session() as s:
        now = dt.datetime(2026, 10, 10, 15, 0)
        home = get_or_create_team(s, "Man City", "EPL")
        away = get_or_create_team(s, "Man United", "EPL")
        s.commit()

        upsert_fixture(s, "EPL", now.date(), home.id, away.id, "SCHEDULED", kickoff_utc=now + dt.timedelta(hours=1))
        upsert_fixture(s, "EPL", now.date() + dt.timedelta(days=1), away.id, home.id, "SCHEDULED", kickoff_utc=None)
        s.commit()

        df = live_fixtures_across_leagues(s, now, WINDOW)
        assert df.empty


def test_kickoff_at_exact_window_boundary_is_included():
    with _session() as s:
        now = dt.datetime(2026, 10, 10, 15, 0)
        home = get_or_create_team(s, "Man City", "EPL")
        away = get_or_create_team(s, "Man United", "EPL")
        s.commit()

        upsert_fixture(s, "EPL", now.date(), home.id, away.id, "SCHEDULED", kickoff_utc=now - WINDOW)
        s.commit()

        df = live_fixtures_across_leagues(s, now, WINDOW)
        assert len(df) == 1
