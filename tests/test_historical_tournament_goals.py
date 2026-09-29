from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import (
    get_or_create_team,
    goals_for_team_season,
    replace_historical_tournament_goals,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_replace_historical_tournament_goals_then_read_back(session):
    team = get_or_create_team(session, "England", "WC")
    replace_historical_tournament_goals(
        session,
        "WC",
        "1966",
        [
            {"team_id": team.id, "player_name": "Geoff Hurst", "minute": 18, "own_goal": False, "penalty": False},
            {"team_id": team.id, "player_name": "Geoff Hurst", "minute": 101, "own_goal": False, "penalty": False},
            {"team_id": team.id, "player_name": "Geoff Hurst", "minute": 120, "own_goal": False, "penalty": False},
        ],
        source="worldcup_archive",
    )

    rows = goals_for_team_season(session, team.id, "WC", "1966")
    assert len(rows) == 3
    assert all(r.player_name == "Geoff Hurst" for r in rows)
    assert {r.minute for r in rows} == {18, 101, 120}
    assert all(r.source == "worldcup_archive" for r in rows)


def test_goals_for_team_season_scoped_to_league_and_season(session):
    team = get_or_create_team(session, "England", "WC")
    replace_historical_tournament_goals(
        session, "WC", "1966", [{"team_id": team.id, "player_name": "Hurst"}], source="worldcup_archive"
    )
    replace_historical_tournament_goals(
        session, "WC", "1970", [{"team_id": team.id, "player_name": "Someone Else"}], source="worldcup_archive"
    )

    rows_1966 = goals_for_team_season(session, team.id, "WC", "1966")
    assert [r.player_name for r in rows_1966] == ["Hurst"]


def test_replace_historical_tournament_goals_is_idempotent_not_additive(session):
    team = get_or_create_team(session, "England", "WC")
    goals = [{"team_id": team.id, "player_name": "Hurst"}]
    replace_historical_tournament_goals(session, "WC", "1966", goals, source="worldcup_archive")
    replace_historical_tournament_goals(session, "WC", "1966", goals, source="worldcup_archive")

    rows = goals_for_team_season(session, team.id, "WC", "1966")
    assert len(rows) == 1


def test_replace_historical_tournament_goals_defaults_own_goal_and_penalty_false(session):
    team = get_or_create_team(session, "England", "WC")
    replace_historical_tournament_goals(
        session, "WC", "1966", [{"team_id": team.id, "player_name": "Hurst"}], source="worldcup_archive"
    )
    row = goals_for_team_season(session, team.id, "WC", "1966")[0]
    assert row.own_goal is False
    assert row.penalty is False
    assert row.minute is None
    assert row.match_date is None
