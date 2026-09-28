"""Regression tests for letting a team belong to more than one league (see
scripts/migrate_multi_league_teams.py) -- before this, Team.canonical_name
was globally unique, so a club already tracked under a domestic league (e.g.
Manchester City under EPL) could never get its own row for UEFA Champions
League; get_or_create_team would just silently return the EPL row.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import (
    find_team_by_alias,
    get_or_create_team,
    upsert_team_alias,
)


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def test_same_canonical_name_gets_separate_rows_per_league():
    with _session() as s:
        epl_city = get_or_create_team(s, "Manchester City", "EPL")
        ucl_city = get_or_create_team(s, "Manchester City", "UCL")
        s.commit()

        assert epl_city.id != ucl_city.id
        assert epl_city.league_code == "EPL"
        assert ucl_city.league_code == "UCL"


def test_get_or_create_team_is_idempotent_within_a_league():
    with _session() as s:
        first = get_or_create_team(s, "Arsenal", "EPL")
        second = get_or_create_team(s, "Arsenal", "EPL")
        s.commit()

        assert first.id == second.id


def test_alias_scoped_by_league_resolves_to_the_right_team():
    with _session() as s:
        epl_city = get_or_create_team(s, "Manchester City", "EPL")
        ucl_city = get_or_create_team(s, "Manchester City", "UCL")
        upsert_team_alias(s, epl_city, "Manchester City FC", source="api")
        upsert_team_alias(s, ucl_city, "Manchester City FC", source="api")
        s.commit()

        assert find_team_by_alias(s, "Manchester City FC", "api", "EPL").id == epl_city.id
        assert find_team_by_alias(s, "Manchester City FC", "api", "UCL").id == ucl_city.id


def test_find_team_by_alias_returns_none_for_wrong_league():
    with _session() as s:
        epl_city = get_or_create_team(s, "Manchester City", "EPL")
        upsert_team_alias(s, epl_city, "Manchester City FC", source="api")
        s.commit()

        assert find_team_by_alias(s, "Manchester City FC", "api", "UCL") is None
