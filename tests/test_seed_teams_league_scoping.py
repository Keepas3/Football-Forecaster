from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import soccer_predictor.ingest.team_mapper as team_mapper
from soccer_predictor.config import TeamAlias
from soccer_predictor.storage.models import Base, Team
from soccer_predictor.storage.repository import all_alias_texts, teams_for_league

FAKE_ALIASES = [
    TeamAlias(canonical_name="Arsenal", csv_name="Arsenal", api_name="Arsenal FC", league="EPL"),
    TeamAlias(canonical_name="Man City", csv_name="Man City", api_name="Manchester City FC", league="EPL"),
    TeamAlias(canonical_name="Real Madrid", csv_name="Real Madrid", api_name="Real Madrid CF", league="LALIGA"),
    TeamAlias(canonical_name="Barcelona", csv_name="Barcelona", api_name="FC Barcelona", league="LALIGA"),
]


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(team_mapper, "load_team_aliases", lambda: FAKE_ALIASES)
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_seeding_one_league_does_not_create_other_leagues_teams(session):
    team_mapper.seed_teams_and_aliases(session, "EPL")
    session.commit()

    epl_teams = teams_for_league(session, "EPL")
    laliga_teams = teams_for_league(session, "LALIGA")

    assert set(epl_teams.values()) == {"Arsenal", "Man City"}
    assert laliga_teams == {}


def test_seeding_both_leagues_keeps_them_separate(session):
    team_mapper.seed_teams_and_aliases(session, "EPL")
    team_mapper.seed_teams_and_aliases(session, "LALIGA")
    session.commit()

    epl_teams = teams_for_league(session, "EPL")
    laliga_teams = teams_for_league(session, "LALIGA")

    assert set(epl_teams.values()) == {"Arsenal", "Man City"}
    assert set(laliga_teams.values()) == {"Real Madrid", "Barcelona"}

    # And no cross-league aliases were created either.
    epl_csv_aliases = {text for text, _ in all_alias_texts(session, "csv", "EPL")}
    laliga_csv_aliases = {text for text, _ in all_alias_texts(session, "csv", "LALIGA")}
    assert epl_csv_aliases == {"Arsenal", "Man City"}
    assert laliga_csv_aliases == {"Real Madrid", "Barcelona"}


def test_seeding_same_league_twice_is_idempotent(session):
    team_mapper.seed_teams_and_aliases(session, "EPL")
    team_mapper.seed_teams_and_aliases(session, "EPL")
    session.commit()

    all_teams = session.scalars(select(Team)).all()
    assert len(all_teams) == 2
