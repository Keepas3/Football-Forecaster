"""The Champions League's club names: football-data.org (2023-24 on) and the
openfootball archive (earlier) spell the same club differently, and the alias rows in
config/team_aliases.yaml are what keep each club one team."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import load_leagues, load_team_aliases
from soccer_predictor.ingest.team_mapper import resolve, seed_teams_and_aliases
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import find_team_by_alias
from soccer_predictor.storage.team_links import linked_teams


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        seed_teams_and_aliases(s, "UCL")
        s.commit()
        yield s


def test_no_spelling_is_claimed_by_two_clubs():
    rows = [alias for alias in load_team_aliases() if alias.league == "UCL"]
    assert len(rows) > 100
    for field in ("canonical_name", "csv_name", "api_name"):
        names = [getattr(alias, field) for alias in rows]
        assert len(names) == len(set(names)), field


# (openfootball spelling, football-data.org spelling) of clubs that played in both eras.
SAME_CLUB = [
    ("Atlético Madrid", "Club Atlético de Madrid"),
    ("Bayern München", "FC Bayern München"),
    ("Inter", "FC Internazionale Milano"),
    ("Lazio Roma", "SS Lazio"),
    ("Manchester City", "Manchester City FC"),
    ("Manchester United", "Manchester United FC"),
    ("PSV Eindhoven", "PSV"),
    ("SL Benfica", "Sport Lisboa e Benfica"),
    ("Olympique Marseille", "Olympique de Marseille"),
    ("RB Salzburg", "FC Red Bull Salzburg"),
]


@pytest.mark.parametrize(("archive_name", "api_name"), SAME_CLUB)
def test_both_eras_spellings_of_a_club_are_one_team(session, archive_name, api_name):
    assert resolve(session, archive_name, "csv", "UCL") == resolve(session, api_name, "api", "UCL")


def test_lookalike_clubs_are_not_merged(session):
    ids = {
        resolve(session, name, "csv", "UCL")
        for name in ("Manchester City", "Manchester United", "Lazio Roma", "AS Roma", "Inter", "AC Milan", "Dinamo Zagreb", "Dinamo Kiev")
    }
    assert len(ids) == 8


def test_a_club_only_the_archive_knows_still_resolves_and_has_one_spelling(session):
    team_id = resolve(session, "Zenit St. Petersburg", "csv", "UCL")
    assert find_team_by_alias(session, "Zenit St. Petersburg", "api", "UCL").id == team_id


def test_old_ucl_clubs_link_to_their_domestic_team_through_the_shared_api_name(session):
    # Valencia and Hoffenheim have not played in the competition since the archive's years,
    # yet their team pages should still show those games next to their league ones.
    for league_code in ("LALIGA", "BUNDESLIGA"):
        seed_teams_and_aliases(session, league_code)
    session.commit()
    leagues = load_leagues()

    for api_name, domestic_code in (("Valencia CF", "LALIGA"), ("TSG 1899 Hoffenheim", "BUNDESLIGA")):
        ucl_team = find_team_by_alias(session, api_name, "api", "UCL")
        assert ucl_team is not None
        assert [t.league_code for t in linked_teams(session, ucl_team, leagues)] == [domestic_code]
