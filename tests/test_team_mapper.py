from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.ingest.team_mapper import UnresolvedTeamName, resolve
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, upsert_team_alias


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        man_utd = get_or_create_team(s, "Man United", "EPL")
        upsert_team_alias(s, man_utd, "Man United", source="csv")
        upsert_team_alias(s, man_utd, "Manchester United FC", source="api")

        man_city = get_or_create_team(s, "Man City", "EPL")
        upsert_team_alias(s, man_city, "Man City", source="csv")
        upsert_team_alias(s, man_city, "Manchester City FC", source="api")

        s.commit()
        yield s


def test_exact_match(session):
    man_utd = get_or_create_team(session, "Man United", "EPL")
    assert resolve(session, "Man United", "csv", league_code="EPL") == man_utd.id


def test_fuzzy_match_close_typo_resolves(session):
    man_utd = get_or_create_team(session, "Man United", "EPL")
    # A minor typo shouldn't be treated as a different, unmatched team.
    assert resolve(session, "Man Untied", "csv", league_code="EPL") == man_utd.id


def test_ambiguous_name_raises_instead_of_guessing(session, tmp_path, monkeypatch):
    import soccer_predictor.ingest.team_mapper as team_mapper

    monkeypatch.setattr(team_mapper, "UNMATCHED_LOG", tmp_path / "unmatched.log")

    with pytest.raises(UnresolvedTeamName):
        resolve(session, "Totally Unrelated FC", "csv", league_code="EPL")

    assert (tmp_path / "unmatched.log").exists()


def test_fuzzy_match_does_not_cross_leagues(session):
    """A name should only fuzzy-match within its own league, even when a
    different league's alias is a textually closer match.

    "Man Untied FC" (LALIGA, scored below) is a *closer* WRatio match to the
    query than EPL's real "Man United" -- if fuzzy matching weren't scoped
    by league, the LALIGA lookalike would incorrectly win.
    """
    man_utd = get_or_create_team(session, "Man United", "EPL")

    laliga_lookalike = get_or_create_team(session, "Man Untied FC", "LALIGA")
    upsert_team_alias(session, laliga_lookalike, "Man Untied FC", source="csv")
    session.commit()

    assert resolve(session, "Man Untied", "csv", league_code="EPL") == man_utd.id
