from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.ingest.team_mapper import UnresolvedTeamName, resolve, seed_teams_from_names
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import (
    all_alias_texts,
    get_or_create_team,
    teams_for_league,
    upsert_team_alias,
)


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


def test_seed_teams_from_names_creates_a_team_per_name(session):
    count = seed_teams_from_names(session, "WC", {"England", "France", "Brazil"})

    assert count == 3
    names = set(teams_for_league(session, "WC").values())
    assert names == {"England", "France", "Brazil"}


def test_seed_teams_from_names_bootstraps_resolve_with_no_prior_aliases(session):
    # resolve()'s fuzzy fallback can never seed a brand-new league from
    # nothing (it needs at least one existing alias to fuzzy-match
    # against) -- this is the real reason seed_teams_from_names exists.
    seed_teams_from_names(session, "WC", {"England"})

    team_id = resolve(session, "England", "csv", league_code="WC")
    england = get_or_create_team(session, "England", "WC")
    assert team_id == england.id


def test_seed_teams_from_names_registers_a_csv_alias(session):
    seed_teams_from_names(session, "WC", {"Brazil"})

    aliases = dict(all_alias_texts(session, "csv", "WC"))
    assert "Brazil" in aliases


def test_seed_teams_from_names_is_idempotent(session):
    seed_teams_from_names(session, "WC", {"England"})
    seed_teams_from_names(session, "WC", {"England"})

    assert len(teams_for_league(session, "WC")) == 1
