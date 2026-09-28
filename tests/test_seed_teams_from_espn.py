from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import espn_client, team_mapper
from soccer_predictor.storage.models import Base, Team
from soccer_predictor.storage.repository import teams_for_league

LEAGUE = League(
    code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1"
)

ESPN_TEAMS = [
    espn_client.EspnTeam(espn_id="183", display_name="Columbus Crew", logo_url="https://example.com/crew.png"),
    espn_client.EspnTeam(espn_id="20232", display_name="Inter Miami CF", logo_url=None),
]


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


@pytest.fixture(autouse=True)
def no_conference_by_default(monkeypatch):
    # Isolates tests that don't care about conference from espn_client's
    # own internals -- individual tests override this where it matters.
    monkeypatch.setattr(espn_client, "fetch_team_conference", lambda *a, **k: None)


def test_seeds_teams_crest_and_espn_id(monkeypatch):
    monkeypatch.setattr(espn_client, "fetch_teams", lambda *a, **k: ESPN_TEAMS)
    monkeypatch.setattr(
        espn_client,
        "fetch_team_conference",
        lambda slug, espn_id: "Eastern Conference" if espn_id == "183" else "Western Conference",
    )
    with _session() as s:
        count = team_mapper.seed_teams_from_espn(s, LEAGUE)
        s.commit()

        assert count == 2
        assert set(teams_for_league(s, "MLS").values()) == {"Columbus Crew", "Inter Miami CF"}
        crew = s.query(Team).filter_by(canonical_name="Columbus Crew").one()
        assert crew.espn_team_id == 183
        assert crew.crest_url == "https://example.com/crew.png"
        assert crew.conference == "Eastern Conference"
        miami = s.query(Team).filter_by(canonical_name="Inter Miami CF").one()
        assert miami.conference == "Western Conference"


def test_seeding_twice_is_idempotent(monkeypatch):
    monkeypatch.setattr(espn_client, "fetch_teams", lambda *a, **k: ESPN_TEAMS)
    with _session() as s:
        team_mapper.seed_teams_from_espn(s, LEAGUE)
        team_mapper.seed_teams_from_espn(s, LEAGUE)
        s.commit()

        assert len(teams_for_league(s, "MLS")) == 2


def test_returns_zero_when_espn_has_nothing(monkeypatch):
    # espn_client.fetch_teams already degrades a RequestException (or any
    # other failure) to [] itself (see test_espn_client.py) -- this is what
    # seed_teams_from_espn actually sees at its own boundary.
    monkeypatch.setattr(espn_client, "fetch_teams", lambda *a, **k: [])
    with _session() as s:
        assert team_mapper.seed_teams_from_espn(s, LEAGUE) == 0
