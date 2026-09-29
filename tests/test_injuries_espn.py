from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import asa_client, espn_client, injuries
from soccer_predictor.ingest.asa_client import AsaPlayerStats
from soccer_predictor.ingest.player_importance import DEFAULT_API_IMPORTANCE_WEIGHT
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, injuries_for_team, update_espn_team_id

LEAGUE = League(
    code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1"
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        crew = get_or_create_team(s, "Columbus Crew", "MLS")
        update_espn_team_id(s, crew.id, 183)
        s.commit()
        yield s


@pytest.fixture(autouse=True)
def no_asa_by_default(monkeypatch):
    monkeypatch.setattr(asa_client, "fetch_team_season", lambda *a, **k: [])


def _crew_injury(session):
    team = get_or_create_team(session, "Columbus Crew", "MLS")
    return injuries_for_team(session, team.id)[0]


def test_syncs_espn_injury_with_position_bucket_from_roster(session, monkeypatch):
    entry = espn_client.EspnInjuryEntry(player_name="Bukayo Saka", position_bucket="attack", note="Hamstring")
    monkeypatch.setattr(espn_client, "fetch_team_roster", lambda *a, **k: ([], [entry]))

    synced, skipped = injuries.sync_injuries_to_db(session, LEAGUE, 2026)
    session.commit()

    assert synced == 1
    assert skipped == 0
    row = _crew_injury(session)
    assert row.player_name == "Bukayo Saka"
    assert row.position == "attack"
    assert row.note == "Hamstring"


def test_no_injuries_still_counts_as_synced(session, monkeypatch):
    monkeypatch.setattr(espn_client, "fetch_team_roster", lambda *a, **k: ([], []))

    synced, skipped = injuries.sync_injuries_to_db(session, LEAGUE, 2026)

    assert synced == 1
    assert skipped == 0


def test_skips_team_without_espn_id(session, monkeypatch):
    get_or_create_team(session, "Some Other Club", "MLS")
    session.commit()
    monkeypatch.setattr(espn_client, "fetch_team_roster", lambda *a, **k: ([], []))

    synced, skipped = injuries.sync_injuries_to_db(session, LEAGUE, 2026)

    # Columbus Crew (has an espn_team_id) syncs fine; "Some Other Club"
    # (no espn_team_id) is skipped.
    assert synced == 1
    assert skipped == 1


def test_importance_weight_computed_from_real_asa_stats_when_available(session, monkeypatch):
    entry = espn_client.EspnInjuryEntry(player_name="Bukayo Saka", position_bucket="attack", note="Hamstring")
    monkeypatch.setattr(espn_client, "fetch_team_roster", lambda *a, **k: ([], [entry]))
    monkeypatch.setattr(
        asa_client,
        "fetch_team_season",
        lambda *a, **k: [
            AsaPlayerStats(
                name="Bukayo Saka",
                position="W",
                minutes=2000,
                goals=15,
                assists=10,
                xg=12.0,
                xa=8.0,
                shots=60,
                key_passes=30,
                points_added=3.0,
            )
        ],
    )

    injuries.sync_injuries_to_db(session, LEAGUE, 2026)
    session.commit()

    assert _crew_injury(session).importance_weight != DEFAULT_API_IMPORTANCE_WEIGHT


def test_importance_weight_falls_back_to_default_when_no_stats(session, monkeypatch):
    entry = espn_client.EspnInjuryEntry(player_name="Bukayo Saka", position_bucket="attack", note="Hamstring")
    monkeypatch.setattr(espn_client, "fetch_team_roster", lambda *a, **k: ([], [entry]))

    injuries.sync_injuries_to_db(session, LEAGUE, 2026)
    session.commit()

    assert _crew_injury(session).importance_weight == DEFAULT_API_IMPORTANCE_WEIGHT


def test_non_espn_league_is_a_no_op(session):
    non_espn_league = League(code="EPL", name="English Premier League", seasons=["2526"], csv_code="E0")

    synced, skipped = injuries.sync_injuries_to_db(session, non_espn_league, 2026)

    assert (synced, skipped) == (0, 0)
