from __future__ import annotations

import pytest
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import live_scores
from soccer_predictor.ingest.api_client import MissingApiKey
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, upsert_team_alias

EPL = League(code="EPL", name="English Premier League", api_competition_id=2021, seasons=["2526"], csv_code="E0")
LALIGA = League(code="LALIGA", name="La Liga", api_competition_id=2014, seasons=["2526"], csv_code="SP1")
LEAGUES = {"EPL": EPL, "LALIGA": LALIGA}


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _match(
    competition_id=2021,
    home_name="Arsenal FC",
    away_name="Chelsea FC",
    home_goals=2,
    away_goals=1,
    status="IN_PLAY",
    minute=63,
    injury_time=None,
):
    return {
        "competition": {"id": competition_id},
        "homeTeam": {"name": home_name},
        "awayTeam": {"name": away_name},
        "score": {"fullTime": {"home": home_goals, "away": away_goals}},
        "status": status,
        "minute": minute,
        "injuryTime": injury_time,
    }


def test_fetch_live_matches_maps_competition_id_to_league_code(session, monkeypatch):
    monkeypatch.setattr(live_scores, "load_leagues", lambda: LEAGUES)
    monkeypatch.setattr(live_scores.api_client, "get", lambda *a, **k: {"matches": [_match(competition_id=2021)]})

    results = live_scores.fetch_live_matches_football_data_org(session)

    assert len(results) == 1
    assert results[0].league_code == "EPL"
    assert results[0].home_score == 2
    assert results[0].away_score == 1


def test_fetch_live_matches_skips_untracked_competition(session, monkeypatch):
    monkeypatch.setattr(live_scores, "load_leagues", lambda: LEAGUES)
    monkeypatch.setattr(
        live_scores.api_client, "get", lambda *a, **k: {"matches": [_match(competition_id=99999)]}
    )

    assert live_scores.fetch_live_matches_football_data_org(session) == []


def test_fetch_live_matches_resolves_team_name_via_alias(session, monkeypatch):
    monkeypatch.setattr(live_scores, "load_leagues", lambda: LEAGUES)
    team = get_or_create_team(session, "Arsenal", "EPL")
    team.crest_url = "https://example.test/arsenal.png"
    upsert_team_alias(session, team, "Arsenal FC", source="api")
    session.commit()

    monkeypatch.setattr(
        live_scores.api_client, "get", lambda *a, **k: {"matches": [_match(home_name="Arsenal FC")]}
    )

    results = live_scores.fetch_live_matches_football_data_org(session)

    assert results[0].home_name == "Arsenal"
    assert results[0].home_crest == "https://example.test/arsenal.png"


def test_fetch_live_matches_falls_back_to_raw_name_when_unresolved(session, monkeypatch):
    monkeypatch.setattr(live_scores, "load_leagues", lambda: LEAGUES)
    monkeypatch.setattr(
        live_scores.api_client, "get", lambda *a, **k: {"matches": [_match(home_name="Some New Club FC")]}
    )

    results = live_scores.fetch_live_matches_football_data_org(session)

    assert results[0].home_name == "Some New Club FC"
    assert results[0].home_crest is None


def test_fetch_live_matches_returns_empty_on_missing_api_key(session, monkeypatch):
    monkeypatch.setattr(live_scores, "load_leagues", lambda: LEAGUES)

    def raise_missing_key(*a, **k):
        raise MissingApiKey("no key")

    monkeypatch.setattr(live_scores.api_client, "get", raise_missing_key)

    assert live_scores.fetch_live_matches_football_data_org(session) == []


def test_fetch_live_matches_returns_empty_on_request_exception(session, monkeypatch):
    monkeypatch.setattr(live_scores, "load_leagues", lambda: LEAGUES)

    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(live_scores.api_client, "get", raise_error)

    assert live_scores.fetch_live_matches_football_data_org(session) == []


def test_fetch_live_matches_returns_empty_when_nothing_live(session, monkeypatch):
    monkeypatch.setattr(live_scores, "load_leagues", lambda: LEAGUES)
    monkeypatch.setattr(live_scores.api_client, "get", lambda *a, **k: {"matches": []})

    assert live_scores.fetch_live_matches_football_data_org(session) == []


@pytest.mark.parametrize(
    "status,minute,injury_time,expected",
    [
        ("IN_PLAY", 63, None, "63'"),
        ("IN_PLAY", 90, 3, "90+3'"),
        ("PAUSED", 45, None, "HT"),
        ("IN_PLAY", None, None, "Live"),
    ],
)
def test_clock_label_formatting(status, minute, injury_time, expected):
    assert live_scores._clock_label(status, minute, injury_time) == expected


def test_fetch_all_live_matches_combines_football_data_org_and_espn(session, monkeypatch):
    monkeypatch.setattr(live_scores, "load_leagues", lambda: LEAGUES)
    monkeypatch.setattr(live_scores.api_client, "get", lambda *a, **k: {"matches": [_match()]})

    mls = League(code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1")

    def fake_espn(*a, **k):
        return live_scores.LiveMatch(
            league_code="MLS",
            home_name="Atlanta United FC",
            away_name="Inter Miami CF",
            home_crest=None,
            away_crest=None,
            home_score=1,
            away_score=1,
            clock_label="70'",
        )

    monkeypatch.setattr(live_scores, "fetch_live_matches_espn", lambda league: [fake_espn()])

    results = live_scores.fetch_all_live_matches(session, {"EPL": EPL, "MLS": mls})

    assert {m.league_code for m in results} == {"EPL", "MLS"}


def test_espn_live_fetch_bypasses_the_six_hour_scoreboard_cache(monkeypatch):
    # With the default 6h cache the banner froze on whichever games were live
    # at first load and never showed later kickoffs.
    from soccer_predictor.ingest import live_scores

    seen = {}

    def fake_fetch(slug, day, cache_ttl_seconds=None):
        seen["ttl"] = cache_ttl_seconds
        return []

    monkeypatch.setattr(live_scores.espn_client, "fetch_day_matches", fake_fetch)
    league = League(code="NL", name="UEFA Nations League", seasons=["2627"], data_source="espn", espn_league_slug="uefa.nations")

    live_scores.fetch_live_matches_espn(league)

    assert seen["ttl"] == live_scores.LIVE_SCORE_CACHE_TTL_SECONDS
