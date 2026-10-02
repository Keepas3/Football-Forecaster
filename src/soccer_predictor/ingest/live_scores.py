"""Real live scores for matches currently in progress -- a display-time-
only fetch (never persisted to `matches`/`fixtures`) for the Leagues page's
top "LIVE NOW" banner, same pattern as Understat/ASA's current-season
stats: short-TTL disk cache, degrade to empty on any failure, never raises.

football-data.org's free tier is a hard 10 requests/minute
(ingest/api_client.py) -- the existing rule there is "the dashboard must
never trigger a live call per page render." This module is a deliberate,
narrow exception to that: ONE global call (no competition scoping) covers
every football-data.org-backed league at once, cached 30s and shared
across every concurrent viewer (Streamlit Cloud runs one process), so
repeated dashboard reruns during a live match stay far under budget.

Confirmed live (2026-09) our existing free key already has access to
`GET /matches?status=LIVE`. Confirmed via football-data.org's own docs:
`score.fullTime` is the ACTUAL RUNNING SCORE while a match is
IN_PLAY/PAUSED (0-0 at kickoff, updates as goals happen), not a final-
score-only field despite the name -- `minute`/`injuryTime` track elapsed
time.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests
from sqlalchemy.orm import Session

from soccer_predictor.config import League, load_leagues
from soccer_predictor.ingest import api_client, espn_client
from soccer_predictor.storage.repository import find_team_by_alias

LIVE_SCORE_CACHE_TTL_SECONDS = 30


@dataclass
class LiveMatch:
    league_code: str
    home_name: str  # resolved to our canonical name when possible
    away_name: str
    home_crest: str | None
    away_crest: str | None
    home_score: int
    away_score: int
    clock_label: str  # "63'", "HT", "45+2'", or "Live" if unknown


def _resolve_display_name(session: Session, api_name: str, league_code: str) -> tuple[str, str | None]:
    """Returns (display_name, crest_url) -- our own canonical name/crest via
    the existing team_aliases table (read-only, no writes) when resolvable,
    otherwise the raw API name with no crest. Never raises.
    """
    team = find_team_by_alias(session, api_name, source="api", league_code=league_code)
    if team is not None:
        return team.canonical_name, team.crest_url
    return api_name, None


def _clock_label(status: str, minute, injury_time) -> str:
    if status == "PAUSED":
        return "HT"
    if not isinstance(minute, int):
        return "Live"
    if injury_time:
        return f"{minute}+{injury_time}'"
    return f"{minute}'"


def fetch_live_matches_football_data_org(session: Session) -> list[LiveMatch]:
    """One global call covers every football-data.org-backed league this
    app tracks at once. Empty list on any failure or when nothing's live.
    """
    try:
        data = api_client.get(
            "/matches", params={"status": "LIVE"}, cache_ttl_seconds=LIVE_SCORE_CACHE_TTL_SECONDS
        )
    except (api_client.MissingApiKey, requests.RequestException):
        return []

    competition_id_to_league = {
        league.api_competition_id: league.code
        for league in load_leagues().values()
        if league.api_competition_id is not None
    }

    results = []
    for match in data.get("matches", []):
        try:
            league_code = competition_id_to_league.get(match["competition"]["id"])
            if league_code is None:
                continue  # a live match in a competition this app doesn't track

            home_name, home_crest = _resolve_display_name(session, match["homeTeam"]["name"], league_code)
            away_name, away_crest = _resolve_display_name(session, match["awayTeam"]["name"], league_code)
            full_time = match.get("score", {}).get("fullTime", {})

            results.append(
                LiveMatch(
                    league_code=league_code,
                    home_name=home_name,
                    away_name=away_name,
                    home_crest=home_crest,
                    away_crest=away_crest,
                    home_score=full_time.get("home") or 0,
                    away_score=full_time.get("away") or 0,
                    clock_label=_clock_label(match.get("status", ""), match.get("minute"), match.get("injuryTime")),
                )
            )
        except (KeyError, TypeError):
            continue
    return results


def fetch_live_matches_espn(league: League) -> list[LiveMatch]:
    """MLS only (any data_source == "espn" league). Reuses
    espn_client.fetch_day_fixtures for today -- it already includes
    in-progress matches, just filtered here to the genuinely-live ones.
    """
    import datetime as dt

    matches = espn_client.fetch_day_fixtures(league.espn_league_slug, dt.date.today())
    results = []
    for match in matches:
        if match.state != "in" or match.home_score is None or match.away_score is None:
            continue
        results.append(
            LiveMatch(
                league_code=league.code,
                home_name=match.home_name,
                away_name=match.away_name,
                home_crest=None,
                away_crest=None,
                home_score=match.home_score,
                away_score=match.away_score,
                clock_label=match.clock_label or "Live",
            )
        )
    return results


def fetch_all_live_matches(session: Session, leagues: dict[str, League]) -> list[LiveMatch]:
    """Combines both sources for the global "LIVE NOW" banner."""
    results = fetch_live_matches_football_data_org(session)
    for league in leagues.values():
        if league.data_source == "espn":
            results.extend(fetch_live_matches_espn(league))
    return results
