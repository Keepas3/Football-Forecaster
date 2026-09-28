"""Fetches upcoming fixtures from football-data.org and syncs them to SQLite."""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import api_client
from soccer_predictor.ingest.team_mapper import UnresolvedTeamName, resolve
from soccer_predictor.storage.repository import update_team_crest, upsert_fixture, upsert_match

FIXTURES_CACHE_TTL_SECONDS = 6 * 3600
# A past season's results never change once the season is over, but the
# *current* season's are refetched often (see refresh_live_data.py) --
# short enough to pick up yesterday's results without wasting the day's
# football-data.org quota on every dashboard page load.
RESULTS_CACHE_TTL_SECONDS = 6 * 3600


def fetch_upcoming_fixtures(league: League) -> list[dict]:
    """Raw API rows for this league's next scheduled matches."""
    data = api_client.get(
        f"/competitions/{league.api_competition_id}/matches",
        params={"status": "SCHEDULED"},
        cache_ttl_seconds=FIXTURES_CACHE_TTL_SECONDS,
    )
    return data.get("matches", [])


def sync_fixtures_to_db(session: Session, league: League) -> tuple[int, int]:
    """Resolves team names and upserts fixtures. Returns (synced, skipped)."""
    synced = 0
    skipped = 0
    for match in fetch_upcoming_fixtures(league):
        home_name = match["homeTeam"]["name"]
        away_name = match["awayTeam"]["name"]
        try:
            home_team_id = resolve(session, home_name, source="api", league_code=league.code)
            away_team_id = resolve(session, away_name, source="api", league_code=league.code)
        except UnresolvedTeamName:
            skipped += 1
            continue

        update_team_crest(session, home_team_id, match["homeTeam"].get("crest"))
        update_team_crest(session, away_team_id, match["awayTeam"].get("crest"))

        match_datetime_utc = dt.datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))
        upsert_fixture(
            session,
            league_code=league.code,
            date=match_datetime_utc.date(),
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            status=match["status"],
            kickoff_utc=match_datetime_utc.replace(tzinfo=None),
        )
        synced += 1
    return synced, skipped


def fetch_finished_matches(league: League, season_year: int) -> list[dict]:
    """Raw API rows for one season of this league's completed matches.

    Only used for leagues with no football-data.co.uk CSV (see
    League.csv_code) -- those are the only ones missing a `matches` history
    to build standings/train from, since domestic leagues get theirs from
    the CSV pipeline instead (scripts/fetch_historical_data.py).
    """
    data = api_client.get(
        f"/competitions/{league.api_competition_id}/matches",
        params={"status": "FINISHED", "season": season_year},
        cache_ttl_seconds=RESULTS_CACHE_TTL_SECONDS,
    )
    return data.get("matches", [])


def sync_results_to_db(session: Session, league: League, season: str) -> tuple[int, int]:
    """Resolves team names and upserts finished matches for one season.
    Returns (synced, skipped). Mirrors sync_fixtures_to_db, but writes to
    the `matches` table (standings/training history) instead of `fixtures`
    (the upcoming-games list).
    """
    synced = 0
    skipped = 0
    for match in fetch_finished_matches(league, league.api_season_year(season)):
        score = match.get("score", {}).get("fullTime", {})
        if score.get("home") is None or score.get("away") is None:
            continue
        home_name = match["homeTeam"]["name"]
        away_name = match["awayTeam"]["name"]
        try:
            home_team_id = resolve(session, home_name, source="api", league_code=league.code)
            away_team_id = resolve(session, away_name, source="api", league_code=league.code)
        except UnresolvedTeamName:
            skipped += 1
            continue

        update_team_crest(session, home_team_id, match["homeTeam"].get("crest"))
        update_team_crest(session, away_team_id, match["awayTeam"].get("crest"))

        match_date = dt.datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")).date()
        upsert_match(
            session,
            league_code=league.code,
            season=season,
            date=match_date,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=score["home"],
            away_goals=score["away"],
            source="football-data.org",
        )
        synced += 1
    return synced, skipped
