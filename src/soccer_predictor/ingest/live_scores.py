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

import datetime as dt
from dataclasses import dataclass

import requests
from sqlalchemy.orm import Session

from soccer_predictor.config import League, load_leagues
from soccer_predictor.ingest import api_client, espn_client
from soccer_predictor.storage.repository import find_team_by_alias

LIVE_SCORE_CACHE_TTL_SECONDS = 30
# Finished results change far less than running scores; this keeps the
# extra football-data.org call (see fetch_recent_finished_football_data_org)
# to a handful per hour.
FINISHED_CACHE_TTL_SECONDS = 120
# No feed says when a match actually ended, so it is estimated as a generous
# fixed time after kickoff (90 min + halftime + stoppage; the same estimate
# the Leagues page's clock-based LIVE marker uses). A finished match then
# stays on the banner for FINISHED_LINGER, and drops off after that.
ESTIMATED_MATCH_LENGTH = dt.timedelta(hours=2, minutes=15)
FINISHED_LINGER = dt.timedelta(hours=3, minutes=30)


@dataclass
class LiveMatch:
    league_code: str
    home_name: str  # resolved to our canonical name when possible
    away_name: str
    home_crest: str | None
    away_crest: str | None
    # None when no score is known (the clock-based fallback banner).
    home_score: int | None
    away_score: int | None
    clock_label: str  # "63'", "HT", "45+2'", "Live" if unknown, or "FT" once finished
    finished: bool = False
    kickoff_utc: dt.datetime | None = None  # naive UTC


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


def _recently_finished(kickoff_utc: dt.datetime, now: dt.datetime) -> bool:
    """Ended (by the length estimate) no more than FINISHED_LINGER ago."""
    ended = kickoff_utc + ESTIMATED_MATCH_LENGTH
    return ended <= now < ended + FINISHED_LINGER


def fetch_recent_finished_football_data_org(
    session: Session, now: dt.datetime | None = None
) -> list[LiveMatch]:
    """Matches from the football-data.org competitions this app tracks that
    finished within the last FINISHED_LINGER -- kept on the banner as final
    scores. One global call over yesterday-today (a recent match can
    straddle UTC midnight), filtered to the linger window here. Empty list
    on any failure.
    """
    now = now if now is not None else dt.datetime.now(dt.UTC).replace(tzinfo=None)
    today = now.date()
    try:
        data = api_client.get(
            "/matches",
            params={
                "status": "FINISHED",
                "dateFrom": (today - dt.timedelta(days=1)).isoformat(),
                "dateTo": (today + dt.timedelta(days=1)).isoformat(),
            },
            cache_ttl_seconds=FINISHED_CACHE_TTL_SECONDS,
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
            if league_code is None or match.get("status") != "FINISHED":
                continue
            kickoff = dt.datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")).replace(tzinfo=None)
            full_time = match.get("score", {}).get("fullTime", {})
            if not _recently_finished(kickoff, now) or full_time.get("home") is None or full_time.get("away") is None:
                continue
            home_name, home_crest = _resolve_display_name(session, match["homeTeam"]["name"], league_code)
            away_name, away_crest = _resolve_display_name(session, match["awayTeam"]["name"], league_code)
            results.append(
                LiveMatch(
                    league_code=league_code,
                    home_name=home_name,
                    away_name=away_name,
                    home_crest=home_crest,
                    away_crest=away_crest,
                    home_score=full_time["home"],
                    away_score=full_time["away"],
                    clock_label="FT",
                    finished=True,
                    kickoff_utc=kickoff,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return results


def fetch_live_matches_espn(league: League, now: dt.datetime | None = None) -> list[LiveMatch]:
    """Any data_source == "espn" league: matches in progress right now, plus
    ones that finished within the last FINISHED_LINGER (shown as "FT").
    Reads yesterday's and today's (UTC) scoreboards, since a recent match
    can straddle midnight.
    """
    now = now if now is not None else dt.datetime.now(dt.UTC).replace(tzinfo=None)
    # The scoreboard's normal 6h cache would freeze the banner on whatever
    # was live when it was first fetched (and miss every later kickoff).
    matches = []
    for day in (now.date() - dt.timedelta(days=1), now.date()):
        matches.extend(
            espn_client.fetch_day_matches(league.espn_league_slug, day, cache_ttl_seconds=LIVE_SCORE_CACHE_TTL_SECONDS)
        )

    results = []
    for match in matches:
        if match.home_score is None or match.away_score is None:
            continue
        is_live = match.state == "in"
        is_recent_final = (
            match.completed
            and match.state == "post"
            and match.kickoff_utc is not None
            and _recently_finished(match.kickoff_utc, now)
        )
        if not (is_live or is_recent_final):
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
                clock_label=(match.clock_label or "Live") if is_live else "FT",
                finished=not is_live,
                kickoff_utc=match.kickoff_utc,
            )
        )
    return results


def fetch_all_live_matches(session: Session, leagues: dict[str, League]) -> list[LiveMatch]:
    """Everything the "LIVE NOW" banner shows, from both sources: matches in
    progress first, then recently finished ones (most recent first). A match
    both feeds briefly report (cached live and finished copies around the
    final whistle) appears once, as finished.
    """
    results = fetch_live_matches_football_data_org(session)
    results.extend(fetch_recent_finished_football_data_org(session))
    for league in leagues.values():
        if league.data_source == "espn":
            results.extend(fetch_live_matches_espn(league))

    finished_keys = {(m.league_code, m.home_name, m.away_name) for m in results if m.finished}
    results = [m for m in results if m.finished or (m.league_code, m.home_name, m.away_name) not in finished_keys]
    live = [m for m in results if not m.finished]
    finished = sorted(
        (m for m in results if m.finished), key=lambda m: m.kickoff_utc or dt.datetime.min, reverse=True
    )
    return live + finished
