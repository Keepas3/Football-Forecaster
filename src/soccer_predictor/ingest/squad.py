"""Fetches full team squads (every player + basic bio) from football-data.org.

One call covers every team in a competition at once, so this stays cheap
even though the dashboard asks for it one team at a time -- api_client.get
already caches the whole response to disk, so repeat calls for different
teams in the same league don't cost extra requests.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import api_client
from soccer_predictor.storage.repository import find_team_by_alias

SQUAD_CACHE_TTL_SECONDS = 24 * 3600  # squads change slowly outside transfer windows


@dataclass
class SquadPlayer:
    name: str
    position: str
    nationality: str
    date_of_birth: str | None


@dataclass
class TeamInfo:
    """Club background from the same football-data.org response the squad
    comes from -- founded year, home ground, colors, current head coach,
    official site. Any field the API left null stays None; the dashboard
    only shows what's actually populated rather than printing "None"."""

    founded: int | None
    venue: str | None
    club_colors: str | None
    address: str | None
    website: str | None
    coach_name: str | None
    coach_nationality: str | None


def _fetch_raw_teams(league: League) -> list[dict]:
    try:
        data = api_client.get(
            f"/competitions/{league.api_competition_id}/teams",
            cache_ttl_seconds=SQUAD_CACHE_TTL_SECONDS,
        )
    except (api_client.MissingApiKey, requests.RequestException):
        return []
    return data.get("teams", [])


def _find_raw_team_entry(session: Session, league: League, team_id: int) -> dict | None:
    for entry in _fetch_raw_teams(league):
        team = find_team_by_alias(session, entry.get("name", ""), source="api", league_code=league.code)
        if team is not None and team.id == team_id:
            return entry
    return None


def fetch_squad_for_team(session: Session, league: League, team_id: int) -> list[SquadPlayer] | None:
    """Returns this team's squad, or None if unavailable -- no API key, a
    network error, or this team's API-side name isn't a known alias yet
    (e.g. it's never appeared in a live fixture sync). Degrades to None
    rather than raising, same as ingest/league_meta.py's emblem fetch.
    """
    entry = _find_raw_team_entry(session, league, team_id)
    if entry is None:
        return None
    return [
        SquadPlayer(
            name=p["name"],
            position=p.get("position") or "Unknown",
            nationality=p.get("nationality") or "",
            date_of_birth=p.get("dateOfBirth"),
        )
        for p in entry.get("squad", [])
    ]


def fetch_team_info(session: Session, league: League, team_id: int) -> TeamInfo | None:
    """Returns club background for this team, or None under the same
    unavailable conditions as fetch_squad_for_team (they share one cached
    API response, so this costs nothing extra once the squad's been loaded).
    """
    entry = _find_raw_team_entry(session, league, team_id)
    if entry is None:
        return None
    coach = entry.get("coach") or {}
    return TeamInfo(
        founded=entry.get("founded"),
        venue=entry.get("venue") or None,
        club_colors=entry.get("clubColors") or None,
        address=entry.get("address") or None,
        website=entry.get("website") or None,
        coach_name=coach.get("name") or None,
        coach_nationality=coach.get("nationality") or None,
    )
