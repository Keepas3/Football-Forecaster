"""Best-effort injury fetch from API-Football (api-sports.io), the bonus
source layered on top of the reliable config/injuries.yaml manual list.

Two things API-Football's /injuries endpoint doesn't give us directly, both
worked around here rather than accepted as-is:

1. No reliable position -- its player object is just {id, name, photo, type,
   reason}, no position field. Guessing "attack" for anything not explicitly
   "defender"/"goalkeeper" would silently mis-bucket every injured defender.
   Fixed by cross-referencing the player's real, current position from
   football-data.org's squad list (ingest/squad.py) instead.
2. No importance signal -- every injury used to get the same flat
   DEFAULT_API_IMPORTANCE_WEIGHT regardless of whether it's the team's top
   scorer or a bench player. Fixed by looking up the player's own historical
   stats (ingest/player_stats.py) and computing a real weight
   (ingest/player_importance.py) from their actual goal/assist/minutes
   output, preferring current-season xG/xA from Understat
   (ingest/understat_client.py) when available, and falling back to the flat
   default only when there's truly no data to compute from (new signing,
   unresolved name, fetch failure).

Either fix degrades to the old flat/best-guess behavior on any failure --
this stays a best-effort bonus source, never a hard dependency.
"""

from __future__ import annotations

import requests
from rapidfuzz import fuzz, process
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import api_football_client, player_importance, player_stats, squad, understat_client
from soccer_predictor.ingest.api_football_client import MissingApiKey
from soccer_predictor.ingest.squad import SquadPlayer
from soccer_predictor.storage.models import Team
from soccer_predictor.storage.repository import replace_injuries, update_api_football_team_id

INJURIES_CACHE_TTL_SECONDS = 24 * 3600  # squad injury status doesn't change within a day


def _get(path: str, params: dict) -> dict:
    return api_football_client.get(path, params, cache_ttl_seconds=INJURIES_CACHE_TTL_SECONDS)


def _guess_position(player_position: str) -> str:
    """Maps a free-text position string to our attack/defense bucket."""
    lowered = (player_position or "").lower()
    if lowered in {"defender", "goalkeeper"}:
        return "defense"
    return "attack"  # midfielders bucketed with attack: they still drive chance creation


def _resolve_position(player_name: str, api_position_field: str, squad_players: list[SquadPlayer]) -> str:
    """Prefers the player's real, current position from the football-data.org
    squad list (fuzzy-matched by name) over API-Football's own (likely
    absent) position field, which is only used as a last-resort fallback.
    """
    if squad_players:
        names = [p.name for p in squad_players]
        match = process.extractOne(player_name, names, scorer=fuzz.WRatio)
        if match is not None:
            _, score, index = match
            if score >= player_importance.PLAYER_NAME_FUZZY_THRESHOLD:
                return _guess_position(squad_players[index].position)
    return _guess_position(api_position_field)


def _search_api_football_team_id(team_api_name: str) -> int | None:
    team_search = _get("/teams", params={"name": team_api_name})
    results = team_search.get("response", [])
    if not results:
        return None
    return results[0]["team"]["id"]


def fetch_injuries_for_api_team(api_football_team_id: int, season: int) -> list[dict]:
    injuries = _get("/injuries", params={"team": api_football_team_id, "season": season})
    return injuries.get("response", [])


def sync_injuries_to_db(session: Session, league: League, season_year: int) -> tuple[int, int]:
    """Fetches + stores API-sourced injuries for every team in `league`.

    Returns (teams_synced, teams_skipped). Skips (rather than raising) any
    team whose API-Football name can't be resolved -- this is a best-effort
    bonus source, so one bad lookup shouldn't block the rest of the league.

    Resolving our team to API-Football's own team id used to cost a
    `/teams?name=` search call on *every* sync -- doubling the API calls
    made here for no reason, since a team's identity never changes. That id
    is now cached on Team.api_football_team_id the first time it's found
    (see storage.repository.update_api_football_team_id) and reused forever
    after, so a repeat sync only ever costs the one `/injuries` call/team.
    """
    from soccer_predictor.storage.repository import teams_for_league

    synced = 0
    skipped = 0
    for team_id, team_name in teams_for_league(session, league.code).items():
        team = session.get(Team, team_id)
        api_football_team_id = team.api_football_team_id if team is not None else None

        if api_football_team_id is None:
            try:
                api_football_team_id = _search_api_football_team_id(team_name)
            except requests.HTTPError:
                skipped += 1
                continue
            if api_football_team_id is None:
                skipped += 1
                continue
            update_api_football_team_id(session, team_id, api_football_team_id)

        try:
            raw_injuries = fetch_injuries_for_api_team(api_football_team_id, season_year)
        except requests.HTTPError:
            skipped += 1
            continue

        # football-data.org, a separate quota pool from API-Football --
        # cheap even called per-team, since one call already covers the
        # whole league's squads and is cached 24h (see squad.py).
        squad_players = squad.fetch_squad_for_team(session, league, team_id) or []

        team_stats: list[player_stats.HistoricalPlayerStats] = []
        team_totals = player_importance.TeamOutputTotals(goal_contribution_total=0, max_minutes=0)
        understat_players: list[understat_client.UnderstatPlayerStats] = []
        understat_totals = player_importance.UnderstatTeamTotals(xg_contribution_total=0)
        if raw_injuries:
            # Only fetched when this team actually has an injury to price --
            # most teams have none on a given sync, so this keeps the extra
            # API-Football call rare rather than doubling every sync's cost.
            # Deliberately uses the latest *available* season (2024, the
            # free plan's ceiling -- see player_stats.AVAILABLE_SEASONS),
            # not season_year (the live year): the free plan blocks
            # current-season /players data entirely, so this is the best
            # real proxy for "how good is this player right now" we can
            # get, not a bug to later "fix" into matching season_year (doing
            # so would silently zero out every result).
            team_stats = player_stats.fetch_team_player_stats(
                team_name, league.code, season=max(player_stats.AVAILABLE_SEASONS)
            )
            team_totals = player_importance.aggregate_team_output(team_stats)

            # Understat has no API-Football-style season restriction, so
            # this is the CURRENT season -- a fresher "how good are they
            # right now" signal than team_stats above. Best-effort:
            # understat_client already degrades to [] on any failure (site
            # format change, network error, unlisted league), so
            # resolve_api_importance_weight's own fallback chain (Understat
            # -> API-Football goals/assists -> flat default) holds either
            # way. Gated the same as team_stats above -- no point scraping a
            # team with nothing to price, and Understat has no published
            # rate limit to lean on the way API-Football's cache does.
            understat_players = understat_client.fetch_team_season(team_name, league.code, season_year)
            understat_totals = player_importance.aggregate_understat_team_output(understat_players)

        entries = []
        for row in raw_injuries:
            player_name = row["player"]["name"]
            position = _resolve_position(player_name, row["player"].get("position", ""), squad_players)
            importance_weight = player_importance.resolve_api_importance_weight(
                player_name, position, team_stats, team_totals, understat_players, understat_totals
            )
            entries.append(
                {
                    "player_name": player_name,
                    "position": position,
                    "importance_weight": importance_weight,
                    "note": row.get("player", {}).get("reason", ""),
                }
            )
        replace_injuries(session, team_id, source="api", injuries=entries)
        synced += 1
    return synced, skipped
