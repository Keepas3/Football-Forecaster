"""Historical per-player statistics (goals, assists, saves, tackles, cards,
rating) from API-Football -- a separate, season-queryable lookup, not
merged onto the current squad list (football-data.org's squad is already
the source of truth for "who's on the team now"; API-Football's free plan
can't see the current season at all, so mixing the two into one table just
produces a confusing mismatch between current names and old stats).

Real constraints, all surfaced to the caller rather than hidden:

1. The free plan only allows seasons 2022-2024 for the /players endpoint --
   not the current in-progress season. AVAILABLE_SEASONS is exactly that
   list; the dashboard's season picker is built from it directly, and
   bumping it is the only change needed if the plan is ever upgraded.
2. API-Football's /teams?name= search requires a near-exact substring of
   ITS OWN team name (verified empirically: "Man City" and "Manchester City
   FC" both return zero results against a team actually named "Manchester
   City") -- there's no fuzzy tolerance server-side. So team resolution
   here fetches the whole league's team list once per season (cached) and
   fuzzy-matches locally, the same approach ingest/team_mapper.py already
   uses for reconciling name differences between our other two data sources.
3. Because this is a standalone historical view, not merged with the
   current squad, player names/positions shown here are API-Football's own
   ("M. Akanji") -- no cross-API name matching needed at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests
from rapidfuzz import fuzz, process

from soccer_predictor.ingest import api_football_client

AVAILABLE_SEASONS = [2022, 2023, 2024]  # free plan's allowed range, oldest first

# API-Football's own numeric league ids (stable, verified: EPL=39 came back
# directly on a real /players response; LALIGA=140 confirmed via a live
# /leagues?name=La Liga lookup; BUNDESLIGA/SERIEA are well-established
# published ids, not independently verified against this account to
# conserve the free plan's 100/day quota -- if either is wrong, team
# resolution for that league just degrades to "no stats available", not
# an error, so it's a safe assumption to ship on.
API_FOOTBALL_LEAGUE_ID = {
    "EPL": 39,
    "LALIGA": 140,
    "BUNDESLIGA": 78,
    "SERIEA": 135,
    "LIGUE1": 61,
    # Well-established published ids for these three, same "not
    # independently verified against this account" caveat as
    # SERIEA/BUNDESLIGA above.
    "UCL": 2,
    "EURO": 4,
    "WC": 1,
}

TEAM_LIST_CACHE_TTL_SECONDS = 30 * 24 * 3600  # a past season's team list never changes
PLAYER_STATS_CACHE_TTL_SECONDS = 30 * 24 * 3600  # a past season's stats never change

TEAM_NAME_FUZZY_THRESHOLD = 75  # "Man City"/"Nott'm Forest" vs full names score 80-86 (verified)

# API-Football's own competition-name strings, used to pick the right row
# out of a player's per-competition stats breakdown (which also includes
# cup/continental competitions we don't want mixed in).
PRIMARY_COMPETITION_NAME = {
    "EPL": "Premier League",
    "LALIGA": "La Liga",
    "BUNDESLIGA": "Bundesliga",
    "SERIEA": "Serie A",
    "LIGUE1": "Ligue 1",
    # Unverified guesses (same caveat as above) -- more consequential to get
    # wrong here than for a domestic league, since a club's per-player stats
    # breakdown always includes rows for every competition they played that
    # season (domestic league, cup, continental). If this name string
    # doesn't match, _select_primary_row's fallback picks the row with the
    # most appearances instead, which for a club would usually be its
    # domestic league -- i.e. this could silently show domestic-league
    # stats mislabeled as Champions League ones rather than "no stats".
    "UCL": "UEFA Champions League",
    "EURO": "UEFA Euro Championship",
    "WC": "World Cup",
}


@dataclass
class HistoricalPlayerStats:
    name: str
    position: str | None
    nationality: str | None
    appearances: int | None
    minutes: int | None
    goals: int | None
    assists: int | None
    saves: int | None  # goalkeepers only; None for outfield players
    tackles: int | None
    yellow_cards: int | None
    red_cards: int | None
    rating: float | None


def _fetch_league_teams(league_code: str, season: int) -> list[dict]:
    api_league_id = API_FOOTBALL_LEAGUE_ID.get(league_code)
    if api_league_id is None:
        return []
    try:
        data = api_football_client.get(
            "/teams",
            {"league": api_league_id, "season": season},
            cache_ttl_seconds=TEAM_LIST_CACHE_TTL_SECONDS,
        )
    except (api_football_client.MissingApiKey, requests.RequestException):
        return []
    return data.get("response", [])


def _find_team_id(our_team_name: str, league_code: str, season: int) -> int | None:
    teams = _fetch_league_teams(league_code, season)
    if not teams:
        return None
    names = [t["team"]["name"] for t in teams]
    match = process.extractOne(our_team_name, names, scorer=fuzz.WRatio)
    if match is None:
        return None
    _, score, index = match
    if score < TEAM_NAME_FUZZY_THRESHOLD:
        return None
    return teams[index]["team"]["id"]


def _select_primary_row(statistics: list[dict], league_code: str) -> dict | None:
    competition_name = PRIMARY_COMPETITION_NAME.get(league_code)
    for row in statistics:
        if row.get("league", {}).get("name") == competition_name:
            return row
    # Fall back to whichever row has the most appearances, rather than
    # showing nothing, if the primary-league name didn't match anything.
    with_apps = [r for r in statistics if r.get("games", {}).get("appearences")]
    if not with_apps:
        return statistics[0] if statistics else None
    return max(with_apps, key=lambda r: r["games"]["appearences"])


def _parse_player(entry: dict, league_code: str) -> HistoricalPlayerStats | None:
    row = _select_primary_row(entry.get("statistics", []), league_code)
    if row is None:
        return None
    games = row.get("games", {})
    goals = row.get("goals", {})
    tackles = row.get("tackles", {})
    cards = row.get("cards", {})
    rating = games.get("rating")
    return HistoricalPlayerStats(
        name=entry["player"]["name"],
        position=games.get("position"),
        nationality=entry["player"].get("nationality"),
        appearances=games.get("appearences"),
        minutes=games.get("minutes"),
        goals=goals.get("total"),
        assists=goals.get("assists"),
        saves=goals.get("saves"),
        tackles=tackles.get("total"),
        yellow_cards=cards.get("yellow"),
        red_cards=cards.get("red"),
        rating=float(rating) if rating else None,
    )


def fetch_team_player_stats(
    our_team_name: str, league_code: str, season: int
) -> list[HistoricalPlayerStats]:
    """Every player who has a stats row for this team in `season` (one of
    AVAILABLE_SEASONS). Empty list on any failure (no key, network error,
    team not matched) -- degrades to "no stats" rather than raising.
    """
    team_id = _find_team_id(our_team_name, league_code, season)
    if team_id is None:
        return []

    players: list[HistoricalPlayerStats] = []
    page = 1
    while True:
        try:
            data = api_football_client.get(
                "/players",
                {"team": team_id, "season": season, "page": page},
                cache_ttl_seconds=PLAYER_STATS_CACHE_TTL_SECONDS,
            )
        except (api_football_client.MissingApiKey, requests.RequestException):
            return players

        for entry in data.get("response", []):
            player = _parse_player(entry, league_code)
            if player is not None:
                players.append(player)

        paging = data.get("paging", {})
        if page >= paging.get("total", 1):
            break
        page += 1

    return players
