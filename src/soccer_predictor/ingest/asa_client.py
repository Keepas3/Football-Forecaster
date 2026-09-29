"""American Soccer Analysis (app.americansocceranalysis.com) -- a real, free,
public, keyless API dedicated to MLS/NWSL/USL, used here for MLS only (this
app has no other American Soccer Analysis-covered league). Confirmed live
(2026-09) this genuinely has current-season data, unlike API-Football's free
tier -- and unlike API-Football, no account/key at all, so nothing here can
get suspended the way that did.

MLS-only, deliberately: `fetch_team_season` takes no `league_code` param,
unlike understat_client.py's five-league `fetch_team_season`, since this
client is never called for anything but MLS (see
ingest/player_importance.py::resolve_team_historical_stats, which sends MLS
here and everywhere else through Understat/API-Football).

Two separate endpoints, joined locally on `player_id` since the stats
endpoint doesn't carry player names:
- `/mls/players` -- bios (name/position/nationality), fetched once as a
  whole-league lookup table, same fetch-once-match-locally shape as
  understat_client.py's league-wide fetch.
- `/mls/players/xgoals?season_name=&team_id=` -- real per-player season
  stats, already scoped to exactly that team's roster for that season (no
  Understat-style mid-season-transfer combined-title issue here, since
  `team_id` filtering happens server-side).

No `games`-played/appearances field exists in this response at all --
left out of AsaPlayerStats rather than faked from something else.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from rapidfuzz import fuzz, process

from soccer_predictor.config import DATA_DIR

BASE_URL = "https://app.americansocceranalysis.com/api/v1"
CACHE_DIR = DATA_DIR / "cache" / "asa"

# Same reasoning as understat_client.py's TTL: reflects the in-progress
# season, so a day-old cache is as stale as this app tolerates anywhere else.
CACHE_TTL_SECONDS = 24 * 3600

TEAM_NAME_FUZZY_THRESHOLD = 75  # same tolerance as the other clients' team matching

# ASA's real floor, confirmed live (2026-09): season 2013 returns real
# players, 2012 returns none.
ASA_AVAILABLE_SEASONS = list(range(2013, dt.date.today().year + 1))

# "LAFC" (this app's canonical name, from ESPN) scores only 60 against ASA's
# "Los Angeles FC" with fuzz.WRatio -- well under TEAM_NAME_FUZZY_THRESHOLD,
# and no fuzzy scorer tried got it above ~67 (confirmed live) since an
# acronym shares almost no substring with the spelled-out name it stands
# for. Every other one of this app's 29 MLS teams matched at 82+ without
# help, verified live -- this is the one real exception, not a sign the
# whole approach needs a wholesale alias table.
TEAM_NAME_OVERRIDES = {"LAFC": "Los Angeles FC"}


@dataclass
class AsaPlayerStats:
    name: str
    position: str | None
    minutes: int
    goals: int
    assists: int
    xg: float
    xa: float
    shots: int
    key_passes: int
    points_added: float


def _cache_path(name: str) -> Path:
    key = hashlib.sha256(name.encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _get_cached_json(cache_key: str, url: str, params: dict) -> list[dict]:
    cache_file = _cache_path(cache_key)
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        return []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data


def _fetch_all_teams() -> list[dict]:
    return _get_cached_json("teams", f"{BASE_URL}/mls/teams", {})


def _fetch_all_players() -> list[dict]:
    return _get_cached_json("players", f"{BASE_URL}/mls/players", {})


def _resolve_team_id(our_team_name: str, teams: list[dict]) -> str | None:
    query = TEAM_NAME_OVERRIDES.get(our_team_name, our_team_name)
    team_names = [t["team_name"] for t in teams]
    match = process.extractOne(query, team_names, scorer=fuzz.WRatio)
    if match is None:
        return None
    matched_name, score, index = match
    if score < TEAM_NAME_FUZZY_THRESHOLD:
        return None
    return teams[index]["team_id"]


def fetch_team_season(our_team_name: str, season: int) -> list[AsaPlayerStats]:
    """Every ASA-tracked MLS player for `our_team_name` in `season` (a
    start-year int, same convention as understat_client.py). Empty list on
    any failure (network error, no team match, unmapped response shape) --
    degrades to "no stats" rather than raising, same contract as every
    other ingest client in this app.
    """
    try:
        teams = _fetch_all_teams()
        if not teams:
            return []
        team_id = _resolve_team_id(our_team_name, teams)
        if team_id is None:
            return []

        stat_rows = _get_cached_json(
            f"xgoals:{team_id}:{season}",
            f"{BASE_URL}/mls/players/xgoals",
            {"season_name": season, "team_id": team_id, "minimum_minutes": 1},
        )
        if not stat_rows:
            return []

        players_by_id = {p["player_id"]: p for p in _fetch_all_players()}

        results = []
        for row in stat_rows:
            bio = players_by_id.get(row.get("player_id"))
            if bio is None:
                continue
            results.append(_parse_stat_row(row, bio))
        return results
    except (requests.RequestException, ValueError, KeyError):
        return []


def _parse_stat_row(row: dict, bio: dict) -> AsaPlayerStats:
    return AsaPlayerStats(
        name=bio["player_name"],
        position=row.get("general_position") or bio.get("primary_general_position"),
        minutes=int(row.get("minutes_played") or 0),
        goals=int(row.get("goals") or 0),
        assists=int(row.get("primary_assists") or 0),
        xg=float(row.get("xgoals") or 0),
        xa=float(row.get("xassists") or 0),
        shots=int(row.get("shots") or 0),
        key_passes=int(row.get("key_passes") or 0),
        points_added=float(row.get("points_added") or 0),
    )


def fetch_all_teams_totals(season: int) -> dict[str, list[AsaPlayerStats]]:
    """Every ASA-tracked MLS team's players for `season`, keyed by ASA's own
    `team_name` -- lets a caller build a league-wide baseline (see
    ingest/player_importance.py::resolve_current_attack_strength). Unlike
    Understat's league-wide endpoint, ASA's `/xgoals` stats are scoped
    server-side per `team_id` -- there's no single whole-league call, so
    this makes one request per MLS team (~29). Each is disk-cached 24h same
    as fetch_team_season, so this is only ever slow on a cold cache; a
    keyless API with no daily quota, unlike the old API-Football, so the
    extra calls cost nothing but time. Empty dict on any failure.
    """
    try:
        teams = _fetch_all_teams()
        if not teams:
            return {}
        players_by_id = {p["player_id"]: p for p in _fetch_all_players()}

        result: dict[str, list[AsaPlayerStats]] = {}
        for team in teams:
            team_id = team["team_id"]
            stat_rows = _get_cached_json(
                f"xgoals:{team_id}:{season}",
                f"{BASE_URL}/mls/players/xgoals",
                {"season_name": season, "team_id": team_id, "minimum_minutes": 1},
            )
            players = []
            for row in stat_rows:
                bio = players_by_id.get(row.get("player_id"))
                if bio is None:
                    continue
                players.append(_parse_stat_row(row, bio))
            result[team["team_name"]] = players
        return result
    except (requests.RequestException, ValueError, KeyError):
        return {}
