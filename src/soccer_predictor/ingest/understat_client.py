"""Best-effort scrape of Understat (understat.com) for Expected Goals (xG)/
Expected Assists (xA) -- a bonus signal for ingest/player_importance.py, on
top of API-Football's own goals/assists (ingest/player_stats.py).

No official API exists, so this hits Understat's own internal AJAX endpoint
(`main/getPlayersStats/`, found by inspecting its league page's own
league.min.js -- not documented, so it can change without notice; every
failure mode here degrades to an empty list, never raises, same contract as
the rest of this app's external integrations).

This intentionally does NOT use the classic "regex out a `var playersData =
JSON.parse('...')` blob embedded in the page HTML" technique documented
around the web for Understat scraping: verified live (2026-09) that its
league pages no longer embed that data server-side at all -- the tables are
now populated client-side by a POST to the endpoint used here, which returns
clean JSON directly. That also sidesteps the classic gotcha with the
embedded-JSON approach (naive unicode_escape decoding mangling accented
player names) since this is real JSON via `response.json()`, not a hand
-decoded string.

Understat's real edge over API-Football's free tier: no season restriction.
API-Football's /players and /injuries endpoints refuse anything outside
2022-2024 on the free plan (see player_stats.py's AVAILABLE_SEASONS); a live
check confirmed Understat happily serves the in-progress current season, a
genuinely fresher "how good is this player right now" signal, not just a
less-noisy one.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from rapidfuzz import fuzz, process

from soccer_predictor.config import DATA_DIR

BASE_URL = "https://understat.com"
CACHE_DIR = DATA_DIR / "cache" / "understat"

# Shorter than player_stats.py's 30-day cache -- the whole point of this
# source is that it reflects the CURRENT season, which changes match to
# match, not a frozen past season.
CACHE_TTL_SECONDS = 24 * 3600

TEAM_NAME_FUZZY_THRESHOLD = 75  # same tolerance as player_stats.py's own team matching

# Understat's own league identifiers, exactly as its league-page <select
# name="league"> option values (verified live 2026-09 against
# understat.com/league/<slug>/<season> -- not a documented/stable API, so
# this is what actually worked, not a guess). No UCL/EURO/WC coverage:
# Understat only tracks the "Big 5" domestic leagues (+ RFPL); those
# competitions just get no Understat data, same graceful-no-data contract
# as everywhere else in this app.
UNDERSTAT_LEAGUE_SLUG = {
    "EPL": "EPL",
    "LALIGA": "La liga",
    "BUNDESLIGA": "Bundesliga",
    "SERIEA": "Serie A",
    "LIGUE1": "Ligue 1",
}


@dataclass
class UnderstatPlayerStats:
    name: str
    team_title: str
    position: str | None
    minutes: int
    goals: int
    assists: int
    xg: float
    xa: float


def _cache_path(league_slug: str, season: int) -> Path:
    key = hashlib.sha256(f"{league_slug}:{season}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _fetch_league_players(league_slug: str, season: int) -> list[dict]:
    """Raw player rows for every team in `league_slug`/`season` -- one POST
    covers the whole league, so team resolution below is a local fuzzy
    match, the same fetch-once-match-locally approach player_stats.py's
    `_find_team_id` already uses for API-Football.
    """
    cache_file = _cache_path(league_slug, season)
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    response = requests.post(
        f"{BASE_URL}/main/getPlayersStats/",
        headers={
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{BASE_URL}/league/{league_slug}/{season}",
        },
        data={"league": league_slug, "season": str(season)},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("success"):
        return []
    players = data.get("players", [])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(players), encoding="utf-8")
    return players


def _parse_player(row: dict) -> UnderstatPlayerStats:
    return UnderstatPlayerStats(
        name=row["player_name"],
        team_title=row["team_title"],
        position=row.get("position") or None,
        minutes=int(row.get("time") or 0),
        goals=int(row.get("goals") or 0),
        assists=int(row.get("assists") or 0),
        xg=float(row.get("xG") or 0),
        xa=float(row.get("xA") or 0),
    )


def fetch_team_season(our_team_name: str, league_code: str, season: int) -> list[UnderstatPlayerStats]:
    """Every Understat-tracked player for `our_team_name` in `season` (a
    start-year int, e.g. 2026 for the 2026/27 season -- Understat's own
    convention, matching League.api_season_year()'s). Empty list on any
    failure (unmapped league, network error, site format change, no
    matching team) -- degrades to "no stats" rather than raising, same
    contract as ingest/player_stats.py.
    """
    league_slug = UNDERSTAT_LEAGUE_SLUG.get(league_code)
    if league_slug is None:
        return []

    try:
        raw_players = _fetch_league_players(league_slug, season)
        if not raw_players:
            return []

        team_titles = list({row["team_title"] for row in raw_players})
        match = process.extractOne(our_team_name, team_titles, scorer=fuzz.WRatio)
        if match is None:
            return []
        matched_title, score, _ = match
        if score < TEAM_NAME_FUZZY_THRESHOLD:
            return []

        return [_parse_player(row) for row in raw_players if row["team_title"] == matched_title]
    except (requests.RequestException, ValueError, KeyError):
        return []
