"""Thin, rate-limited, caching HTTP client for ESPN's public soccer API
(site.api.espn.com) -- used for leagues football-data.org doesn't cover at
all (e.g. MLS; see League.data_source).

Undocumented and unofficial: no published rate limit or stability
guarantee, and no API key exists for it (there is nothing to authenticate).
Every parsing function here degrades to an empty list/None on any failure
(bad response shape, network error) rather than raising -- same
best-effort contract as ingest/understat_client.py, the closest existing
analog (also keyless, also an undocumented endpoint).

Endpoints used, all verified live against real 2026 MLS data:
- GET /teams -- the whole league's team list (id + display name + crest).
- GET /teams/{id}/schedule?season=YYYY -- one team's results for a season.
  Verified this only ever returns COMPLETED matches (a live check across a
  full team-season came back 27/27 completed, zero upcoming, with or
  without the season= param) -- never use it for upcoming fixtures.
- GET /scoreboard?dates=YYYYMMDD -- one day's games league-wide, including
  upcoming ones (verified live: several real near-future MLS matchdays
  returned real fixtures). Only a single date per call is supported -- a
  YYYYMMDD-YYYYMMDD range returned HTTP 400 in testing.
- GET /teams/{id}?enable=roster,injuries,stats,standings -- full roster with
  real positions (G/D/M/F) and a per-player `injuries` array (present but
  empty on every player checked live -- treat as thin/best-effort, not a
  guaranteed source of real injury data), plus `groups.id` -- a conference
  id (verified live: MLS's Eastern/Western conferences are ids "1"/"2").
  Note this endpoint's own top-level `/standings` (no team id) returned an
  empty `{}` in testing -- conference data is only reliably available
  per-team, not as one league-wide standings call.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import requests

from soccer_predictor.config import DATA_DIR

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
# The standings endpoint lives under a different path (/apis/v2, not
# /apis/site/v2) -- see fetch_group_membership.
STANDINGS_BASE_URL = "https://site.api.espn.com/apis/v2/sports/soccer"
CACHE_DIR = DATA_DIR / "cache" / "espn"
MAX_REQUESTS_PER_MINUTE = 30  # self-imposed courtesy limit -- no published cap exists

# A team's schedule/roster changes slowly; a scoreboard day is finalized
# once played. 6h mirrors ingest/fixtures.py's own TTLs for the same reason
# (short enough to pick up new results without hammering an unofficial
# endpoint on every dashboard load).
SCHEDULE_CACHE_TTL_SECONDS = 6 * 3600
TEAM_LIST_CACHE_TTL_SECONDS = 24 * 3600
ROSTER_CACHE_TTL_SECONDS = 24 * 3600

# ESPN's position.abbreviation -> this app's attack/defense bucket, same
# "midfielders count as attack" rationale already documented in
# ingest/injuries.py::_guess_position.
_POSITION_BUCKET = {"G": "defense", "D": "defense", "M": "attack", "F": "attack"}
# -> the same full-word labels football-data.org's squad data already uses
# (dashboard/views/team_detail.py's _POSITION_ORDER), so ESPN-sourced
# squads sort correctly with zero MLS-specific dashboard code.
_POSITION_LABEL = {"G": "Goalkeeper", "D": "Defence", "M": "Midfield", "F": "Offence"}

# team.groups.id (from /teams/{id}?enable=standings) -> MLS's own conference
# name -- confirmed live against real teams (Columbus Crew, an Eastern
# Conference club, has groups.id "1"; LA Galaxy, Western, has "2"), and
# cross-checked against the human-readable names ESPN itself uses on
# scoreboard entries ("Eastern Conference"/"Western Conference"). Unknown
# ids (e.g. a league with no real conference split) map to None.
_CONFERENCE_NAME = {"1": "Eastern Conference", "2": "Western Conference"}


class RateLimiter:
    """Blocks just long enough to keep requests under N per rolling minute."""

    def __init__(self, max_per_minute: int = MAX_REQUESTS_PER_MINUTE):
        self.max_per_minute = max_per_minute
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] > 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_per_minute:
            sleep_for = 60 - (now - self._timestamps[0]) + 0.1
            time.sleep(max(sleep_for, 0))
        self._timestamps.append(time.monotonic())


_rate_limiter = RateLimiter()


def _cache_path(path: str, params: dict, base_url: str = BASE_URL) -> Path:
    # The default base is left out of the key so cache files written before
    # base_url existed keep resolving.
    base_part = "" if base_url == BASE_URL else base_url
    key = hashlib.sha256(f"{base_part}{path}?{sorted(params.items())}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def get(
    path: str,
    params: dict | None = None,
    cache_ttl_seconds: int = 6 * 3600,
    base_url: str = BASE_URL,
) -> dict:
    """GETs `{BASE_URL}{path}`, serving from an on-disk cache within TTL.
    No API key exists for this endpoint -- the only failure mode is a
    network/HTTP error, left to the caller (mirrors understat_client.get).
    """
    params = params or {}
    cache_file = _cache_path(path, params, base_url)
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < cache_ttl_seconds:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    _rate_limiter.wait()
    response = requests.get(f"{base_url}{path}", params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data


@dataclass
class EspnTeam:
    espn_id: str
    display_name: str
    logo_url: str | None


@dataclass
class EspnMatch:
    espn_home_id: str
    espn_away_id: str
    home_name: str
    away_name: str
    date: dt.date
    kickoff_utc: dt.datetime | None
    completed: bool
    home_score: int | None
    away_score: int | None
    # ESPN's own status.type.state ("pre"/"in"/"post") -- "in" means
    # genuinely live right now, distinct from completed=False, which is
    # also true for a match that simply hasn't started yet. See
    # ingest/live_scores.py, the only current consumer of these two fields.
    state: str = "pre"
    clock_label: str | None = None  # e.g. "63'", "HT", "45+2'" -- only meaningful when state == "in"
    # Group-competition context (e.g. the Nations League), where ESPN
    # provides it: `group_name` is the scoreboard's own per-match group
    # label (not present for every edition -- see fetch_group_membership for
    # the reliable source), `stage` is the schedule endpoint's season-type
    # name ("Group Stage", "Semifinals", or a division like "League A").
    group_name: str | None = None
    stage: str | None = None


@dataclass
class EspnRosterPlayer:
    name: str
    position: str  # full-word label, see _POSITION_LABEL
    position_bucket: str  # "attack" | "defense", see _POSITION_BUCKET


@dataclass
class EspnInjuryEntry:
    player_name: str
    position_bucket: str
    note: str


def fetch_teams(league_slug: str) -> list[EspnTeam]:
    """Every team in this league. Empty list on any failure."""
    try:
        data = get(
            "/" + league_slug + "/teams",
            params={"limit": 50},
            cache_ttl_seconds=TEAM_LIST_CACHE_TTL_SECONDS,
        )
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return []

    try:
        raw_teams = data["sports"][0]["leagues"][0]["teams"]
    except (KeyError, IndexError):
        return []

    teams = []
    for entry in raw_teams:
        team = entry.get("team", {})
        logos = team.get("logos") or []
        logo_url = logos[0]["href"] if logos else None
        teams.append(EspnTeam(espn_id=str(team["id"]), display_name=team["displayName"], logo_url=logo_url))
    return teams


def _parse_score(competitor: dict) -> int | None:
    """`score` is a plain string on the scoreboard endpoint (e.g. "2") but a
    nested {"value": 2.0, ...} object on the schedule endpoint -- handle both.
    """
    score = competitor.get("score")
    if score is None:
        return None
    if isinstance(score, dict):
        score = score.get("value")
    if score is None:
        return None
    return int(score)


def normalize_group_name(raw: str | None) -> str | None:
    """ESPN's group labels come in inconsistent casing across editions
    ("GROUP D1", "Group B1", "LEAGUE D - GROUP 2") -- one display form."""
    if not raw or not raw.strip():
        return None
    return raw.strip().title()


def _parse_event(event: dict) -> EspnMatch | None:
    try:
        competition = event["competitions"][0]
        competitors = competition["competitors"]
        home = next(c for c in competitors if c["homeAway"] == "home")
        away = next(c for c in competitors if c["homeAway"] == "away")
        status_type = competition["status"]["type"]
        completed = bool(status_type["completed"])
        state = status_type.get("state", "pre")
        clock_label = competition["status"].get("displayClock") if state == "in" else None
        kickoff_utc = dt.datetime.fromisoformat(event["date"].replace("Z", "+00:00")).replace(tzinfo=None)
        # Real score exists once the match has actually started (state !=
        # "pre") -- not just once it's fully completed, so a genuinely live
        # match's running score is captured too (see ingest/live_scores.py).
        home_score = _parse_score(home) if state != "pre" else None
        away_score = _parse_score(away) if state != "pre" else None
        return EspnMatch(
            espn_home_id=str(home["team"]["id"]),
            espn_away_id=str(away["team"]["id"]),
            home_name=home["team"]["displayName"],
            away_name=away["team"]["displayName"],
            date=kickoff_utc.date(),
            kickoff_utc=kickoff_utc,
            completed=completed,
            home_score=home_score,
            away_score=away_score,
            state=state,
            clock_label=clock_label,
            group_name=normalize_group_name((competition.get("group") or {}).get("name")),
            stage=(event.get("seasonType") or {}).get("name"),
        )
    except (KeyError, IndexError, StopIteration, ValueError, TypeError, AttributeError):
        return None


def fetch_team_results(league_slug: str, espn_team_id: str, season: int) -> list[EspnMatch]:
    """This team's COMPLETED matches for `season`. Verified live that this
    endpoint never returns upcoming matches -- use fetch_day_fixtures for
    those. Empty list on any failure.
    """
    try:
        data = get(
            f"/{league_slug}/teams/{espn_team_id}/schedule",
            params={"season": season},
            cache_ttl_seconds=SCHEDULE_CACHE_TTL_SECONDS,
        )
    except requests.RequestException:
        return []

    matches = (_parse_event(e) for e in data.get("events", []))
    return [m for m in matches if m is not None and m.completed and m.home_score is not None]


def fetch_day_matches(
    league_slug: str, date: dt.date, cache_ttl_seconds: int = SCHEDULE_CACHE_TTL_SECONDS
) -> list[EspnMatch]:
    """Every match league-wide on `date` -- upcoming, live and finished.
    Empty list on any failure. Only a single date per call is supported by
    this endpoint. `cache_ttl_seconds` lets the live banner
    (ingest/live_scores.py) ask for a fresh copy -- the default 6h is far
    too stale for running scores.
    """
    try:
        data = get(
            f"/{league_slug}/scoreboard",
            params={"dates": date.strftime("%Y%m%d")},
            cache_ttl_seconds=cache_ttl_seconds,
        )
    except requests.RequestException:
        return []

    matches = (_parse_event(e) for e in data.get("events", []))
    return [m for m in matches if m is not None]


def fetch_day_fixtures(
    league_slug: str, date: dt.date, cache_ttl_seconds: int = SCHEDULE_CACHE_TTL_SECONDS
) -> list[EspnMatch]:
    """Every not-yet-completed match league-wide on `date` (so including
    ones in progress). Empty list on any failure."""
    return [m for m in fetch_day_matches(league_slug, date, cache_ttl_seconds) if not m.completed]


def fetch_group_membership(league_slug: str, season: int) -> dict[str, str]:
    """{espn team id: group name} for one edition of a group competition
    (e.g. the Nations League), from the standings endpoint's per-group
    children. Verified live that, unlike the scoreboard's per-match group
    label, this answers for past editions too (`season=` is the edition's
    start year) -- though ESPN's own data is incomplete for the oldest one
    (2018-19 lists only League D). Empty dict on any failure or for a league
    with no groups (a flat standings table has no `children`).
    """
    try:
        data = get(
            f"/{league_slug}/standings",
            params={"season": season},
            cache_ttl_seconds=SCHEDULE_CACHE_TTL_SECONDS,
            base_url=STANDINGS_BASE_URL,
        )
    except (requests.RequestException, ValueError):
        return {}

    membership: dict[str, str] = {}
    for child in data.get("children") or []:
        group = normalize_group_name(child.get("name"))
        if group is None:
            continue
        for entry in (child.get("standings") or {}).get("entries") or []:
            team_id = (entry.get("team") or {}).get("id")
            if team_id is not None:
                membership[str(team_id)] = group
    return membership


def fetch_team_roster(
    league_slug: str, espn_team_id: str
) -> tuple[list[EspnRosterPlayer], list[EspnInjuryEntry]]:
    """This team's roster and any flagged injuries, from one cached call.
    Returns ([], []) on any failure.
    """
    try:
        data = get(
            f"/{league_slug}/teams/{espn_team_id}",
            params={"enable": "roster,injuries,stats,standings"},
            cache_ttl_seconds=ROSTER_CACHE_TTL_SECONDS,
        )
    except requests.RequestException:
        return [], []

    athletes = data.get("team", {}).get("athletes", [])
    roster: list[EspnRosterPlayer] = []
    injuries: list[EspnInjuryEntry] = []
    for p in athletes:
        abbrev = (p.get("position") or {}).get("abbreviation") or ""
        bucket = _POSITION_BUCKET.get(abbrev, "attack")
        label = _POSITION_LABEL.get(abbrev, "Unknown")
        name = p.get("displayName") or p.get("fullName")
        if not name:
            continue
        roster.append(EspnRosterPlayer(name=name, position=label, position_bucket=bucket))
        for injury in p.get("injuries") or []:
            note = injury.get("status") or injury.get("type", {}).get("description") or ""
            injuries.append(EspnInjuryEntry(player_name=name, position_bucket=bucket, note=note))

    return roster, injuries


def fetch_team_conference(league_slug: str, espn_team_id: str) -> str | None:
    """This team's conference (e.g. "Eastern Conference"), or None if this
    league has no real conference split or the id isn't recognized. Uses
    the exact same request as fetch_team_roster (same path+params), so this
    shares its on-disk cache -- calling both for the same team costs one
    network request, not two.
    """
    try:
        data = get(
            f"/{league_slug}/teams/{espn_team_id}",
            params={"enable": "roster,injuries,stats,standings"},
            cache_ttl_seconds=ROSTER_CACHE_TTL_SECONDS,
        )
    except requests.RequestException:
        return None

    group_id = data.get("team", {}).get("groups", {}).get("id")
    return _CONFERENCE_NAME.get(group_id)
