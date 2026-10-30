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
import os
import re
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import requests
from rapidfuzz import fuzz

from soccer_predictor.config import DATA_DIR
from soccer_predictor.ingest import http_session
from soccer_predictor.ingest.memo import ttl_memo

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
# The standings endpoint lives under a different path (/apis/v2, not
# /apis/site/v2) -- see fetch_group_membership.
STANDINGS_BASE_URL = "https://site.api.espn.com/apis/v2/sports/soccer"
CACHE_DIR = DATA_DIR / "cache" / "espn"
DEFAULT_MAX_REQUESTS_PER_MINUTE = 60  # self-imposed courtesy limit -- no published cap exists
HARD_MAX_REQUESTS_PER_MINUTE = 120


def _configured_max_requests_per_minute() -> int:
    """One request a second by default (it was one every two seconds, which with
    kept-alive connections -- ingest/http_session.py -- made the limiter, not the
    network, the bottleneck). ESPN_MAX_REQUESTS_PER_MINUTE lowers or raises it,
    e.g. if ESPN ever starts refusing requests from a CI runner."""
    try:
        value = int(os.environ.get("ESPN_MAX_REQUESTS_PER_MINUTE", DEFAULT_MAX_REQUESTS_PER_MINUTE))
    except ValueError:
        return DEFAULT_MAX_REQUESTS_PER_MINUTE
    return max(1, min(HARD_MAX_REQUESTS_PER_MINUTE, value))


MAX_REQUESTS_PER_MINUTE = _configured_max_requests_per_minute()

# A team's schedule/roster changes slowly; a scoreboard day is finalized
# once played. 6h mirrors ingest/fixtures.py's own TTLs for the same reason
# (short enough to pick up new results without hammering an unofficial
# endpoint on every dashboard load).
SCHEDULE_CACHE_TTL_SECONDS = 6 * 3600
TEAM_LIST_CACHE_TTL_SECONDS = 24 * 3600
ROSTER_CACHE_TTL_SECONDS = 24 * 3600
# A finished match's lineups/stats never change; a live one's change every
# minute; a not-yet-started one only gains lineups about an hour before kickoff.
MATCH_DETAIL_CACHE_TTL_SECONDS = {"post": 24 * 3600, "in": 30, "pre": 300}
STAT_LEADERS_CACHE_TTL_SECONDS = 6 * 3600

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
        # The live banner (ingest/live_scores.py) fetches from several threads
        # at once; the lock keeps the shared timestamp window consistent (a
        # caller that has to sleep makes the others queue behind it, which is
        # exactly the throttling wanted).
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
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
    response = http_session.get(f"{base_url}{path}", params=params, timeout=15)
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
    # ESPN's id for the match itself -- what the match detail page (see
    # fetch_match_detail) is keyed by.
    event_id: str | None = None


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
    # 2019's Asian Cup labels its last group "AFC ASIAN CUP - GROUP F"; drop that
    # competition prefix so it reads like its siblings ("Group F").
    label = re.sub(r"^afc asian cup\s*-\s*", "", raw.strip(), flags=re.IGNORECASE)
    return label.title()


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
            event_id=str(event["id"]) if event.get("id") else None,
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


def fetch_year_fixtures(
    league_slug: str, year: int, cache_ttl_seconds: int = SCHEDULE_CACHE_TTL_SECONDS
) -> list[EspnMatch]:
    """Every not-yet-completed match the league has scheduled in calendar year
    `year`, in ONE request -- verified live that `dates=<year>` returns the
    whole year's scoreboard (51 events for the 2027 Asian Cup). The day-by-day
    scoreboard only reaches a couple of weeks ahead, so on its own a tournament
    that starts in three months has no fixtures (and so no group tables or
    predictions) until it's nearly here. Empty list on any failure.
    """
    try:
        data = get(
            f"/{league_slug}/scoreboard",
            params={"dates": str(year)},
            cache_ttl_seconds=cache_ttl_seconds,
        )
    except requests.RequestException:
        return []

    matches = (_parse_event(e) for e in data.get("events", []))
    return [m for m in matches if m is not None and not m.completed]


def fetch_day_fixtures(
    league_slug: str, date: dt.date, cache_ttl_seconds: int = SCHEDULE_CACHE_TTL_SECONDS
) -> list[EspnMatch]:
    """Every not-yet-completed match league-wide on `date` (so including
    ones in progress). Empty list on any failure."""
    return [m for m in fetch_day_matches(league_slug, date, cache_ttl_seconds) if not m.completed]


@dataclass
class EspnParticipant:
    """A team that played an edition of a group competition, as listed by that
    edition's standings."""

    espn_id: str
    name: str
    group: str
    logo_url: str | None = None


def fetch_group_participants(league_slug: str, season: int) -> list[EspnParticipant]:
    """Every team in one edition of a group competition (e.g. the Nations
    League, the Asian Cup), with its group, from the standings endpoint's
    per-group children. Verified live that, unlike the scoreboard's per-match
    group label, this answers for past editions too (`season=` is the
    edition's start year) -- though ESPN's own data is incomplete for the
    oldest Nations League edition (2018-19 lists only League D). Empty list on
    any failure or for a league with no groups (a flat standings table has no
    `children`).
    """
    try:
        data = get(
            f"/{league_slug}/standings",
            params={"season": season},
            cache_ttl_seconds=SCHEDULE_CACHE_TTL_SECONDS,
            base_url=STANDINGS_BASE_URL,
        )
    except (requests.RequestException, ValueError):
        return []

    participants: list[EspnParticipant] = []
    for child in data.get("children") or []:
        group = normalize_group_name(child.get("name"))
        if group is None:
            continue
        for entry in (child.get("standings") or {}).get("entries") or []:
            team = entry.get("team") or {}
            if team.get("id") is None:
                continue
            logos = team.get("logos") or []
            participants.append(
                EspnParticipant(
                    espn_id=str(team["id"]),
                    name=team.get("displayName") or team.get("name") or f"team#{team['id']}",
                    group=group,
                    logo_url=logos[0].get("href") if logos else None,
                )
            )
    return participants


def fetch_group_membership(league_slug: str, season: int) -> dict[str, str]:
    """{espn team id: group name} for one edition -- see fetch_group_participants."""
    return {p.espn_id: p.group for p in fetch_group_participants(league_slug, season)}


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


@dataclass
class LineupPlayer:
    name: str
    jersey: str
    position: str  # ESPN's abbreviation: "G", "D", "M", "F", "SUB"...
    starter: bool
    subbed_in: bool = False
    subbed_out: bool = False
    formation_place: int | None = None  # ESPN's slot number in the starting formation (1 = goalkeeper)
    sub_minute: str | None = None  # when he came on or went off, e.g. "64'" (only for a substitution)
    # When each kind of thing happened to him, from his match plays: "goal" (not own
    # goals), "assist", "yellow", "red", "sub" -> minute labels in order, e.g. ["23'", "90'+2'"].
    event_minutes: dict[str, list[str]] = field(default_factory=dict)
    # ESPN stat name -> count, only for the stats in PLAYER_STAT_CHIPS and only
    # when non-zero.
    stats: dict[str, int] = field(default_factory=dict)


@dataclass
class TeamMatchInfo:
    espn_id: str
    name: str
    home_away: str  # "home" | "away"
    formation: str | None
    # stat name -> (ESPN's label, display value), e.g. "possessionPct" -> ("Possession", "58.9")
    stats: dict[str, tuple[str, str]] = field(default_factory=dict)
    players: list[LineupPlayer] = field(default_factory=list)
    logo: str | None = None


@dataclass
class MatchEvent:
    minute: str  # "25'", "90'+3'"
    kind: str  # "goal" | "yellow" | "red" | "sub"
    text: str
    team_id: str | None
    # Who it was about (the scorer, the player carded or, for a substitution, the
    # one coming on). For an own goal, `team_id` is the side that is credited.
    player: str | None = None


@dataclass
class MatchDetail:
    event_id: str
    state: str  # "pre" | "in" | "post"
    status_label: str  # e.g. "FT", "HT", "63'"
    home: TeamMatchInfo
    away: TeamMatchInfo
    home_score: int | None
    away_score: int | None
    venue: str | None
    city: str | None
    attendance: int | None
    events: list[MatchEvent] = field(default_factory=list)
    match_date: dt.date | None = None  # UTC kickoff date

    @property
    def has_lineups(self) -> bool:
        return bool(self.home.players or self.away.players)


# ESPN keyEvents "type.type" values -> the few kinds the timeline shows
# (kickoff, half-time, delays etc. are noise here).
_EVENT_KIND = {
    "goal": "goal",
    "penalty---scored": "goal",
    "own-goal": "goal",
    "yellow-card": "yellow",
    "red-card": "red",
    "substitution": "sub",
}
# ESPN's per-player stat names -> short chip label, in display order.
PLAYER_STAT_CHIPS = {
    "totalGoals": "goals",
    "goalAssists": "assists",
    "totalShots": "shots",
    "shotsOnTarget": "on target",
    "saves": "saves",
    "yellowCards": "yellow",
    "redCards": "red",
}


def _to_int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _sub_minute(plays: list[dict] | None) -> str | None:
    """The minute of the player's substitution, from his match plays."""
    for play in plays or []:
        if play.get("substitution"):
            return (play.get("clock") or {}).get("displayValue") or None
    return None


def _play_minutes(plays: list[dict] | None) -> dict[str, list[str]]:
    """Minute labels by kind (see LineupPlayer.event_minutes) from a player's match plays."""
    minutes: dict[str, list[str]] = {}
    for play in plays or []:
        label = (play.get("clock") or {}).get("displayValue")
        if not label:
            continue
        kinds = []
        if play.get("didScore") and not play.get("ownGoal"):
            kinds.append("goal")
        if play.get("didAssist"):
            kinds.append("assist")
        if play.get("yellowCard"):
            kinds.append("yellow")
        if play.get("redCard"):
            kinds.append("red")
        if play.get("substitution"):
            kinds.append("sub")
        for kind in kinds:
            minutes.setdefault(kind, []).append(label)
    return minutes


def _parse_lineup_players(roster: list[dict] | None) -> list[LineupPlayer]:
    players = []
    for entry in roster or []:
        athlete = entry.get("athlete") or {}
        name = athlete.get("displayName") or athlete.get("fullName")
        if not name:
            continue
        stats: dict[str, int] = {}
        for stat in entry.get("stats") or []:
            if stat.get("name") in PLAYER_STAT_CHIPS:
                value = _to_int(stat.get("value"))
                if value:
                    stats[stat["name"]] = value
        players.append(
            LineupPlayer(
                name=name,
                jersey=str(entry.get("jersey") or ""),
                position=(entry.get("position") or {}).get("abbreviation") or "",
                starter=bool(entry.get("starter")),
                subbed_in=bool(entry.get("subbedIn")),
                subbed_out=bool(entry.get("subbedOut")),
                formation_place=_to_int(entry.get("formationPlace")),
                sub_minute=_sub_minute(entry.get("plays")),
                event_minutes=_play_minutes(entry.get("plays")),
                stats=stats,
            )
        )
    return players


def _event_kind(event: dict) -> str | None:
    event_type = (event.get("type") or {}).get("type") or ""
    kind = _EVENT_KIND.get(event_type)
    # ESPN has many goal types ("goal---header", "goal---free-kick", ...); every
    # one of them is flagged as a scoring play, so go by that rather than a list.
    if kind is None and (event.get("scoringPlay") or event_type.startswith("goal")):
        kind = "goal"
    return kind


def _parse_match_events(key_events: list[dict] | None) -> list[MatchEvent]:
    events = []
    for event in key_events or []:
        if event.get("shootout"):  # penalty shoot-out kicks aren't goals in the match
            continue
        kind = _event_kind(event)
        if kind is None:
            continue
        team_id = (event.get("team") or {}).get("id")
        participants = event.get("participants") or []
        player = ((participants[0] if participants else {}).get("athlete") or {}).get("displayName")
        events.append(
            MatchEvent(
                minute=(event.get("clock") or {}).get("displayValue") or "",
                kind=kind,
                text=event.get("text") or event.get("shortText") or "",
                team_id=str(team_id) if team_id else None,
                player=player,
            )
        )
    return events


def _utc_date(iso: str | None) -> dt.date | None:
    try:
        return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).date()
    except (AttributeError, ValueError):
        return None


def _team_logo(team: dict) -> str | None:
    if team.get("logo"):
        return team["logo"]
    logos = team.get("logos") or []
    return logos[0].get("href") if logos else None


def parse_match_detail(event_id: str, data: dict) -> MatchDetail | None:
    """None when the payload doesn't look like a match summary."""
    try:
        competition = data["header"]["competitions"][0]
        status_type = competition["status"]["type"]
        competitors = {c["homeAway"]: c for c in competition["competitors"]}
        home_competitor, away_competitor = competitors["home"], competitors["away"]
        str(home_competitor["team"]["id"]), str(away_competitor["team"]["id"])
    except (KeyError, IndexError, TypeError, AttributeError):
        return None

    boxscore_by_team: dict[str, dict[str, tuple[str, str]]] = {}
    for team in (data.get("boxscore") or {}).get("teams") or []:
        team_id = str((team.get("team") or {}).get("id"))
        boxscore_by_team[team_id] = {
            stat["name"]: (stat.get("label") or stat["name"], str(stat.get("displayValue", "")))
            for stat in team.get("statistics") or []
            if stat.get("name")
        }
    roster_by_team = {str((r.get("team") or {}).get("id")): r for r in data.get("rosters") or []}

    def team_info(side: str, competitor: dict) -> TeamMatchInfo:
        team_id = str(competitor["team"]["id"])
        roster = roster_by_team.get(team_id) or {}
        return TeamMatchInfo(
            espn_id=team_id,
            name=competitor["team"].get("displayName") or "",
            home_away=side,
            formation=roster.get("formation") or None,
            stats=boxscore_by_team.get(team_id, {}),
            players=_parse_lineup_players(roster.get("roster")),
            logo=_team_logo(competitor["team"]),
        )

    game_info = data.get("gameInfo") or {}
    venue = game_info.get("venue") or {}
    state = status_type.get("state", "pre")
    return MatchDetail(
        event_id=event_id,
        state=state,
        status_label=status_type.get("shortDetail") or status_type.get("detail") or "",
        home=team_info("home", home_competitor),
        away=team_info("away", away_competitor),
        home_score=_to_int(home_competitor.get("score")) if state != "pre" else None,
        away_score=_to_int(away_competitor.get("score")) if state != "pre" else None,
        venue=venue.get("fullName"),
        city=(venue.get("address") or {}).get("city"),
        attendance=_to_int(game_info.get("attendance")) or None,
        events=_parse_match_events(data.get("keyEvents")),
        match_date=_utc_date(competition.get("date")),
    )


def fetch_match_detail(
    league_slug: str, event_id: str, state: str = "post", cache_ttl_seconds: int | None = None
) -> MatchDetail | None:
    """Lineups, team/player match stats and the goals/cards/subs timeline for
    one match, from ESPN's summary endpoint (keyless, same host as the rest of
    this client). `state` ("pre"/"in"/"post") only picks the cache lifetime,
    unless `cache_ttl_seconds` is given (0 = always fetch: the prediction lock
    must not read a "no lineups yet" page cached a few minutes ago).
    None on any failure -- the page shows a friendly caption instead.
    """
    if cache_ttl_seconds is None:
        cache_ttl_seconds = MATCH_DETAIL_CACHE_TTL_SECONDS.get(state, 300)
    try:
        data = get(
            f"/{league_slug}/summary",
            params={"event": event_id},
            cache_ttl_seconds=cache_ttl_seconds,
        )
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return parse_match_detail(event_id, data)


def find_event_id(
    league_slug: str, date: dt.date, espn_home_id: str, espn_away_id: str
) -> tuple[str, str] | None:
    """(event id, state) of the match between these two ESPN teams on `date`,
    found on that day's scoreboard (one 6h-cached request per day). Used when
    the database has no stored event id for a match. None if not found."""
    for match in fetch_day_matches(league_slug, date):
        if (
            match.event_id
            and match.espn_home_id == str(espn_home_id)
            and match.espn_away_id == str(espn_away_id)
        ):
            return match.event_id, match.state
    return None


TEAM_NAME_FUZZY_THRESHOLD = 75  # same tolerance as the other clients' team matching
# Club-form words that differ between sources ("Arsenal FC" vs "Arsenal").
_NAME_NOISE = {"fc", "afc", "cf", "sc", "ac", "ssc", "fk", "sk"}
# Words ESPN spells differently from the football-data.org names we store, too far
# apart for fuzzy matching to bridge ("FC Cologne" v "FC Koln", "Stade Rennais" v
# "Rennes"): each is mapped to the spelling we use before comparing.
_NAME_WORD_ALIASES = {"cologne": "koln", "rennais": "rennes"}


def _name_key(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    words = re.sub(r"[^a-z0-9 ]+", " ", ascii_name.casefold()).split()
    return " ".join(_NAME_WORD_ALIASES.get(w, w) for w in words if w not in _NAME_NOISE)


def _name_score(a: str, b: str) -> float:
    return fuzz.WRatio(_name_key(a), _name_key(b))


def find_event_by_names(
    league_slug: str,
    date: dt.date,
    home_name: str,
    away_name: str,
    cache_ttl_seconds: int = SCHEDULE_CACHE_TTL_SECONDS,
) -> tuple[str, str] | None:
    """(event id, state) of the match between teams NAMED like this, for
    leagues whose teams carry no ESPN id (the football-data.org ones): scans
    the scoreboards of `date` and the days either side (ESPN files late
    kickoffs under a neighbouring day) and takes the best-matching pair, home
    and away both above TEAM_NAME_FUZZY_THRESHOLD. The given date wins over a
    neighbour when it has a match. None if nothing matches."""
    best: tuple[float, str, str] | None = None
    for offset in (0, -1, 1):
        for match in fetch_day_matches(league_slug, date + dt.timedelta(days=offset), cache_ttl_seconds):
            if not match.event_id:
                continue
            home_score = _name_score(home_name, match.home_name)
            away_score = _name_score(away_name, match.away_name)
            if min(home_score, away_score) < TEAM_NAME_FUZZY_THRESHOLD:
                continue
            if best is None or home_score + away_score > best[0]:
                best = (home_score + away_score, match.event_id, match.state)
        if best is not None:
            break
    return (best[1], best[2]) if best else None


@dataclass
class StatLeader:
    name: str
    team: str
    goals: int
    assists: int


@ttl_memo(300)
def fetch_stat_leaders(league_slug: str, season: int | None = None) -> list[StatLeader]:
    """The league's top scorers and assist makers (50 of each) in ONE request to
    ESPN's statistics endpoint, merged by player. ESPN gives no minutes or xG
    here. `season` is the edition's start year; without it ESPN serves the
    current edition, which is empty for a tournament that has not started
    (the Asian Cup between editions). Empty list on any failure."""
    try:
        data = get(
            f"/{league_slug}/statistics",
            params={"season": season} if season else None,
            cache_ttl_seconds=STAT_LEADERS_CACHE_TTL_SECONDS,
        )
    except (requests.RequestException, ValueError):
        return []

    players: dict[str, StatLeader] = {}
    for category in data.get("stats") or []:
        field_name = {"goalsLeaders": "goals", "assistsLeaders": "assists"}.get(category.get("name"))
        if field_name is None:
            continue
        for leader in category.get("leaders") or []:
            athlete = leader.get("athlete") or {}
            name = athlete.get("displayName")
            if not name or athlete.get("id") is None:
                continue
            entry = players.setdefault(
                str(athlete["id"]),
                StatLeader(name=name, team=(athlete.get("team") or {}).get("displayName") or "", goals=0, assists=0),
            )
            setattr(entry, field_name, _to_int(leader.get("value")) or 0)
    return list(players.values())
