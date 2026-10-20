"""Premier League injuries and suspensions from the Fantasy Premier League feed.

The Premier League's own fantasy game publishes, for every player in the league,
a status ("a" available, "d" doubtful, "i" injured, "s" suspended, "u" unavailable
for other reasons such as leaving the club), a one-line news item ("Ankle injury
- Expected back 24 Oct") and a chance of playing. It is the only free source of
injuries this app has found: ESPN's injury fields are empty for every league and
football-data.org has none.

Caveats worth knowing: the feed is public and needs no key but is undocumented
and not offered under a data licence, so it can change or be restricted without
notice (every function here degrades to "nothing" if it does), and it covers the
Premier League only. It is fetched at most once every few hours and cached.
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
from soccer_predictor.ingest import http_session

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
CACHE_FILE = DATA_DIR / "cache" / "fpl" / f"{hashlib.sha256(URL.encode()).hexdigest()}.json"
# Injury news changes through the week (press conferences, training), but not by the minute.
CACHE_TTL_SECONDS = 3 * 3600

# FPL's element_type -> this app's position bucket (goalkeepers and defenders
# count as "defense"; midfielders and forwards as "attack", as everywhere else).
_POSITION_BUCKET = {1: "defense", 2: "defense", 3: "attack", 4: "attack"}
# Out for the game (injured, suspended) or in doubt. "u" (unavailable: loaned,
# sold, not in the squad) is not an injury and is left out.
_STATUSES = {"i", "d", "s"}
# Where a doubtful player has no stated chance, take it as a coin flip.
DEFAULT_CHANCE_OF_PLAYING = 50

# FPL's short club names -> the names this app stores (football-data.co.uk's).
TEAM_NAME_OVERRIDES = {"Spurs": "Tottenham", "Man Utd": "Man United"}
TEAM_NAME_FUZZY_THRESHOLD = 80


@dataclass
class FplInjury:
    player_name: str  # full name, for matching against other sources
    team_name: str  # FPL's name for the club
    position_bucket: str  # "attack" | "defense"
    status: str  # "i" | "d" | "s"
    chance_of_playing: int  # 0-100: how likely he is to play the next match
    news: str  # e.g. "Ankle injury - Expected back 24 Oct"
    minutes: int = 0  # minutes played this season: how much he features


def _load_cached() -> dict | None:
    if CACHE_FILE.exists() and time.time() - CACHE_FILE.stat().st_mtime < CACHE_TTL_SECONDS:
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def fetch_bootstrap() -> dict | None:
    """The whole feed, from the disk cache within its lifetime. None if it can't
    be fetched (callers must treat that as "unknown", not "nobody is injured")."""
    cached = _load_cached()
    if cached is not None:
        return cached
    try:
        response = http_session.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("elements"):
        return None
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data), encoding="utf-8")
    return data


def parse_injuries(data: dict) -> list[FplInjury]:
    """Every injured, doubtful or suspended player in a feed payload."""
    teams = {t["id"]: t["name"] for t in data.get("teams") or [] if "id" in t and "name" in t}
    injuries = []
    for element in data.get("elements") or []:
        try:
            status = element["status"]
            if status not in _STATUSES:
                continue
            chance = element.get("chance_of_playing_next_round")
            if status == "d":
                chance = DEFAULT_CHANCE_OF_PLAYING if chance is None else int(chance)
            else:
                chance = 0 if chance is None else int(chance)
            injuries.append(
                FplInjury(
                    player_name=f"{element.get('first_name', '')} {element.get('second_name', '')}".strip()
                    or element["web_name"],
                    team_name=teams[element["team"]],
                    position_bucket=_POSITION_BUCKET.get(element.get("element_type"), "attack"),
                    status=status,
                    chance_of_playing=max(0, min(100, chance)),
                    news=(element.get("news") or "").strip(),
                    minutes=int(element.get("minutes") or 0),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return injuries


def resolve_team_name(fpl_name: str, our_team_names: list[str]) -> str | None:
    """Our name for an FPL club ("Spurs" -> "Tottenham"), or None if nothing is close."""
    wanted = TEAM_NAME_OVERRIDES.get(fpl_name, fpl_name)
    for name in our_team_names:  # an exact match must never lose to a lookalike ("Man City" / "Man United")
        if name.casefold() == wanted.casefold():
            return name
    match = process.extractOne(wanted, our_team_names, scorer=fuzz.WRatio)
    if match is None or match[1] < TEAM_NAME_FUZZY_THRESHOLD:
        return None
    return match[0]


def fetch_injuries_by_club() -> dict[str, list[FplInjury]] | None:
    """{FPL club name: its injured, doubtful and suspended players} for every
    club in the league (an empty list for a club with none), or None if the feed
    is unavailable -- which is "unknown", not "nobody is injured", so callers
    leave whatever they already have."""
    data = fetch_bootstrap()
    if data is None:
        return None
    by_club: dict[str, list[FplInjury]] = {t["name"]: [] for t in data.get("teams") or [] if "name" in t}
    for injury in parse_injuries(data):
        by_club.setdefault(injury.team_name, []).append(injury)
    return by_club
