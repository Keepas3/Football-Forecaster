"""La Liga and Champions League injuries from Biwenger's fantasy-football feed.

Biwenger (a Spanish fantasy game) publishes, for every player in a competition,
a status ("injured", "doubt", "sanctioned", ...) and a one-line explanation
("Hamstring injury in the left leg. Estimated return: Mid-November."). It is the
only free source this app has found for these two competitions: football-data.org
has no injuries and ESPN's injury fields are empty. The Premier League has its
own feed (ingest/fpl_client.py).

Caveats worth knowing: the feed is the one Biwenger's own website loads, so it
is public and needs no key, but it is undocumented and not offered under a data
licence -- it can change or be restricted without notice (every function here
degrades to "unknown" if it does). Player names are often a surname alone
("Saliba") or a nickname, and the explanation text is free-form. It is fetched
at most once every few hours per competition and cached.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import requests
from rapidfuzz import fuzz, process

from soccer_predictor.config import DATA_DIR
from soccer_predictor.ingest import http_session

# This app's league code -> Biwenger's competition slug.
COMPETITIONS = {"LALIGA": "la-liga", "UCL": "champions-league"}
URL = "https://cf.biwenger.com/api/v2/competitions/{slug}/data?lang=en"
CACHE_DIR = DATA_DIR / "cache" / "biwenger"
# Injury news changes through the week, not by the minute.
CACHE_TTL_SECONDS = 3 * 3600

# Biwenger position 1-4 (goalkeeper, defender, midfielder, forward) -> this app's
# bucket: goalkeepers and defenders count as "defense", the rest as "attack".
_POSITION_BUCKET = {1: "defense", 2: "defense", 3: "attack", 4: "attack"}
# "warned" (one yellow card from a ban) and "unknown" are not unavailability.
_STATUSES = {"injured", "doubt", "sanctioned"}
# Where a doubtful player has no stated chance, take it as a coin flip.
DEFAULT_CHANCE_OF_PLAYING = 50

# Biwenger's club names this app's own can't be matched to by words alone (a
# translation, an abbreviation, a different spelling), keyed by the accent- and
# case-folded name -- see fold_name. Values are this app's names.
TEAM_NAME_OVERRIDES = {
    "psg": "Paris Saint-Germain FC",
    "inter": "FC Internazionale Milano",
    "bayern munich": "FC Bayern München",
    "aek athen": "PAE AEK",
    "slavia prag": "SK Slavia Praha",
}
TEAM_NAME_FUZZY_THRESHOLD = 95

# Biwenger's English text is machine-translated; "golpe" (a knock) comes out as
# "Stroke", which on an injury list reads as a medical emergency.
_TRANSLATION_FIXES = {"stroke": "Knock"}

_RETURN = re.compile(r"estimated return:\s*(?P<when>.+?)\.?\s*$", re.IGNORECASE)
_MONTH_DAY_YEAR = re.compile(r"^([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})$")
_UNKNOWN_RETURNS = ("undetermined", "indefinite", "unknown")


@dataclass
class BiwengerInjury:
    player_name: str  # often just a surname
    team_name: str  # Biwenger's name for the club
    position_bucket: str  # "attack" | "defense"
    status: str  # "injured" | "doubt" | "sanctioned"
    chance_of_playing: int  # 0-100: how likely he is to play the next match
    news: str  # normalised, e.g. "Hamstring injury - Expected back mid-November"
    games_played: int = 0  # league games this season
    price_share: float = 0.0  # his market price as a share of his club's most expensive player


def _cache_file(slug: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha256(URL.format(slug=slug).encode()).hexdigest()}.json"


def _load_cached(slug: str) -> dict | None:
    cache_file = _cache_file(slug)
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_TTL_SECONDS:
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def fetch_competition(slug: str) -> dict | None:
    """A competition's whole player/team payload, from the disk cache within its
    lifetime. None if it can't be fetched (callers must treat that as "unknown",
    not "nobody is injured")."""
    cached = _load_cached(slug)
    if cached is not None:
        return cached
    try:
        response = http_session.get(URL.format(slug=slug), headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        data = (response.json() or {}).get("data")
    except (requests.RequestException, ValueError, AttributeError):
        return None
    if not isinstance(data, dict) or not data.get("players") or not data.get("teams"):
        return None
    cache_file = _cache_file(slug)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data


def _return_phrase(when: str) -> str:
    """Biwenger's "Estimated return" wording as the feed-neutral phrase the
    injuries table reads: "Nov 25, 2026" -> "Expected back 25 Nov 2026", "Mid-November"
    -> "Expected back mid-November", "Undetermined" -> "Unknown return date"."""
    when = when.strip().rstrip(".")
    if not when or any(word in when.lower() for word in _UNKNOWN_RETURNS):
        return "Unknown return date"
    dated = _MONTH_DAY_YEAR.match(when)
    if dated:
        return f"Expected back {int(dated.group(2))} {dated.group(1).title()} {dated.group(3)}"
    return f"Expected back {when[0].lower() + when[1:]}"


def clean_news(status: str, status_info: str | None) -> str:
    """The free-form explanation as "Problem - When". Biwenger writes things like
    "Ankle injury.. Estimated return: Nov 25, 2026", "Muscle discomfort." or
    "Direct red card. Suspended through Matchday 4."; a suspension always says so."""
    text = (status_info or "").strip()
    when = ""
    return_match = _RETURN.search(text)
    if return_match:
        when = _return_phrase(return_match.group("when"))
        text = text[: return_match.start()]
    problem, _, rest = text.strip().partition(". ")
    problem = _TRANSLATION_FIXES.get(problem.strip(" .").lower(), problem.strip(" ."))
    rest = rest.strip(" .")
    if rest and not when:
        when = rest
    if status == "sanctioned" and "suspend" not in f"{problem} {when}".lower():
        when = f"Suspended - {when}" if when else "Suspended"
    if not problem and status == "injured":
        problem = "Injury"
    return " - ".join(part for part in (problem, when) if part)


def parse_injuries(data: dict) -> list[BiwengerInjury]:
    """Every injured, doubtful or suspended player in a competition payload."""
    teams = {str(key): team["name"] for key, team in (data.get("teams") or {}).items() if "name" in team}
    players = [p for p in (data.get("players") or {}).values() if isinstance(p, dict)]
    top_price: dict[object, int] = {}
    for player in players:
        try:
            top_price[player["teamID"]] = max(top_price.get(player["teamID"], 0), int(player.get("price") or 0))
        except (KeyError, TypeError, ValueError):
            continue

    injuries = []
    for player in players:
        try:
            status = player["status"]
            if status not in _STATUSES or str(player["teamID"]) not in teams:
                continue
            chance = DEFAULT_CHANCE_OF_PLAYING if status == "doubt" else 0
            top = top_price.get(player["teamID"]) or 0
            injuries.append(
                BiwengerInjury(
                    player_name=str(player["name"]).strip(),
                    team_name=teams[str(player["teamID"])],
                    position_bucket=_POSITION_BUCKET.get(player.get("position"), "attack"),
                    status=status,
                    chance_of_playing=chance,
                    news=clean_news(status, player.get("statusInfo")),
                    games_played=int(player.get("playedHome") or 0) + int(player.get("playedAway") or 0),
                    price_share=min(1.0, int(player.get("price") or 0) / top) if top else 0.0,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return injuries


def fold_name(name: str) -> str:
    """Lowercase, accents removed, punctuation turned into spaces."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", plain.lower()).split())


def resolve_team_name(biwenger_name: str, our_team_names: list[str]) -> str | None:
    """Our name for a Biwenger club ("Atlético" -> "Atletico Madrid"), or None
    if nothing matches clearly. The words of one name must be all of the other's
    ("Lens" in "Racing Club de Lens"), and if several of ours fit equally well
    ("Manchester" alone) none is taken -- a wrong club is worse than no club."""
    folded = fold_name(biwenger_name)
    wanted = fold_name(TEAM_NAME_OVERRIDES.get(folded, biwenger_name))
    by_folded = {fold_name(name): name for name in our_team_names}
    if wanted in by_folded:  # an exact match must never lose to a lookalike
        return by_folded[wanted]
    scored = process.extract(wanted, list(by_folded), scorer=fuzz.token_set_ratio, limit=2)
    if not scored or scored[0][1] < TEAM_NAME_FUZZY_THRESHOLD:
        return None
    if len(scored) > 1 and scored[1][1] >= scored[0][1]:
        return None
    return by_folded[scored[0][0]]


def fetch_injuries_by_club(league_code: str) -> dict[str, list[BiwengerInjury]] | None:
    """{Biwenger club name: its injured, doubtful and suspended players} for
    every club in the competition (an empty list for a club with none), or None
    if the league has no Biwenger competition or the feed is unavailable -- which
    is "unknown", not "nobody is injured", so callers leave whatever they have."""
    slug = COMPETITIONS.get(league_code)
    if slug is None:
        return None
    data = fetch_competition(slug)
    if data is None:
        return None
    by_club: dict[str, list[BiwengerInjury]] = {
        team["name"]: [] for team in (data.get("teams") or {}).values() if "name" in team
    }
    for injury in parse_injuries(data):
        by_club.setdefault(injury.team_name, []).append(injury)
    return by_club
