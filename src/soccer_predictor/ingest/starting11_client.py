"""MLS, Bundesliga, Serie A and Ligue 1 injuries from starting11.com.

Starting 11 (a lineup-prediction site) publishes a page per league listing every
player out, doubtful or suspended, grouped by club: shirt number, surname,
position, what's wrong ("Hip Injury", "Red Card") and a status badge. It is the
only free source this app has found for these four leagues: football-data.org has
no injuries, ESPN's injury fields are empty, and the Premier League, La Liga and
Champions League have feeds of their own (ingest/fpl_client.py,
ingest/biwenger_client.py).

Caveats worth knowing: there is no data API for it, so the page's HTML is read --
a redesign will break it (every function here degrades to "unknown" if it does,
and the parse is checked against the page's own headline totals so a half-read
page is rejected rather than trusted). The site's terms say its football data
comes from a third-party provider and bar copying its proprietary content
without permission, though its robots.txt welcomes crawlers; it is requested once
per league per daily refresh, under a User-Agent that names this project. The
list has surnames only, no return dates and no chance of playing, and its
"doubtful" is loose (many more than the Premier League's own feed shows).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path

import requests

from soccer_predictor.config import DATA_DIR
from soccer_predictor.ingest import club_names, http_session

# This app's league code -> the page name on starting11.com.
COMPETITIONS = {"MLS": "mls", "BUNDESLIGA": "bundesliga", "SERIEA": "serie-a", "LIGUE1": "ligue-1"}
URL = "https://starting11.com/injuries/{slug}"
USER_AGENT = "Football-Forecaster/1.0 (+https://github.com/Keepas3/Football-Forecaster; hobby project)"
CACHE_DIR = DATA_DIR / "cache" / "starting11"
# Injury news changes through the week, not by the minute.
CACHE_TTL_SECONDS = 3 * 3600

# The page's position labels -> this app's bucket: goalkeepers and defenders
# count as "defense", the rest as "attack".
_POSITION_BUCKET = {"GK": "defense", "DEF": "defense", "MID": "attack", "FWD": "attack"}
STATUSES = ("Out", "Doubtful", "Suspended")
# The feed gives no chance of playing, and its doubts are loose, so a doubtful
# player is taken as likely to play -- a quarter of his usual weight.
DOUBTFUL_CHANCE_OF_PLAYING = 75

# Starting 11's club names this app's own can't be matched to by words alone,
# keyed by the accent- and case-folded name (see club_names.fold_name).
TEAM_NAME_OVERRIDES = {
    "dc united": "D.C. United",
    "los angeles fc": "LAFC",
    "los angeles galaxy": "LA Galaxy",
    "bayern munchen": "Bayern Munich",
    "borussia monchengladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt",
    "hamburger sv": "Hamburg",
    "paris saint germain": "Paris SG",
    "stade brestois 29": "Brest",
}

# Reasons the page lists that are about the squad, not about being hurt or banned:
# a player who has left or been loaned out, or is not in the matchday squad, is
# not a loss to the club's strength the way an injured starter is.
_NOT_AN_INJURY = {"off the roster", "loan agreement", "inactive", "coach's decision"}

_TOTALS = re.compile(r"All injuries\s+(\d+)\s+Out\s+(\d+)\s+Doubtful\s+(\d+)\s+Suspended")


@dataclass
class Starting11Injury:
    player_name: str  # a surname
    team_name: str  # Starting 11's name for the club
    position_bucket: str  # "attack" | "defense"
    status: str  # one of STATUSES
    chance_of_playing: int  # 0-100: how likely he is to play the next match
    problem: str  # e.g. "Hip Injury", "Red Card"


class _PageParser(HTMLParser):
    """Reads the club cards: <article id="club-..."> holding an <h3> club name and
    one <li> per player. Within a row the surname is the link to /player/..., the
    problem is the <p>, and the shirt number, position and status are the other
    texts -- told apart by their values, not their order, so a shuffled layout
    still reads right."""

    def __init__(self) -> None:
        super().__init__()
        self.clubs: dict[str, list[dict]] = {}
        self._club: str | None = None
        self._in_club_card = False
        self._in_heading = False
        self._heading = ""
        self._row: dict | None = None
        self._capture: str | None = None  # "name" | "problem" | "other"

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attributes = dict(attrs)
        if tag == "article" and (attributes.get("id") or "").startswith("club-"):
            self._in_club_card, self._club = True, None
        elif not self._in_club_card:
            return
        elif tag == "h3":
            self._in_heading, self._heading = True, ""
        elif tag == "li" and self._club is not None:
            self._row = {"name": "", "problem": "", "other": []}
        elif self._row is not None:
            if tag == "a" and (attributes.get("href") or "").startswith("/player/"):
                self._capture = "name"
            elif tag == "p":
                self._capture = "problem"
            else:
                self._capture = "other"

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self._in_club_card:
            self._in_club_card, self._club, self._row = False, None, None
        elif tag == "h3" and self._in_heading:
            self._in_heading = False
            self._club = " ".join(self._heading.split())
            if self._club:
                self.clubs.setdefault(self._club, [])
        elif tag == "li" and self._row is not None and self._club is not None:
            self.clubs[self._club].append(self._row)
            self._row = None
        elif tag in ("a", "p"):
            self._capture = "other" if self._row is not None else None

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._in_heading:
            self._heading += f" {text}"
        elif self._row is not None:
            if self._capture == "name":
                self._row["name"] += f" {text}"
            elif self._capture == "problem":
                self._row["problem"] += f" {text}"
            else:
                self._row["other"].append(text)


def parse_page(html: str) -> dict[str, list[Starting11Injury]] | None:
    """{club name: its listed players} out of a league page, or None if the page
    doesn't read like one: no club cards, or a count of players by status that
    disagrees with the totals the page itself states (a layout change that left
    some rows unread)."""
    parser = _PageParser()
    parser.feed(html)
    by_club: dict[str, list[Starting11Injury]] = {}
    for club, rows in parser.clubs.items():
        players = []
        for row in rows:
            name, problem = " ".join(row["name"].split()), " ".join(row["problem"].split())
            status = next((t for t in row["other"] if t in STATUSES), None)
            position = next((t for t in row["other"] if t in _POSITION_BUCKET), None)
            if not name or status is None:
                continue
            players.append(
                Starting11Injury(
                    player_name=name,
                    team_name=club,
                    position_bucket=_POSITION_BUCKET.get(position, "attack"),
                    status=status,
                    chance_of_playing=DOUBTFUL_CHANCE_OF_PLAYING if status == "Doubtful" else 0,
                    problem=problem,
                )
            )
        by_club[club] = players

    listed = [p for players in by_club.values() for p in players]
    if not by_club or not listed:
        return None
    totals = _TOTALS.search(" ".join(re.sub(r"<[^>]+>", " ", html).split()))
    if totals:
        stated = tuple(int(n) for n in totals.groups())
        found = tuple(sum(1 for p in listed if p.status == s) for s in STATUSES)
        if stated != found:
            return None
    # A club left with nobody stays in the result: it is on the page, so it is known to be fit.
    return {
        club: [p for p in players if p.problem.lower() not in _NOT_AN_INJURY] for club, players in by_club.items()
    }


def _cache_file(slug: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha256(URL.format(slug=slug).encode()).hexdigest()}.json"


def _load_cached(slug: str) -> dict[str, list[Starting11Injury]] | None:
    cache_file = _cache_file(slug)
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_TTL_SECONDS:
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            return {club: [Starting11Injury(**p) for p in players] for club, players in raw.items()}
        except (OSError, ValueError, TypeError):
            return None
    return None


def fetch_injuries_by_club(league_code: str) -> dict[str, list[Starting11Injury]] | None:
    """{club name: its listed players} for the clubs the page lists, from the disk
    cache within its lifetime, or None if the league isn't covered or the page
    can't be fetched or read -- which is "unknown", not "nobody is injured", so
    callers leave whatever they have."""
    slug = COMPETITIONS.get(league_code)
    if slug is None:
        return None
    cached = _load_cached(slug)
    if cached is not None:
        return cached
    try:
        response = http_session.get(URL.format(slug=slug), headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        by_club = parse_page(response.text)
    except requests.RequestException:
        return None
    if by_club is None:
        return None
    cache_file = _cache_file(slug)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({club: [asdict(p) for p in players] for club, players in by_club.items()}), encoding="utf-8"
    )
    return by_club


def resolve_team_name(feed_name: str, our_team_names: list[str]) -> str | None:
    """Our name for a Starting 11 club, or None if nothing matches clearly
    (see ingest/club_names.py)."""
    return club_names.resolve_club_name(feed_name, our_team_names, TEAM_NAME_OVERRIDES)
