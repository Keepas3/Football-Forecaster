"""European Championship historical match + best-effort goal-scorer archive,
from openfootball/euro (GitHub, CC0 public domain) -- plain-text
"football.txt" files, one per tournament, hand-maintained since the 1960s.

Downloaded once and cached permanently, same reasoning as
ingest/worldcup_archive.py: a 1960 result never changes.

Confirmed live (2026-09): match RESULTS are reliably present for every
match, every tournament, back to 1960 -- but the exact line format is NOT
consistent across tournaments (three real variants found, all handled
below):
  - 1960/1964/1968: "TeamA  v  TeamB  N-N  @ Venue" (score AFTER both names)
  - 1972-2024 (the vast majority): "TeamA  N-N  TeamB" (score BETWEEN
    names), with a date either sharing the match's own line or on its own
    line just above.
Per-player goal-scorer detail is genuinely inconsistent (confirmed live:
the full 1996 file only had scorer detail for its Final, not the other 30
matches) -- parse_goalscorers is deliberately best-effort and returns []
for anything it can't confidently parse, never raising.

SEASON_TO_FOLDER's "2020" entry is UEFA Euro 2020 -- delayed a year by
COVID-19, so its real matches were played in 2021 and its folder is named
"2021--europe"; the season CODE here stays "2020" to match how everyone
(UEFA included) actually refers to that tournament.
"""

from __future__ import annotations

import re
from pathlib import Path

import requests

from soccer_predictor.config import DATA_DIR

BASE_URL = "https://raw.githubusercontent.com/openfootball/euro/master"
CACHE_DIR = DATA_DIR / "cache" / "euro_archive"

# Real folder names, confirmed live (2026-09) via the GitHub contents API --
# a fixed, hand-maintained map rather than a live per-run directory listing,
# since a new one of these is only ever added once every 4 years (right
# alongside the next config/leagues.yaml edit that adds its season code).
SEASON_TO_FOLDER = {
    "1960": "1960--france",
    "1964": "1964--spain",
    "1968": "1968--italy",
    "1972": "1972--belgium",
    "1976": "1976--yugoslavia",
    "1980": "1980--italy",
    "1984": "1984--france",
    "1988": "1988--west-germany",
    "1992": "1992--sweden",
    "1996": "1996--england",
    "2000": "2000--belgium-netherlands",
    "2004": "2004--portugal",
    "2008": "2008--austria-switzerland",
    "2012": "2012--poland-ukraine",
    "2016": "2016--france",
    "2020": "2021--europe",  # see module docstring
    "2024": "2024--germany",
}

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}  # fmt: skip

# A line that OPENS with a recognized month name sets the "current date" for
# any match line(s) that follow, whether or not that same line also carries
# match content (both real layouts occur -- see module docstring).
_DATE_RE = re.compile(
    r"^(?:[A-Za-z]{3}\s+)?(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})\b"
)

_SCORE_TOKEN_RE = re.compile(r"(?P<hg>\d{1,2})\s*[-–]\s*(?P<ag>\d{1,2})")

# The older (1960/1964/1968) "TeamA v TeamB N-N" layout -- score comes AFTER
# both team names, joined by a bare " v ".
_V_FORMAT_RE = re.compile(
    r"(?P<home>[A-Za-z][A-Za-z.'À-ſ]*(?:\s[A-Za-z][A-Za-z.'À-ſ]*)*)"
    r"\s+v\.?\s+"
    r"(?P<away>[A-Za-z][A-Za-z.'À-ſ]*(?:\s[A-Za-z][A-Za-z.'À-ſ]*)*?)"
    r"\s+(?P<hg>\d{1,2})\s*[-–]\s*(?P<ag>\d{1,2})\b"
)


def list_tournament_folders() -> list[str]:
    return sorted(SEASON_TO_FOLDER.values())


def _download_txt(folder: str, force: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{folder}.txt"
    if dest.exists() and not force:
        return dest
    response = requests.get(f"{BASE_URL}/{folder}/euro.txt", timeout=30)
    response.raise_for_status()
    # write_bytes (not write_text(response.text)) -- the source's raw bytes
    # already use \r\n; round-tripping through write_text's own newline
    # translation on Windows double-translates \r\n into \r\r\n, which then
    # reads back as a spurious extra blank line before every real line.
    dest.write_bytes(response.content)
    return dest


def fetch_tournament_text(season: str) -> str:
    folder = SEASON_TO_FOLDER.get(season)
    if folder is None:
        return ""
    return _download_txt(folder).read_text(encoding="utf-8")


# A kickoff time optionally followed by a "(local time in another
# timezone)" annotation -- e.g. "20:45 (21:45 EEST) Germany ..." (2012) --
# confirmed live this can be separated from the team name by only a single
# space, not the usual 2+-space alignment gap, so it has to be stripped
# explicitly rather than relying on the 2+-space split below to isolate it.
_LEADING_TIME_TZ_RE = re.compile(r"^\d{1,2}:\d{2}\s*(?:\([^)]*\)\s*)?")


def _strip_alignment_padding(text: str) -> str:
    """Real team names in this source are always single-space-separated
    ("West Germany", "Czech Republic") -- only the source's own column
    alignment ever uses a run of 2+ spaces. So splitting on any 2+-space run
    and keeping the LAST piece isolates the actual name even when a date/
    time prefix ("21:00   France") shares the same line.
    """
    text = _LEADING_TIME_TZ_RE.sub("", text.strip())
    pieces = re.split(r"\s{2,}", text.strip())
    return pieces[-1].strip() if pieces else ""


def _looks_like_a_team_name(name: str) -> bool:
    """Rejects a "team name" extracted from a line that turns out NOT to be
    a real match, or an in-line annotation sitting between the score and
    the real away team -- e.g. a penalty-shootout breakdown, a kickoff-
    time/timezone annotation, or a bare "a.e.t." (confirmed live on real
    tournament files: "Penalties: 0-1 Causio, 1-1 Masny,",
    "Collovati (saved), 9-8 Barmos", and "Czechoslovakia 3-1 a.e.t.
    Netherlands" all score-token-match, or sit next to one, without being
    real matches/team names). A real team name always starts with an
    uppercase letter and never contains a digit, colon, comma, or
    parenthesis -- "a.e.t." fails on the first rule, the other two fail on
    the second.
    """
    return bool(name) and name[0].isupper() and not re.search(r"[\d:,()]", name)


def _clean_away_side(text: str) -> str | None:
    # A trailing score annotation, e.g. "(3-0)" (half-time), or an inline
    # one with no parens at all, e.g. "a.e.t." or "a.e.t., 5-3 pen.", can
    # sit between the score and the real away team's name -- skip any
    # leading alignment-delimited piece that doesn't look like a real team
    # name (rather than assuming the very first piece always is one).
    text = re.sub(r"^(\([^)]*\)\s*)+", "", text.strip())
    for piece in re.split(r"\s{2,}|@", text):
        piece = piece.strip()
        if _looks_like_a_team_name(piece):
            return piece
    return None


def _parse_match_line(line: str) -> tuple[str, int, int, str] | None:
    """Returns (home, home_goals, away_goals, away) for one line that
    contains a match, or None if this line has no parseable match on it
    (a pure date/venue header, a blank line, a lineup/goal-scorer line,
    etc. -- all expected and silently skipped, never an error)."""
    v_match = _V_FORMAT_RE.search(line)
    if v_match:
        home = _strip_alignment_padding(v_match.group("home"))
        away = v_match.group("away").strip()
        if _looks_like_a_team_name(home) and _looks_like_a_team_name(away):
            return home, int(v_match.group("hg")), int(v_match.group("ag")), away

    score = _SCORE_TOKEN_RE.search(line)
    if score is None:
        return None
    home = _strip_alignment_padding(line[: score.start()])
    away = _clean_away_side(line[score.end() :])
    if not away or not _looks_like_a_team_name(home) or not _looks_like_a_team_name(away):
        return None
    return home, int(score.group("hg")), int(score.group("ag")), away


def _parse_date_line(line: str, year: int) -> "dt.date | None":
    import datetime as dt

    match = _DATE_RE.match(line.strip())
    if match is None:
        return None
    month = _MONTHS.get(match.group("month").lower())
    if month is None:
        return None
    try:
        return dt.date(year, month, int(match.group("day")))
    except ValueError:
        return None


def parse_matches(text: str, year: int) -> list[dict]:
    """Canonical match dicts (date, home_team_name, away_team_name,
    home_goals, away_goals) for one tournament's raw euro.txt. `year` seeds
    a fallback date (the tournament's own year, month/day unknown) for the
    handful of tournaments whose file never states a per-match date at all
    (e.g. 1972's file only labels stages, not dates on some lines) -- close
    enough for Dixon-Coles time-weighting, which only cares about
    approximate recency, not exact kickoff day.
    """
    import datetime as dt

    fallback_date = dt.date(year, 7, 1)
    current_date = fallback_date
    matches: list[dict] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        parsed_date = _parse_date_line(line, year)
        if parsed_date is not None:
            current_date = parsed_date

        parsed = _parse_match_line(line)
        if parsed is None:
            continue
        home, home_goals, away_goals, away = parsed
        matches.append(
            {
                "date": current_date,
                "home_team_name": home,
                "away_team_name": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )

    return matches


# A name is optional before each minute -- when absent, the goal belongs to
# whichever player's name was last seen (this source is inconsistent about
# whether a repeat scorer's next goal is comma-separated, e.g. "G.Müller 24,
# 71", or just space-separated like the next player over, e.g. the 1960s
# files' "Vincent 12' Heutte 43'" -- so punctuation can't be the delimiter;
# only "was there a name here or not" can).
_GOAL_TOKEN_RE = re.compile(
    r"(?:(?P<name>[A-Za-zÀ-ſ][A-Za-zÀ-ſ.'\-]*(?:\s[A-Za-zÀ-ſ][A-Za-zÀ-ſ.'\-]*)*)\s+)?"
    r"(?P<minute>\d{1,3}(?:\+\d{1,2})?)"
    r"(?P<flags>\s*\(?[a-z]*\)?)"
)


def _parse_scorer_side(text: str, team_name: str) -> list[dict]:
    """Splits one side ("home" or "away") of a goal-scorer parenthetical --
    e.g. "G.Müller 24, 71" (one player, two goals), "Giorgos Karagounis 7,
    Angelos Basinas 51pen" (two players), or "Vincent 12' Heutte 43'" (two
    players, no comma at all) -- into individual goal dicts.
    """
    # Apostrophes are purely a minute-marker convention here ("73'") --
    # stripped up front so they never have to be threaded through the
    # regex below.
    text = text.replace("'", " ")
    goals = []
    last_name: str | None = None
    for match in _GOAL_TOKEN_RE.finditer(text):
        name = match.group("name")
        if name:
            last_name = name.strip()
        if last_name is None:
            continue
        flags = match.group("flags").lower()
        goals.append(
            {
                "team_name": team_name,
                "player_name": last_name,
                "minute": int(match.group("minute").split("+")[0]),
                "own_goal": "og" in flags,
                "penalty": "pen" in flags,
            }
        )
    return goals


def parse_goalscorers(text: str, year: int) -> list[dict]:
    """Best-effort per-goal dicts (team_name, player_name, minute, own_goal,
    penalty) -- team_name is whichever of the match's two sides this goal
    block segment belongs to (split on the first top-level ";"). Returns []
    for any match with no scorer block at all, which is expected and common
    for this source (see module docstring) -- never raises.

    match_date isn't produced here (unlike worldcup_archive.parse_goals):
    this walks the goal-scorer block independently of parse_matches's own
    per-line date tracking, and re-deriving it wouldn't be worth the extra
    coupling for a best-effort secondary source.
    """
    lines = text.splitlines()
    results: list[dict] = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        parsed = _parse_match_line(line) if line.strip() else None
        i += 1
        if parsed is None:
            continue
        home, _, _, away = parsed

        block_lines = []
        while i < len(lines):
            candidate = lines[i]
            stripped = candidate.strip()
            if not stripped:
                break
            if not candidate.startswith((" ", "\t")):
                break
            if _parse_match_line(candidate) is not None:
                break
            block_lines.append(stripped)
            i += 1
            if stripped.endswith(")"):
                break

        block = " ".join(block_lines).strip()
        if not (block.startswith("(") and ")" in block):
            continue
        block = block[1 : block.rindex(")")]

        sides = block.split(";", 1)
        home_side = sides[0]
        away_side = sides[1] if len(sides) > 1 else ""
        results.extend(_parse_scorer_side(home_side, home))
        results.extend(_parse_scorer_side(away_side, away))

    return results
