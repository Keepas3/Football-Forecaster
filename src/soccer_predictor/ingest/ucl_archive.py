"""UEFA Champions League historical results, from openfootball/champions-league
(GitHub, CC0 public domain) -- one plain-text file per season.

football-data.org's free tier serves the Champions League only from 2023-24
on (every earlier season is a 403), so the seasons before that come from this
static archive instead, the same way the World Cup and Euros do (see
ingest/worldcup_archive.py, ingest/euro_archive.py). Downloaded once and cached
permanently: a 2017 result never changes.

These seasons are training data only -- they sit in League.history_seasons, so
they never reach the season picker or the standings (their eight-group format
cannot be shown by the flat league-phase table).

The line format, verified against every season from 2011-12 on:

    21:00  Liverpool FC (ENG)      v Paris Saint-Germain (FRA)  3-2 (2-1)
    21:00  Real Madrid (ESP)       v Bayern München (GER)       4-2 a.e.t. (1-2, 0-0)
    21:00  Manchester City (ENG)   v Real Madrid CF (ESP)       3-4 pen. 1-1 a.e.t. (1-1, 0-1)

The score right after the teams is the final one -- after extra time, and for a
shoot-out the penalty tally comes first. The bracket after it is (score after
90 minutes, half-time score); a match decided in normal time has only the
half-time score there, and some lines have no bracket at all. The model is fit
on goals scored in 90 minutes (extra-time goals come from a different, tired,
all-or-nothing game), so for an extra-time match this reads the FIRST bracket
number -- checked against football-data.org's own `regularTime` scores for the
2023-24 knockout ties.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd
import requests

from soccer_predictor.config import DATA_DIR

BASE_URL = "https://raw.githubusercontent.com/openfootball/champions-league/master"
CACHE_DIR = DATA_DIR / "cache" / "ucl_archive"

# Season code (as in config/leagues.yaml) -> the repository folder. Fixed and
# hand-listed: a season only ever gets added once, and the folders before 2023-24
# are the ones football-data.org will not serve.
SEASON_TO_FOLDER = {
    f"{start % 100:02d}{(start + 1) % 100:02d}": f"{start}-{(start + 1) % 100:02d}" for start in range(2011, 2023)
}
ARCHIVE_SEASONS = frozenset(SEASON_TO_FOLDER)

_MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}  # fmt: skip

# "  Tue Sep 18 2018" (the first date of a file carries the year) or "  Wed Oct 3".
_DATE_RE = re.compile(r"^\s*[A-Z][a-z]{2}\s+(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})(?:\s+(?P<year>\d{4}))?\s*$")

# "<time>  Home (CTY)   v Away (CTY)   <score and brackets>". The three-letter country
# code after each name is what bounds a name (which may itself hold spaces), and the
# time is absent for the second match of a day.
_MATCH_RE = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?"
    r"(?P<home>\S.*?)\s*\([A-Z]{3}\)\s+v\s+"
    r"(?P<away>\S.*?)\s*\([A-Z]{3}\)\s+"
    r"(?P<rest>\d+-\d+.*)$"
)
_SCORE_RE = re.compile(r"(\d+)-(\d+)")
_BRACKET_RE = re.compile(r"\(([^)]*)\)")


def _download_txt(folder: str, force: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{folder}.txt"
    if dest.exists() and not force:
        return dest
    response = requests.get(f"{BASE_URL}/{folder}/cl.txt", timeout=30)
    response.raise_for_status()
    # Raw bytes, so Windows does not turn the source's \r\n into \r\r\n.
    dest.write_bytes(response.content)
    return dest


def fetch_season_text(season: str) -> str:
    folder = SEASON_TO_FOLDER.get(season)
    if folder is None:
        return ""
    return _download_txt(folder).read_text(encoding="utf-8")


def _ninety_minute_score(rest: str) -> tuple[int, int] | None:
    """The 90-minute score from what follows the teams on a match line, or None
    when the line does not say (an extra-time match whose bracket is missing)."""
    if "a.e.t." in rest or "pen." in rest:
        bracket = _BRACKET_RE.search(rest)
        score = _SCORE_RE.search(bracket.group(1)) if bracket else None
    else:
        score = _SCORE_RE.search(rest)
    return (int(score.group(1)), int(score.group(2))) if score else None


def _season_start_year(folder: str) -> int:
    return int(folder[:4])


def _date_in_season(month: int, day: int, start_year: int, explicit_year: int | None) -> dt.date | None:
    # The main draw runs from mid-September, so September-December falls in the
    # season's first calendar year and January-August in the second (the
    # 2019-20 finals were played in August 2020).
    year = explicit_year or (start_year if month >= 9 else start_year + 1)
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def parse_matches(text: str, season: str) -> list[dict]:
    """Canonical match dicts (date, home_team_name, away_team_name, home_goals,
    away_goals) for one season's cl.txt, goals being the 90-minute score. A line
    with no usable score or date is skipped, never an error."""
    folder = SEASON_TO_FOLDER[season]
    start_year = _season_start_year(folder)
    current_date: dt.date | None = None
    matches: list[dict] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith(("#", "=", "▪")):
            continue

        date_match = _DATE_RE.match(line)
        if date_match:
            current_date = _date_in_season(
                _MONTHS[date_match.group("month")],
                int(date_match.group("day")),
                start_year,
                int(date_match.group("year")) if date_match.group("year") else None,
            )
            continue

        match = _MATCH_RE.match(line)
        if match is None or current_date is None:
            continue
        score = _ninety_minute_score(match.group("rest"))
        if score is None:
            continue
        matches.append(
            {
                "date": current_date,
                "home_team_name": match.group("home").strip(),
                "away_team_name": match.group("away").strip(),
                "home_goals": score[0],
                "away_goals": score[1],
            }
        )
    return matches


def matches_frame(season: str) -> pd.DataFrame:
    """One season as the frame historical_csv.ingest_into_db ingests."""
    columns = ["date", "home_team_name", "away_team_name", "home_goals", "away_goals"]
    return pd.DataFrame(parse_matches(fetch_season_text(season), season), columns=columns)
