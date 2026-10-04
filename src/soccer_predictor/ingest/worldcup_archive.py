"""World Cup historical match + goal-scorer archive, from jfjelstul/worldcup
(GitHub, CC-BY-SA 4.0) -- a static, professionally structured CSV database
of official FIFA World Cup results. Downloaded once and cached permanently
(unlike Understat/ASA's 24h-TTL live-season cache): a 1930 result never
changes, so there's nothing to keep re-fetching.

Confirmed live (2026-09): matches.csv has real results back to 1930,
goals.csv has real scorer name + minute back to 1930 (e.g. Lucien Laurent,
minute 19, France 4-1 Mexico). No assists anywhere in this source -- a real
historical-recordkeeping gap (assists were rarely officially tracked before
the 2000s), not a bug.

The dataset also includes the FIFA WOMEN'S World Cup under the same
tournament_id numbering scheme (odd-numbered years from 1991 on, e.g.
WC-1991, WC-1999, WC-2019) -- filtered out here (see _is_mens), since this
app's WC league entry is the men's tournament only, matching
football-data.org's own competition scope for the live/upcoming editions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from soccer_predictor.config import DATA_DIR

BASE_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"
CACHE_DIR = DATA_DIR / "cache" / "worldcup_archive"

# Every real men's World Cup year this archive covers, confirmed live
# (2026-09) -- no tournament in 1942/1946 (World War II). Used by
# scripts/fetch_historical_data.py to tell an archive-covered season apart
# from the current/next tournament's season code (e.g. "2026"), which has
# no archive data yet and still needs football-data.org's live-fixture path.
WORLDCUP_AVAILABLE_SEASONS = frozenset(
    {
        "1930", "1934", "1938", "1950", "1954", "1958", "1962", "1966", "1970", "1974",
        "1978", "1982", "1986", "1990", "1994", "1998", "2002", "2006", "2010", "2014",
        "2018", "2022",
    }
)  # fmt: skip


def _download_csv(name: str, force: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{name}.csv"
    if dest.exists() and not force:
        return dest
    response = requests.get(f"{BASE_URL}/{name}.csv", timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def fetch_matches() -> pd.DataFrame:
    return pd.read_csv(_download_csv("matches"))


def fetch_goals() -> pd.DataFrame:
    return pd.read_csv(_download_csv("goals"))


def _is_mens(df: pd.DataFrame) -> pd.Series:
    return df["tournament_name"].str.contains("Men's", na=False)


def _season_from_tournament_id(tournament_id: str) -> str:
    # "WC-1930" -> "1930"
    return tournament_id.split("-")[-1]


def _group_label(stage_name: str, group_name: str, group_stage: int) -> str | None:
    """The table a match belongs to, or None for a knockout match (which
    must stay out of every group table). A second round-robin stage (1974,
    1978, 1982) reuses the first stage's group names -- "Group A" -- so it's
    qualified with its stage ("Second Group Stage - Group A") to keep the
    two stages' tables from merging; 1950's decisive "final round" has no
    group name at all and becomes a single table of its own.
    """
    if not group_stage:
        return None
    has_group = bool(group_name) and group_name != "not applicable"
    if stage_name == "group stage":
        return group_name if has_group else None
    stage_label = stage_name.title()
    return f"{stage_label} - {group_name}" if has_group else stage_label


def parse_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical (date, home_team_name, away_team_name, home_goals,
    away_goals, season, group_name) rows, in the same shape
    ingest.historical_csv.ingest_into_db already expects -- reused as-is for
    match persistence (see scripts/fetch_historical_data.py). Scores are
    whatever the source lists as the final result, including extra time
    when played (e.g. Argentina 3-3 France, 2022 final) -- penalty
    shootouts are never reflected in the goal columns, matching how this
    app has always treated a scoreline for Dixon-Coles training.
    `group_name` is None for knockout matches (see _group_label).
    """
    mens = df[_is_mens(df)].copy()
    mens["season"] = mens["tournament_id"].map(_season_from_tournament_id)
    if {"stage_name", "group_name", "group_stage"} <= set(mens.columns):
        mens["group_label"] = [
            _group_label(stage, group, group_stage)
            for stage, group, group_stage in zip(mens["stage_name"], mens["group_name"], mens["group_stage"])
        ]
    else:
        mens["group_label"] = None
    out = mens[
        [
            "match_date",
            "home_team_name",
            "away_team_name",
            "home_team_score",
            "away_team_score",
            "season",
            "group_label",
        ]
    ].rename(
        columns={
            "match_date": "date",
            "home_team_score": "home_goals",
            "away_team_score": "away_goals",
            "group_label": "group_name",
        }
    )
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["group_name"] = out["group_name"].astype(object).where(out["group_name"].notna(), None)
    return out


def _goal_scorer_name(given_name: str, family_name: str) -> str:
    # A handful of very old (pre-1950) entries have no recorded given name
    # -- the source spells this literal string "not applicable" rather than
    # leaving the field blank.
    if not given_name or given_name == "not applicable":
        return family_name
    return f"{given_name} {family_name}"


def parse_goals(df: pd.DataFrame) -> list[dict]:
    """One dict per goal, keyed by the SCORING PLAYER's own team
    (player_team_name) rather than the credited team (team_name) -- these
    differ for an own goal, and a player's goal (even an own goal) belongs
    on their own team's roster page, not their opponent's. Matches
    ingest.historical_csv.ingest_goals_into_db's expected shape.
    """
    mens = df[_is_mens(df)].copy()
    mens["season"] = mens["tournament_id"].map(_season_from_tournament_id)

    goals = []
    for row in mens.itertuples(index=False):
        minute = row.minute_regulation
        match_date = row.match_date
        goals.append(
            {
                "season": row.season,
                "team_name": row.player_team_name,
                "player_name": _goal_scorer_name(row.given_name, row.family_name),
                "minute": int(minute) if pd.notna(minute) else None,
                "match_date": pd.to_datetime(match_date).date() if pd.notna(match_date) else None,
                "own_goal": bool(row.own_goal),
                "penalty": bool(row.penalty),
            }
        )
    return goals
