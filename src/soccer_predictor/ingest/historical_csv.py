"""Parses football-data.co.uk season CSVs into the canonical Match schema.

CSV columns of interest: Date, HomeTeam, AwayTeam, FTHG (full-time home
goals), FTAG (full-time away goals). Date format varies by season
("dd/mm/yy" pre-2019-ish, "dd/mm/yyyy" more recently) so both are tried.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from soccer_predictor.ingest.team_mapper import UnresolvedTeamName, resolve
from soccer_predictor.storage.repository import replace_historical_tournament_goals, upsert_match

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def parse_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}")

    df = df[REQUIRED_COLUMNS].dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.date
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df.rename(
        columns={
            "Date": "date",
            "HomeTeam": "home_team_name",
            "AwayTeam": "away_team_name",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
        }
    )


def _optional_group_name(row) -> str | None:
    """`group_name` only exists on the WC/Euro archive frames (see
    worldcup_archive.parse_matches); football-data.co.uk CSVs never have
    one, and a knockout match's is None/NaN."""
    group = getattr(row, "group_name", None)
    return group if isinstance(group, str) and group else None


def ingest_into_db(
    session: Session, league_code: str, season: str, df: pd.DataFrame
) -> tuple[int, int]:
    """Resolves team names and upserts matches. Returns (ingested, skipped)."""
    ingested = 0
    skipped = 0
    for row in df.itertuples(index=False):
        try:
            home_team_id = resolve(session, row.home_team_name, source="csv", league_code=league_code)
            away_team_id = resolve(session, row.away_team_name, source="csv", league_code=league_code)
        except UnresolvedTeamName:
            skipped += 1
            continue
        upsert_match(
            session,
            league_code=league_code,
            season=season,
            date=row.date,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=row.home_goals,
            away_goals=row.away_goals,
            group_name=_optional_group_name(row),
        )
        ingested += 1
    return ingested, skipped


def ingest_goals_into_db(
    session: Session, league_code: str, season: str, goal_rows: list[dict], source: str
) -> tuple[int, int]:
    """Resolves each goal's scoring player's team name and replaces
    (league_code, season)'s stored HistoricalTournamentGoal rows. Shared by
    ingest/worldcup_archive.py and ingest/euro_archive.py -- both produce
    the same {team_name, player_name, minute, match_date, own_goal, penalty}
    dict shape, keyed the same way `df` rows are for ingest_into_db above
    (team_mapper.resolve(source="csv", ...), so a team must already have
    been seeded via team_mapper.seed_teams_from_names -- same requirement
    as ingest_into_db). Returns (ingested, skipped).
    """
    resolved = []
    skipped = 0
    for row in goal_rows:
        try:
            team_id = resolve(session, row["team_name"], source="csv", league_code=league_code)
        except UnresolvedTeamName:
            skipped += 1
            continue
        resolved.append({**row, "team_id": team_id})
    replace_historical_tournament_goals(session, league_code, season, resolved, source)
    return len(resolved), skipped
