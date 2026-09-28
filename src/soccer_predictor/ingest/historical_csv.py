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
from soccer_predictor.storage.repository import upsert_match

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
        )
        ingested += 1
    return ingested, skipped
