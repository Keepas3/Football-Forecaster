"""Prints a short fingerprint of the data that matters for results and the
Track Record: how many matches are stored (and a checksum of their scores)
and how many predictions are locked in.

The frequent GitHub Actions refresh compares the fingerprint before and
after it runs and only commits data/soccer.db when it changed. Comparing
the file itself doesn't work: SQLite rewrites it (fixture `fetched_at`
timestamps, etc.) on every run, which would mean a ~4.5 MB commit -- and a
Streamlit redeploy -- every couple of hours even when nothing happened.

Uses only the standard library, so it runs before the project's
dependencies are installed.

Usage:
    python scripts/data_fingerprint.py [path/to/soccer.db]
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "soccer.db"


def fingerprint(db_path: Path) -> str:
    connection = sqlite3.connect(db_path)
    try:
        match_count, score_checksum = connection.execute(
            # The checksum catches a corrected score, not just a new match.
            "SELECT COUNT(*), COALESCE(SUM(home_goals * 31 + away_goals), 0) FROM matches"
        ).fetchone()
        prediction_count = connection.execute("SELECT COUNT(*) FROM prediction_records").fetchone()[0]
        # A lineup lock can replace an older lock in place, which leaves the count
        # unchanged: also count the rows locked from a lineup / early forecast.
        # (A database from before that column existed has none.)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(prediction_records)")}
        lineup_locks = (
            connection.execute("SELECT COUNT(*) FROM prediction_records WHERE basis IS NOT NULL").fetchone()[0]
            if "basis" in columns
            else 0
        )
    finally:
        connection.close()
    return f"matches={match_count}:{score_checksum} predictions={prediction_count} lineup_locks={lineup_locks}"


if __name__ == "__main__":
    print(fingerprint(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB))
