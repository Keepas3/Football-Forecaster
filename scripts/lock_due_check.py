"""Is any match about to start with no final forecast locked in yet?

The lock-lineups workflow is pinged every ten minutes, around the clock, so its
first step has to be nearly free: this answers from the committed database with
the standard library only (before `uv sync`), and the workflow does the rest only
when the answer is yes.

A deliberate over-estimate of what prediction.tracking.lock_due_predictions will
lock: it looks a little further ahead than that function does and does not tell a
football-data.org placeholder kickoff (00:00 UTC) from a real one, so now and then
it says yes when nothing ends up locked. It never says no when a lock is due.

Usage:
    python scripts/lock_due_check.py [path/to/soccer.db]

Prints `due=true` or `due=false`, and appends the same line to $GITHUB_OUTPUT when
that is set (as it is in GitHub Actions).
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "soccer.db"
# Keep at or above prediction.tracking.LINEUP_WINDOW_MINUTES (a test checks).
CHECK_WINDOW_MINUTES = 95


def is_due(db_path: Path, now: dt.datetime | None = None) -> bool:
    now = now or dt.datetime.now(dt.UTC).replace(tzinfo=None)
    end = now + dt.timedelta(minutes=CHECK_WINDOW_MINUTES)
    connection = sqlite3.connect(db_path)
    try:
        # A database from before the `basis` column existed has no lineup/early
        # locks at all, so nothing there counts as locked yet.
        columns = {row[1] for row in connection.execute("PRAGMA table_info(prediction_records)")}
        locked = (
            "EXISTS (SELECT 1 FROM prediction_records p WHERE p.league_code = f.league_code "
            "AND p.date = f.date AND p.home_team_id = f.home_team_id "
            "AND p.away_team_id = f.away_team_id AND p.basis IS NOT NULL)"
            if "basis" in columns
            else "0"
        )
        (waiting,) = connection.execute(
            "SELECT COUNT(*) FROM fixtures f WHERE f.kickoff_utc IS NOT NULL "
            "AND datetime(f.kickoff_utc) > datetime(?) AND datetime(f.kickoff_utc) <= datetime(?) "
            f"AND NOT {locked}",
            (now.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")),
        ).fetchone()
    finally:
        connection.close()
    return waiting > 0


def main() -> None:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    line = f"due={'true' if is_due(db_path) else 'false'}"
    print(line)
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


if __name__ == "__main__":
    main()
