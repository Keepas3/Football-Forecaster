"""One-time migration: adds Team.api_football_team_id to an existing DB.

storage/db.py::init_db() only runs Base.metadata.create_all(), which never
alters existing tables -- this explicit ALTER TABLE is required for a
database created before this column existed. Safe to run multiple times
(checks first).

Usage:
    uv run python scripts/migrate_api_football_team_id.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.storage.db import get_engine, init_db  # noqa: E402


def main() -> None:
    init_db()
    engine = get_engine()

    columns = {col["name"] for col in inspect(engine).get_columns("teams")}
    if "api_football_team_id" in columns:
        print("teams.api_football_team_id already present, nothing to do")
        return

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE teams ADD COLUMN api_football_team_id INTEGER"))
    print("Added teams.api_football_team_id")


if __name__ == "__main__":
    main()
