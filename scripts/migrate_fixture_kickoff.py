"""One-time migration: adds Fixture.kickoff_utc to an existing DB.

storage/db.py::init_db() only runs Base.metadata.create_all(), which never
alters existing tables -- this explicit ALTER TABLE is required for a
database created before this column existed. Safe to run multiple times
(checks first). Existing fixture rows just have NULL kickoff_utc until the
next `refresh_live_data.py` run re-syncs them with a real time.

Usage:
    uv run python scripts/migrate_fixture_kickoff.py
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

    columns = {col["name"] for col in inspect(engine).get_columns("fixtures")}
    if "kickoff_utc" in columns:
        print("fixtures.kickoff_utc already present, nothing to do")
        return

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE fixtures ADD COLUMN kickoff_utc DATETIME"))
    print("Added fixtures.kickoff_utc")


if __name__ == "__main__":
    main()
