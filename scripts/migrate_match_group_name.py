"""One-time migration: adds Match.group_name and Fixture.group_name to an
existing DB.

storage/db.py::init_db() only runs Base.metadata.create_all(), which never
alters existing tables -- but it also calls ensure_group_name_columns(), so
this script is mostly a visible, explicit way to do the same thing (and
report what changed), matching the other scripts/migrate_*.py. Safe to run
multiple times (checks first).

Usage:
    uv run python scripts/migrate_match_group_name.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import inspect

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.storage.db import (  # noqa: E402
    ensure_group_name_columns,
    get_engine,
    init_db,
)


def main() -> None:
    engine = get_engine()
    # Check BEFORE init_db(), which would add the columns itself and leave
    # nothing for this script to report.
    existing = {
        table: "group_name" in {col["name"] for col in inspect(engine).get_columns(table)}
        for table in ("matches", "fixtures")
        if inspect(engine).has_table(table)
    }
    init_db()
    altered = ensure_group_name_columns(engine)
    for table, had_column in existing.items():
        print(f"{table}.group_name already present, nothing to do" if had_column else f"Added {table}.group_name")
    if not existing and not altered:
        print("Fresh database: init_db() created the tables with group_name already included")


if __name__ == "__main__":
    main()
