"""One-time migration: adds Injury.expected_return_date to an existing DB.

storage/db.py::init_db() only runs Base.metadata.create_all(), which never
alters existing tables -- the new team_form_notes table is created fine by
that alone, but the new column on the existing injuries table needs this
explicit ALTER TABLE. Safe to run multiple times (checks first).

Usage:
    uv run python scripts/migrate_chat_notes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.storage.db import get_engine, init_db  # noqa: E402


def main() -> None:
    init_db()  # creates team_form_notes (new table) and any fresh DB from scratch
    engine = get_engine()

    columns = {col["name"] for col in inspect(engine).get_columns("injuries")}
    if "expected_return_date" in columns:
        print("injuries.expected_return_date already present, nothing to do")
        return

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE injuries ADD COLUMN expected_return_date DATE"))
    print("Added injuries.expected_return_date")


if __name__ == "__main__":
    main()
