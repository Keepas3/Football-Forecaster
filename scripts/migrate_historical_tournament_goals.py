"""One-time migration: adds the historical_tournament_goals table to an
existing DB (World Cup/Euro archive goal-scorer data -- see
ingest/worldcup_archive.py, ingest/euro_archive.py).

storage/db.py::init_db() already calls Base.metadata.create_all(), which
DOES create any missing table from scratch -- this script exists purely for
visibility/consistency with this project's other scripts/migrate_*.py
scripts (which are only strictly required for an ALTER TABLE on an
already-existing table). Safe to run multiple times.

Usage:
    uv run python scripts/migrate_historical_tournament_goals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import inspect

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.storage.db import get_engine, init_db  # noqa: E402


def main() -> None:
    init_db()
    engine = get_engine()

    if "historical_tournament_goals" in inspect(engine).get_table_names():
        print("historical_tournament_goals already present, nothing to do")
    else:
        print("historical_tournament_goals was missing -- init_db() just created it")


if __name__ == "__main__":
    main()
