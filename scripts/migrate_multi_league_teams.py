"""One-time migration: lets a team belong to more than one league.

Before this, `teams.canonical_name` was globally unique -- a club could only
ever be tied to one `league_code` (see the old EPL-only "Manchester City").
That broke adding UEFA Champions League: its clubs already exist under their
domestic league, so they'd silently resolve back to the wrong league-scoped
row instead of getting their own Champions League standings/squad entry.

This rebuilds `teams` and `team_aliases` with the new constraints
(`UNIQUE(canonical_name, league_code)` and `UNIQUE(alias_text, source,
league_code)` respectively) -- SQLite can't ALTER a table's constraints in
place, so both tables are recreated and their data copied across, preserving
every row's original `id` (every other table's `team_id` foreign keys stay
valid). Every existing team keeps its current league_code; nothing is
duplicated by this migration itself -- duplication (e.g. a second
"Manchester City" row under UCL) only happens later, the first time a UCL
sync actually resolves that club.

Backs up data/soccer.db to data/soccer.db.bak-<timestamp> first. Safe to run
multiple times (checks first).

Usage:
    uv run python scripts/migrate_multi_league_teams.py
"""

from __future__ import annotations

import datetime as dt
import shutil
import sys
from pathlib import Path

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.config import DB_PATH  # noqa: E402
from soccer_predictor.storage.db import get_engine, init_db  # noqa: E402
from soccer_predictor.storage.models import Base  # noqa: E402


def _already_migrated(engine) -> bool:
    inspector = inspect(engine)
    alias_columns = {col["name"] for col in inspector.get_columns("team_aliases")}
    if "league_code" not in alias_columns:
        return False
    # Column present isn't proof enough -- confirm the new unique index
    # (constraint name is SQLite-generated, so check by column set instead).
    for uc in inspector.get_unique_constraints("teams"):
        if set(uc["column_names"]) == {"canonical_name", "league_code"}:
            return True
    return False


def main() -> None:
    if not DB_PATH.exists():
        print(f"{DB_PATH} doesn't exist yet -- nothing to migrate")
        return

    init_db()
    engine = get_engine()

    if _already_migrated(engine):
        print("teams/team_aliases already migrated, nothing to do")
        return

    backup_path = DB_PATH.with_name(f"{DB_PATH.name}.bak-{dt.datetime.now():%Y%m%d%H%M%S}")
    shutil.copy2(DB_PATH, backup_path)
    print(f"Backed up {DB_PATH} -> {backup_path}")

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE teams RENAME TO teams_old"))
        conn.execute(text("ALTER TABLE team_aliases RENAME TO team_aliases_old"))
        # SQLite's ALTER TABLE RENAME moves a table's own *named* indexes
        # along with it -- e.g. old "teams" index ix_teams_league_code is
        # now still called that, just attached to teams_old. Left in place,
        # create_all() below would collide creating a same-named index on
        # the new "teams" table (index names are unique database-wide, not
        # per-table). Autoindexes (SQLite's own for a UNIQUE constraint)
        # aren't touched here -- they can't be dropped directly, but they
        # vanish harmlessly when *_old is dropped at the end.
        for index_name in (
            "ix_teams_canonical_name",
            "ix_teams_league_code",
            "ix_team_aliases_alias_text",
        ):
            conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))

    # Recreates every missing table with the current models.py schema --
    # only teams/team_aliases are missing right now (just renamed away), so
    # this creates exactly those two, with the new constraints.
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO teams (id, canonical_name, league_code, crest_url) "
                "SELECT id, canonical_name, league_code, crest_url FROM teams_old"
            )
        )
        conn.execute(
            text(
                "INSERT INTO team_aliases (id, alias_text, source, team_id, league_code) "
                "SELECT a.id, a.alias_text, a.source, a.team_id, t.league_code "
                "FROM team_aliases_old a JOIN teams_old t ON a.team_id = t.id"
            )
        )
        team_count = conn.execute(text("SELECT COUNT(*) FROM teams")).scalar_one()
        alias_count = conn.execute(text("SELECT COUNT(*) FROM team_aliases")).scalar_one()
        conn.execute(text("DROP TABLE team_aliases_old"))
        conn.execute(text("DROP TABLE teams_old"))

    print(f"Migrated {team_count} teams and {alias_count} aliases to the new schema")


if __name__ == "__main__":
    main()
