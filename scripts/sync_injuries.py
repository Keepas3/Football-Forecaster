"""Pulls just the automatic injuries into the local database, without the fixtures,
results and prediction snapshots a full refresh also touches.

Usage:
    uv run python scripts/sync_injuries.py [LEAGUE_CODE ...]

With no league codes it does every league that has an automatic source (see
ingest.injuries.has_automatic_feed). Handy for trying a feed locally between the
scheduled refreshes: run it, then reload the dashboard's Injuries tab.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.config import load_leagues  # noqa: E402
from soccer_predictor.ingest import injuries  # noqa: E402
from soccer_predictor.storage.db import init_db, session_scope  # noqa: E402


def main() -> None:
    init_db()
    leagues = load_leagues()
    requested = sys.argv[1:] or [code for code, league in leagues.items() if injuries.has_automatic_feed(league)]

    for code in requested:
        league = leagues[code]
        try:
            with session_scope() as session:
                synced, skipped = injuries.sync_injuries_to_db(session, league, season_year=dt.date.today().year)
        except requests.RequestException as exc:
            print(f"{league.name} ({code}): skipped -- {exc}")
            continue
        if (synced, skipped) == (0, 0):
            print(f"{league.name} ({code}): nothing changed (no automatic source, or it could not be reached)")
        else:
            print(f"{league.name} ({code}): {synced} teams synced, {skipped} skipped")


if __name__ == "__main__":
    main()
