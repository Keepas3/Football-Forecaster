"""Refreshes upcoming fixtures (and, best-effort, injuries) from live APIs.

Usage:
    uv run python scripts/refresh_live_data.py [LEAGUE_CODE ...]

Requires FOOTBALL_DATA_ORG_API_KEY in .env for fixtures. API_FOOTBALL_KEY
is optional; without it, injuries fall back to config/injuries.yaml only.

This is a thin printing loop over ingest.refresh.refresh_league -- the
actual per-league orchestration is shared with dashboard/views/admin.py's
in-app refresh, so the two never drift into different behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.config import load_leagues  # noqa: E402
from soccer_predictor.ingest.refresh import refresh_league  # noqa: E402
from soccer_predictor.storage.db import init_db  # noqa: E402


def main() -> None:
    init_db()
    leagues = load_leagues()
    requested = sys.argv[1:] or list(leagues.keys())

    for code in requested:
        league = leagues[code]
        print(f"League {league.name} ({league.code})")

        result = refresh_league(league)

        if result.fixtures_error:
            print(f"  fixtures: skipped -- {result.fixtures_error}")
        else:
            print(f"  fixtures: {result.fixtures_synced} synced, {result.fixtures_skipped} skipped (unresolved teams)")

        if result.results_error:
            print(f"  results: skipped -- {result.results_error}")
        elif result.results_synced or result.results_skipped:
            print(f"  results: {result.results_synced} synced, {result.results_skipped} skipped (unresolved teams)")

        print(f"  prediction tracking: {result.snapshots_created} new snapshot(s) locked in")

        if result.injuries_error:
            print(f"  injuries: skipped -- {result.injuries_error}")
        else:
            print(f"  injuries: {result.injuries_synced} teams synced, {result.injuries_skipped} skipped")


if __name__ == "__main__":
    main()
