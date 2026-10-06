"""Refreshes upcoming fixtures (and, best-effort, injuries) from live APIs.

Usage:
    uv run python scripts/refresh_live_data.py [--quick] [LEAGUE_CODE ...]

--quick is the light refresh the frequent GitHub Actions run uses: recent results
and a short fixture window, no injury sync (see ingest.refresh.refresh_league).

Requires FOOTBALL_DATA_ORG_API_KEY in .env for fixtures. Automatic
injuries only exist for MLS (via ESPN); every other league relies on
config/injuries.yaml and the dashboard's chat notes tab instead.

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
    quick = "--quick" in sys.argv[1:]
    requested = [a for a in sys.argv[1:] if a != "--quick"] or list(leagues.keys())

    for code in requested:
        league = leagues[code]
        print(f"League {league.name} ({league.code})")

        result = refresh_league(league, quick=quick)

        if result.fixtures_error:
            print(f"  fixtures: skipped -- {result.fixtures_error}")
        else:
            print(f"  fixtures: {result.fixtures_synced} synced, {result.fixtures_skipped} skipped (unresolved teams)")

        if result.results_error:
            print(f"  results: skipped -- {result.results_error}")
        elif result.results_synced or result.results_skipped:
            print(f"  results: {result.results_synced} synced, {result.results_skipped} skipped (unresolved teams)")

        print(
            f"  prediction tracking: {result.snapshots_created} new snapshot(s) locked in, "
            f"{result.probabilities_backfilled} given probabilities"
        )

        if quick:
            continue  # injuries aren't synced in a quick refresh

        if result.injuries_error:
            print(f"  injuries: skipped -- {result.injuries_error}")
        else:
            print(f"  injuries: {result.injuries_synced} teams synced, {result.injuries_skipped} skipped")


if __name__ == "__main__":
    main()
