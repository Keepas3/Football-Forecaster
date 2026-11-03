"""Updates the committed copy of Understat's current-season player data.

Usage:
    uv run python scripts/save_understat_snapshot.py

For each league Understat covers, asks for its current season (served from the disk
cache when the refresh that just ran already fetched it) and, if Understat answered,
rewrites data/understat_snapshot/<league>.json when the numbers changed. The file is
what rates an announced lineup, and feeds the Players and injury pages, when Understat
is down and there is no disk cache -- on a GitHub Actions runner, or on a freshly
rebooted Streamlit app (see ingest.understat_client.save_snapshot).

Only the daily refresh-data workflow runs this and commits the result; nothing else
edits those files. Safe to run by hand, and a no-op while Understat is down.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.config import load_leagues  # noqa: E402
from soccer_predictor.ingest import understat_client  # noqa: E402


def main() -> None:
    changed = 0
    for code, league in load_leagues().items():
        if code not in understat_client.UNDERSTAT_LEAGUE_SLUG:
            continue
        season = league.api_season_year(league.seasons[-1])
        updated = understat_client.save_snapshot(code, season)
        changed += updated
        print(f"{league.name} ({code}) {season}: {'snapshot updated' if updated else 'snapshot unchanged'}")
    print(f"{changed} snapshot file(s) changed")


if __name__ == "__main__":
    main()
