"""Refreshes upcoming fixtures (and, best-effort, injuries) from live APIs.

Usage:
    uv run python scripts/refresh_live_data.py [LEAGUE_CODE ...]

Requires FOOTBALL_DATA_ORG_API_KEY in .env for fixtures. API_FOOTBALL_KEY
is optional; without it, injuries fall back to config/injuries.yaml only.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.config import load_leagues  # noqa: E402
from soccer_predictor.ingest import injuries as injuries_module  # noqa: E402
from soccer_predictor.ingest.api_client import MissingApiKey  # noqa: E402
from soccer_predictor.ingest.fixtures import sync_fixtures_to_db, sync_results_to_db  # noqa: E402
from soccer_predictor.ingest.team_mapper import seed_teams_from_api  # noqa: E402
from soccer_predictor.storage.db import init_db, session_scope  # noqa: E402


def main() -> None:
    init_db()
    leagues = load_leagues()
    requested = sys.argv[1:] or list(leagues.keys())

    for code in requested:
        league = leagues[code]
        print(f"League {league.name} ({league.code})")

        try:
            with session_scope() as session:
                synced, skipped = sync_fixtures_to_db(session, league)
            print(f"  fixtures: {synced} synced, {skipped} skipped (unresolved teams)")
        except MissingApiKey as exc:
            print(f"  fixtures: skipped -- {exc}")

        if league.csv_code is None:
            # Only these leagues rely on football-data.org for match
            # *history* too (domestic leagues get it from the CSV pipeline,
            # see fetch_historical_data.py) -- keep the current season's
            # results fresh the same way fixtures are kept fresh above.
            try:
                with session_scope() as session:
                    seed_teams_from_api(session, league)
                    synced, skipped = sync_results_to_db(session, league, league.seasons[-1])
                print(f"  results: {synced} synced, {skipped} skipped (unresolved teams)")
            except (MissingApiKey, requests.RequestException) as exc:
                print(f"  results: skipped -- {exc}")

        try:
            with session_scope() as session:
                synced, skipped = injuries_module.sync_injuries_to_db(
                    session, league, season_year=dt.date.today().year
                )
            print(f"  injuries: {synced} teams synced, {skipped} skipped")
        except injuries_module.MissingApiKey as exc:
            print(f"  injuries: skipped -- {exc}")


if __name__ == "__main__":
    main()
