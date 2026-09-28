"""Downloads football-data.co.uk season CSVs and loads them into SQLite.

Usage:
    uv run python scripts/fetch_historical_data.py [LEAGUE_CODE ...]

With no arguments, fetches every league in config/leagues.yaml. Every
season except the last one in a league's `seasons` list is treated as
concluded and only ever downloaded once; the last season is always
re-downloaded, since football-data.co.uk updates that file continuously
while it's the current in-progress season -- re-run this periodically to
pick up new results for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.config import DATA_DIR, load_leagues  # noqa: E402
from soccer_predictor.ingest.api_client import MissingApiKey  # noqa: E402
from soccer_predictor.ingest.fixtures import sync_results_to_db  # noqa: E402
from soccer_predictor.ingest.historical_csv import ingest_into_db, parse_csv  # noqa: E402
from soccer_predictor.ingest.team_mapper import seed_teams_and_aliases, seed_teams_from_api  # noqa: E402
from soccer_predictor.storage.db import init_db, session_scope  # noqa: E402

BASE_URL = "https://www.football-data.co.uk/mmz4281"
RAW_DIR = DATA_DIR / "raw"


def download_season_csv(csv_code: str, season: str, force: bool = False) -> Path:
    """`force=True` re-downloads even if a local copy exists -- needed for
    the current in-progress season, whose remote CSV keeps growing as games
    are played (unlike a fully-concluded season, which never changes once
    downloaded).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{csv_code}_{season}.csv"
    if dest.exists() and not force:
        print(f"  {dest.name} already downloaded, skipping fetch")
        return dest
    url = f"{BASE_URL}/{season}/{csv_code}.csv"
    print(f"  downloading {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def main() -> None:
    init_db()
    leagues = load_leagues()
    requested = sys.argv[1:] or list(leagues.keys())

    for code in requested:
        league = leagues[code]
        print(f"League {league.name} ({league.code})")

        if league.csv_code is None:
            # No football-data.co.uk source for this competition -- team
            # list and match history both come from football-data.org
            # instead (see ingest/team_mapper.py, ingest/fixtures.py).
            with session_scope() as session:
                seeded = seed_teams_from_api(session, league)
            print(f"  seeded {seeded} teams from the live API")
            if seeded == 0:
                print(
                    "  skipping season sync -- requires FOOTBALL_DATA_ORG_API_KEY in .env"
                )
                continue

            for season in league.seasons:
                try:
                    with session_scope() as session:
                        synced, skipped = sync_results_to_db(session, league, season)
                except (MissingApiKey, requests.RequestException) as exc:
                    print(f"  season {season}: skipped -- {exc}")
                    continue
                print(f"  season {season}: {synced} matches synced, {skipped} skipped (unresolved teams)")
            continue

        with session_scope() as session:
            seed_teams_and_aliases(session, league.code)

        for season in league.seasons:
            is_current_season = season == league.seasons[-1]
            csv_path = download_season_csv(league.csv_code, season, force=is_current_season)
            df = parse_csv(csv_path)
            with session_scope() as session:
                ingested, skipped = ingest_into_db(session, league.code, season, df)
            print(f"  season {season}: {ingested} matches ingested, {skipped} skipped (unresolved teams)")


if __name__ == "__main__":
    main()
