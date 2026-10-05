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

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.config import DATA_DIR, League, load_leagues  # noqa: E402
from soccer_predictor.ingest import euro_archive, worldcup_archive  # noqa: E402
from soccer_predictor.ingest.api_client import MissingApiKey  # noqa: E402
from soccer_predictor.ingest.fixtures import sync_results_to_db, sync_results_to_db_espn  # noqa: E402
from soccer_predictor.ingest.historical_csv import (  # noqa: E402
    ingest_goals_into_db,
    ingest_into_db,
    parse_csv,
)
from soccer_predictor.ingest.team_mapper import (  # noqa: E402
    seed_teams_and_aliases,
    seed_teams_from_api,
    seed_teams_from_espn,
    seed_teams_from_names,
)
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


def _sync_worldcup_archive_season(
    session, league: League, season: str, matches_df: pd.DataFrame, goals: list[dict]
) -> tuple[int, int, int, int]:
    season_matches = matches_df[matches_df["season"] == season]
    season_goals = [g for g in goals if g["season"] == season]
    ingested, skipped = ingest_into_db(session, league.code, season, season_matches)
    goal_ingested, goal_skipped = ingest_goals_into_db(
        session, league.code, season, season_goals, source=league.data_source
    )
    return ingested, skipped, goal_ingested, goal_skipped


def _sync_euro_archive_season(session, league: League, season: str) -> tuple[int, int, int, int]:
    year = 2021 if season == "2020" else int(season)  # see euro_archive's module docstring
    text = euro_archive.fetch_tournament_text(season)
    matches = euro_archive.parse_matches(text, year)
    goals = euro_archive.parse_goalscorers(text, year)

    team_names = {m["home_team_name"] for m in matches} | {m["away_team_name"] for m in matches}
    seed_teams_from_names(session, league.code, team_names)

    matches_df = pd.DataFrame(
        matches,
        columns=["date", "home_team_name", "away_team_name", "home_goals", "away_goals", "group_name"],
    )
    ingested, skipped = ingest_into_db(session, league.code, season, matches_df)
    goal_ingested, goal_skipped = ingest_goals_into_db(
        session, league.code, season, goals, source=league.data_source
    )
    return ingested, skipped, goal_ingested, goal_skipped


def _sync_archive_league(league: League) -> None:
    """WC/EURO: historical seasons come from a static archive
    (ingest/worldcup_archive.py, ingest/euro_archive.py); the current/next
    tournament (not archived yet) still goes through the same
    football-data.org path every other non-CSV league below uses.
    """
    if league.data_source == "archive_worldcup":
        known_seasons = worldcup_archive.WORLDCUP_AVAILABLE_SEASONS
    else:
        known_seasons = set(euro_archive.SEASON_TO_FOLDER)
    archive_seasons = [s for s in league.seasons if s in known_seasons]
    live_seasons = [s for s in league.seasons if s not in known_seasons]

    if archive_seasons:
        if league.data_source == "archive_worldcup":
            matches_df = worldcup_archive.parse_matches(worldcup_archive.fetch_matches())
            goals = worldcup_archive.parse_goals(worldcup_archive.fetch_goals())
            with session_scope() as session:
                seeded = seed_teams_from_names(
                    session,
                    league.code,
                    set(matches_df["home_team_name"]) | set(matches_df["away_team_name"]),
                )
            print(f"  seeded {seeded} teams from the World Cup archive")

        for season in archive_seasons:
            with session_scope() as session:
                if league.data_source == "archive_worldcup":
                    ingested, skipped, goal_ingested, goal_skipped = _sync_worldcup_archive_season(
                        session, league, season, matches_df, goals
                    )
                else:
                    ingested, skipped, goal_ingested, goal_skipped = _sync_euro_archive_season(
                        session, league, season
                    )
            print(
                f"  season {season}: {ingested} matches ingested, {skipped} skipped (unresolved teams); "
                f"{goal_ingested} goals ingested, {goal_skipped} skipped"
            )

    if live_seasons:
        with session_scope() as session:
            seeded = seed_teams_from_api(session, league)
        print(f"  seeded {seeded} teams from the live API (current/next tournament)")
        if seeded == 0:
            print("  skipping live season sync -- requires FOOTBALL_DATA_ORG_API_KEY in .env")
            return
        for season in live_seasons:
            try:
                with session_scope() as session:
                    synced, skipped = sync_results_to_db(session, league, season)
            except (MissingApiKey, requests.RequestException) as exc:
                print(f"  season {season}: skipped -- {exc}")
                continue
            print(f"  season {season}: {synced} matches synced, {skipped} skipped (unresolved teams)")


def main() -> None:
    init_db()
    leagues = load_leagues()
    # --current-season-only: the light run for the frequent GitHub Actions
    # refresh -- re-download just each CSV league's in-progress season (a few
    # small static files, no API key) and skip every other league, which
    # refresh_live_data.py --quick already keeps current.
    current_season_only = "--current-season-only" in sys.argv[1:]
    requested = [a for a in sys.argv[1:] if a != "--current-season-only"] or list(leagues.keys())

    for code in requested:
        league = leagues[code]
        if current_season_only and (league.csv_code is None or league.data_source != "football_data_org"):
            continue
        print(f"League {league.name} ({league.code})")

        if league.data_source in ("archive_worldcup", "archive_euro"):
            _sync_archive_league(league)
            continue

        if league.data_source == "espn":
            # No football-data.org coverage at all for this competition --
            # team list and match history both come from ESPN's public,
            # keyless endpoints instead (see ingest/espn_client.py).
            with session_scope() as session:
                seeded = seed_teams_from_espn(session, league)
            print(f"  seeded {seeded} teams from ESPN")
            if seeded == 0:
                print("  skipping season sync -- ESPN team seeding returned nothing")
                continue

            for season in league.seasons:
                try:
                    with session_scope() as session:
                        synced, skipped = sync_results_to_db_espn(session, league, season)
                except requests.RequestException as exc:
                    print(f"  season {season}: skipped -- {exc}")
                    continue
                print(f"  season {season}: {synced} matches synced, {skipped} skipped (unresolved teams)")
            continue

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

        csv_seasons = league.seasons[-1:] if current_season_only else league.seasons
        for season in csv_seasons:
            is_current_season = season == league.seasons[-1]
            csv_path = download_season_csv(league.csv_code, season, force=is_current_season)
            df = parse_csv(csv_path)
            with session_scope() as session:
                ingested, skipped = ingest_into_db(session, league.code, season, df)
            print(f"  season {season}: {ingested} matches ingested, {skipped} skipped (unresolved teams)")


if __name__ == "__main__":
    main()
