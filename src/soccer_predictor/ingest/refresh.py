"""Refreshes one league's live data (fixtures, results, prediction
snapshots, injuries) -- the shared logic behind both
scripts/refresh_live_data.py (the CLI entrypoint) and
dashboard/views/admin.py (the in-app equivalent for deployments with no
cron, e.g. Streamlit Community Cloud). Factored out so those two callers
can't drift into two different copies of the same orchestration.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import requests

from soccer_predictor.config import League
from soccer_predictor.ingest import injuries as injuries_module
from soccer_predictor.ingest.api_client import MissingApiKey
from soccer_predictor.ingest.fixtures import (
    sync_fixtures_to_db,
    sync_fixtures_to_db_espn,
    sync_results_to_db,
    sync_results_to_db_espn,
)
from soccer_predictor.ingest.team_mapper import seed_teams_from_api, seed_teams_from_espn
from soccer_predictor.prediction.service import load_latest_params
from soccer_predictor.prediction.tracking import snapshot_upcoming_predictions
from soccer_predictor.storage.db import session_scope


@dataclass
class RefreshResult:
    league_code: str
    fixtures_synced: int = 0
    fixtures_skipped: int = 0
    fixtures_error: str | None = None
    results_synced: int = 0
    results_skipped: int = 0
    results_error: str | None = None
    snapshots_created: int = 0
    injuries_synced: int = 0
    injuries_skipped: int = 0
    injuries_error: str | None = None


def refresh_league(league: League) -> RefreshResult:
    """Everything scripts/refresh_live_data.py used to do inline for one
    league -- same branching on league.data_source/league.csv_code, same
    functions, just captured into a result object instead of printed
    directly, so callers can present it however they like.
    """
    result = RefreshResult(league_code=league.code)

    if league.data_source == "espn":
        # No football-data.org coverage at all for this league (e.g. MLS)
        # -- fixtures/results/team-seeding all come from ESPN's public,
        # keyless endpoints instead (ingest/espn_client.py).
        try:
            with session_scope() as session:
                seed_teams_from_espn(session, league)
                result.fixtures_synced, result.fixtures_skipped = sync_fixtures_to_db_espn(session, league)
                result.results_synced, result.results_skipped = sync_results_to_db_espn(
                    session, league, league.seasons[-1]
                )
        except requests.RequestException as exc:
            result.fixtures_error = result.results_error = str(exc)
    else:
        try:
            with session_scope() as session:
                result.fixtures_synced, result.fixtures_skipped = sync_fixtures_to_db(session, league)
        except MissingApiKey as exc:
            result.fixtures_error = str(exc)

        if league.csv_code is None:
            # Only these leagues rely on football-data.org for match
            # *history* too (domestic leagues get it from the CSV
            # pipeline, see fetch_historical_data.py) -- keep the current
            # season's results fresh the same way fixtures are above.
            try:
                with session_scope() as session:
                    seed_teams_from_api(session, league)
                    result.results_synced, result.results_skipped = sync_results_to_db(
                        session, league, league.seasons[-1]
                    )
            except (MissingApiKey, requests.RequestException) as exc:
                result.results_error = str(exc)

    with session_scope() as session:
        params = load_latest_params(session, league.code)
        if params is not None:
            result.snapshots_created = snapshot_upcoming_predictions(session, league, params)

    with session_scope() as session:
        result.injuries_synced, result.injuries_skipped = injuries_module.sync_injuries_to_db(
            session, league, season_year=dt.date.today().year
        )

    return result
