"""Refreshes one league's live data (fixtures, results, prediction
locks, injuries) -- the shared logic behind both
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
from soccer_predictor.ingest import national_injuries
from soccer_predictor.ingest.api_client import MissingApiKey
from soccer_predictor.ingest.fixtures import (
    sync_fixtures_to_db,
    sync_fixtures_to_db_espn,
    sync_results_to_db,
    sync_recent_results_to_db_espn,
    sync_results_to_db_espn,
)
from soccer_predictor.ingest.team_mapper import seed_teams_from_api, seed_teams_from_espn
from soccer_predictor.prediction.service import load_latest_params
from soccer_predictor.prediction.tracking import backfill_prediction_probabilities, lock_due_predictions
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
    locks_created: int = 0
    probabilities_backfilled: int = 0
    injuries_synced: int = 0
    injuries_skipped: int = 0
    injuries_error: str | None = None
    club_injuries_listed: int = 0  # countries given a club player's injury (national competitions only)


# How far ahead a quick refresh looks for fixtures (the full daily refresh
# covers the whole 14-day window; this only needs to catch stragglers).
QUICK_FIXTURE_WINDOW_DAYS = 3


def refresh_league(league: League, quick: bool = False) -> RefreshResult:
    """Everything scripts/refresh_live_data.py used to do inline for one
    league -- same branching on league.data_source/league.csv_code, same
    functions, just captured into a result object instead of printed
    directly, so callers can present it however they like.

    `quick=True` is the light version for the frequent GitHub Actions run
    that keeps results and the Track Record current: ESPN leagues read the
    scoreboard for recent results and a short fixture window (a few requests
    instead of one per team), and the slow injury sync is skipped (the single-request feeds -- Premier League, La Liga, Champions League -- still run). The
    football-data.org leagues are already one request each, so they run as
    usual. The daily full refresh still does everything.
    """
    result = RefreshResult(league_code=league.code)

    if league.data_source == "espn":
        # No football-data.org coverage at all for this league (e.g. MLS)
        # -- fixtures/results/team-seeding all come from ESPN's public,
        # keyless endpoints instead (ingest/espn_client.py).
        try:
            with session_scope() as session:
                if quick:
                    # Teams are already seeded by the daily run (and a group
                    # competition adds any new opponent it meets on its own).
                    result.fixtures_synced, result.fixtures_skipped = sync_fixtures_to_db_espn(
                        session, league, window_days=QUICK_FIXTURE_WINDOW_DAYS
                    )
                    result.results_synced, result.results_skipped = sync_recent_results_to_db_espn(
                        session, league, league.seasons[-1]
                    )
                else:
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

        # The current season's results. A league with no football-data.co.uk CSV
        # (cups, Nations League) has no other source, so its teams are seeded from
        # the API too. A domestic league's history is the CSV (see
        # fetch_historical_data.py), but that file is published a day or more after
        # the games, so the latest round was missing from the form column, the
        # table and the Track Record until then. football-data.org marks a game
        # finished within minutes: it supplies the recent results here, and the CSV
        # later writes over the same rows (matches are keyed on league, date and
        # teams, so there is never a second row).
        try:
            with session_scope() as session:
                if league.csv_code is None:
                    seed_teams_from_api(session, league)
                result.results_synced, result.results_skipped = sync_results_to_db(
                    session, league, league.seasons[-1]
                )
        except (MissingApiKey, requests.RequestException) as exc:
            result.results_error = str(exc)

    # Injuries first, so the forecasts locked below already count them. The slow
    # per-team ESPN sync is for the full refresh only; the Premier League, La Liga
    # and Champions League feeds are each a single cached request, so they run in
    # the quick refresh too (injury news moves through the week, not once a day).
    if not quick or league.code in injuries_module.SINGLE_REQUEST_LEAGUES:
        try:
            with session_scope() as session:
                result.injuries_synced, result.injuries_skipped = injuries_module.sync_injuries_to_db(
                    session, league, season_year=dt.date.today().year
                )
        except requests.RequestException as exc:
            result.injuries_error = str(exc)

    # A country's players are injured at their clubs: list those under the country too, once the club
    # leagues above have been synced (the daily run; the quick one leaves them as they were).
    if league.team_type == "national" and not quick:
        try:
            with session_scope() as session:
                result.club_injuries_listed, _ = national_injuries.sync_club_injuries_to_national_teams(session, league)
        except requests.RequestException as exc:
            result.injuries_error = result.injuries_error or str(exc)

    with session_scope() as session:
        params = load_latest_params(session, league.code)
        if params is not None:
            # A backstop: scripts/lock_lineups.py is what normally locks a match
            # when its lineup is announced; this catches whatever a refresh run
            # happens to coincide with. Locking is idempotent, so both are safe.
            result.locks_created = lock_due_predictions(session, league, params)
            result.probabilities_backfilled = backfill_prediction_probabilities(session, league, params)

    return result
