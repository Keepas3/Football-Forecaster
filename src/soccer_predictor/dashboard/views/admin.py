"""Admin page: refreshes one league's live data or retrains its model from
inside the running app, for deployments with no way to run the CLI scripts
(e.g. Streamlit Community Cloud, which has no cron). Hidden from the
sidebar (see dashboard/navigation.py) and, more importantly, fully disabled
unless ADMIN_PASSWORD is set -- reaching this page's URL alone never does
anything without also passing that password. Never wired to fire
automatically; every action here is an explicit button click.
"""

from __future__ import annotations

import time

import streamlit as st

from soccer_predictor.config import admin_password, load_leagues
from soccer_predictor.dashboard import notes_access as na
from soccer_predictor.dashboard.components import league_option_label
from soccer_predictor.ingest.refresh import refresh_league
from soccer_predictor.prediction.training import train_league
from soccer_predictor.storage.db import session_scope

AUTH_SESSION_KEY = "admin_authenticated"
ADMIN_FAILED_KEY = "admin_failed_attempts"


def _render_refresh_result(result) -> None:
    if result.fixtures_error:
        st.warning(f"Fixtures: skipped - {result.fixtures_error}")
    else:
        st.write(f"Fixtures: {result.fixtures_synced} synced, {result.fixtures_skipped} skipped (unresolved teams)")

    if result.results_error:
        st.warning(f"Results: skipped - {result.results_error}")
    elif result.results_synced or result.results_skipped:
        st.write(f"Results: {result.results_synced} synced, {result.results_skipped} skipped (unresolved teams)")

    st.write(f"Prediction tracking: {result.locks_created} new forecast(s) locked in")

    if result.injuries_error:
        st.warning(f"Injuries: skipped - {result.injuries_error}")
    else:
        st.write(f"Injuries: {result.injuries_synced} teams synced, {result.injuries_skipped} skipped")


def render() -> None:
    st.title("Admin: Data Refresh")

    password = admin_password()
    if password is None:
        st.error("Admin refresh isn't enabled. Set ADMIN_PASSWORD to turn this page on.")
        return

    if not st.session_state.get(AUTH_SESSION_KEY):
        if na.attempts_left(st.session_state, ADMIN_FAILED_KEY) == 0 or na.ADMIN_LIMITER.blocked():
            st.error("Too many wrong passwords. Try again later.")
            return
        entered = st.text_input("Password", type="password")
        if st.button("Unlock"):
            result = na.try_unlock(
                st.session_state,
                entered,
                password,
                limiter=na.ADMIN_LIMITER,
                unlocked_key=AUTH_SESSION_KEY,
                failed_key=ADMIN_FAILED_KEY,
            )
            if result == na.OK:
                st.rerun()
            time.sleep(1)  # slows down guessing
            st.error("Too many wrong passwords. Try again later." if result == na.LOCKED_OUT else "Incorrect password.")
        return

    leagues = load_leagues()
    if not leagues:
        st.error("No leagues configured in config/leagues.yaml.")
        return

    league_code = st.selectbox(
        "League",
        options=list(leagues.keys()),
        format_func=lambda c: league_option_label(leagues[c]),
    )
    league = leagues[league_code]

    st.subheader("Refresh live data")
    st.caption(
        "Same as running `refresh_live_data.py` for this league: fixtures, current-season "
        "results, prediction locks (for matches about to start), and injuries."
    )
    if st.button("Refresh live data"):
        with st.spinner(f"Refreshing {league.name}..."):
            result = refresh_league(league)
        _render_refresh_result(result)

    st.divider()
    st.subheader("Retrain model")
    st.caption("Same as running `run_training.py` for this league.")
    if st.button("Retrain model"):
        if not league.supports_predictions:
            st.warning(f"{league.name} has no historical match data source configured, so it can't be trained.")
        else:
            with st.spinner(f"Training {league.name}..."):
                with session_scope() as session:
                    params = train_league(session, league.code)
            st.success(f"Trained on {params.n_matches:,} matches.")
