"""The panel under an upcoming-matches table that shows how one match's score was
predicted -- opened by the table's predicted-score link (Leagues page, and a team's
page), which names the match in the `fixture` URL parameter.
"""

from __future__ import annotations

import streamlit as st

from soccer_predictor.config import League
from soccer_predictor.dashboard.components import (
    FIXTURE_PARAM,
    SCROLL_TO_BREAKDOWN_JS,
    FixtureRef,
    render_head_to_head,
    render_prediction_breakdown,
)
from soccer_predictor.dashboard.views.match_detail import plain
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.model.team_facts import compute_head_to_head
from soccer_predictor.prediction.service import predict_fixture
from soccer_predictor.storage.db import session_scope
from soccer_predictor.storage.repository import matches_for_team


def render_fixture_workings(ref: FixtureRef, params: DixonColesParams, league: League, close_key: str) -> None:
    """Head-to-head and the full Poisson/Dixon-Coles workings for `ref`, a match in
    `league` that `params` (that league's model) predicts, with a Close button that
    drops the `fixture` parameter again."""
    st.html(f"<script>{SCROLL_TO_BREAKDOWN_JS}</script>", unsafe_allow_javascript=True)
    with st.container(key="fixture_breakdown", border=True):
        st.markdown(f"#### How {plain(ref.home_name)} v {plain(ref.away_name)} was predicted")
        if st.button("Close", key=close_key):
            del st.query_params[FIXTURE_PARAM]
            st.rerun()
        with session_scope() as session:
            home_matches = matches_for_team(session, ref.home_id)
        h2h = compute_head_to_head(ref.home_id, ref.away_id, home_matches)
        if h2h is not None:
            render_head_to_head(h2h, ref.home_name, ref.away_name)
        else:
            st.caption(f"No history on record between {plain(ref.home_name)} and {plain(ref.away_name)} yet.")

        with session_scope() as session:
            prediction = predict_fixture(
                session, params, ref.home_id, ref.away_id, ref.home_name, ref.away_name, fixture_date=ref.date
            )
            render_prediction_breakdown(
                prediction, ref.home_name, ref.away_name, session, league, ref.home_id, ref.away_id
            )
