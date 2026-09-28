"""Constructs every st.Page used by the app -- the one place url_path
strings are defined. st.switch_page matches pages by url_path (not Python
object identity), so calling these factories fresh anywhere is safe; there's
no need to pass the original Page objects around via session_state.
"""

from __future__ import annotations

import streamlit as st

from soccer_predictor.dashboard.views import home, predictor, team_detail


def home_page() -> st.Page:
    return st.Page(home.render, title="Leagues", icon="⚽", url_path="", default=True)


def team_detail_page() -> st.Page:
    # Hidden from the sidebar, not removed -- it's only ever reached by
    # clicking a team's row (Leagues page) or a fixture's team name, never
    # picked from the nav directly, so a visible sidebar entry was a dead
    # end (no team_id in the URL yet). st.switch_page still works against a
    # hidden page; only its sidebar link disappears.
    return st.Page(
        team_detail.render, title="Team Detail", icon="👕", url_path="team_detail", visibility="hidden"
    )


def predictor_page() -> st.Page:
    return st.Page(predictor.render, title="Predictor", icon="🔮", url_path="predictor")
