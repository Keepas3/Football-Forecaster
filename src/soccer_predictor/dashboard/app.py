"""Streamlit multipage entrypoint. Run with:

    uv run streamlit run src/soccer_predictor/dashboard/app.py

Page content lives in dashboard/views/*.py; dashboard/navigation.py builds
the st.Page objects. This file only does the one-time setup that must run
exactly once, before any page renders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soccer_predictor.dashboard import navigation  # noqa: E402
from soccer_predictor.storage.db import init_db  # noqa: E402

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")
init_db()

pg = st.navigation(
    [
        navigation.home_page(),
        navigation.players_page(),
        navigation.team_detail_page(),
        navigation.predictor_page(),
    ]
)
pg.run()
