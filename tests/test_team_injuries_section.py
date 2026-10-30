"""The Injuries section of a team's page: a sortable table like the Injuries tab's, not a bullet list."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from soccer_predictor.dashboard import components, injury_table as it
from soccer_predictor.dashboard.views import team_detail
from soccer_predictor.model.injury_adjustment import InjuryEntry
from soccer_predictor.prediction.service import TeamInjuries

TEAM_PARAM = "441"  # a team in the local database, as the other team-page tests use


def _page(monkeypatch, team_injuries):
    monkeypatch.setattr(team_detail, "get_injuries_for_team", lambda *a, **k: team_injuries)
    monkeypatch.setattr(
        team_detail,
        "_other_competitions",
        lambda *a, **k: team_detail.OtherCompetitions(
            matches=pd.DataFrame(), fixtures=pd.DataFrame(), names_by_league={}, names=[]
        ),
    )

    def app():
        from soccer_predictor.dashboard.views import team_detail as page

        page.render()

    at = AppTest.from_function(app, default_timeout=60)
    at.query_params["team"] = TEAM_PARAM
    return at.run()


def _injured_team():
    return TeamInjuries(
        entries=[InjuryEntry("Bukayo Saka", "attack", 0.54), InjuryEntry("Luke Shaw", "defense", 0.30)],
        sources=["api", "manual"],
        expected_return_dates=[None, dt.date(2027, 1, 5)],
        notes=["FPL: Hamstring injury - 75% chance of playing", "Knee"],
    )


def _injuries_table(at):
    return next((d.value for d in at.dataframe if "Impact" in d.value.columns), None)


def test_a_teams_injuries_are_a_table_with_status_chance_return_and_impact(monkeypatch):
    at = _page(monkeypatch, _injured_team())
    assert not at.exception
    table = _injuries_table(at)
    assert table is not None
    assert list(table.columns) == [c for c in it.PLAYER_COLUMNS if c != "Team"]  # the team's own page: no team column
    by_player = table.set_index("Player")
    assert by_player.loc["Bukayo Saka", "Status"] == it.DOUBTFUL and by_player.loc["Bukayo Saka", "Chance of playing"] == 75
    assert by_player.loc["Luke Shaw", "Expected back"] == dt.date(2027, 1, 5)
    assert list(table["Player"]) == ["Bukayo Saka", "Luke Shaw"]  # biggest impact first
    captions = [c.value for c in at.caption]
    assert any("2 players: 1 out · 1 doubtful · 0 suspended" in c for c in captions)
    assert not any(m.value.startswith("- ") and "source:" in m.value for m in at.markdown)  # the old bullets are gone


def test_a_team_with_nobody_listed_says_so(monkeypatch):
    at = _page(monkeypatch, TeamInjuries(entries=[], sources=[], expected_return_dates=[], notes=[]))
    assert not at.exception
    assert _injuries_table(at) is None
    assert any("No injuries on record." in c.value for c in at.caption)


def test_table_height_fits_the_rows_up_to_a_cap():
    assert components.table_height(0) == components.INJURY_HEADER_PX + 4
    assert components.table_height(5) == components.INJURY_HEADER_PX + 5 * components.INJURY_ROW_PX + 4
    assert components.table_height(10_000) == components.INJURY_TABLE_MAX_PX


@pytest.mark.parametrize(
    ("league", "expected"),
    [
        ("LALIGA", "Biwenger"),
        ("UCL", "Biwenger"),
        ("MLS", "starting11.com"),
        ("BUNDESLIGA", "starting11.com"),
        ("SERIEA", "starting11.com"),
        ("LIGUE1", "starting11.com"),
        ("EPL", None),  # the Premier League's own fantasy feed needs no credit line here
        ("EURO", None),
    ],
)
def test_the_source_credit_names_the_feed_a_league_uses(league, expected):
    note = components.injury_source_note(league)
    assert (expected is None and note is None) or (expected in note)
