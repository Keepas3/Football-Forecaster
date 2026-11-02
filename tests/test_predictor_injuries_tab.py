"""The Predictor page's Injuries tab: both sub-tabs are HTML tables whose team names open the team's
page, and a team's players-listed figure opens who is out for it under the team-impact table without a
reload (a click types into a hidden box, as on the leaderboard)."""

from __future__ import annotations

import re
from contextlib import contextmanager

import pytest
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from streamlit.testing.v1 import AppTest

from soccer_predictor.dashboard.views import predictor
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.model.injury_adjustment import InjuryEntry
from soccer_predictor.prediction.service import TeamInjuries
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team


@pytest.fixture
def page(monkeypatch):
    """The Predictor page over an in-memory database with three clubs, two of them with players
    listed as unavailable (Man City: two, Arsenal: one) and Chelsea with nobody; returns a function
    that runs it, and the clubs' ids. `run.tab_defaults` records which tab each run asked to open."""
    engine = create_engine("sqlite://", future=True, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = Session(engine)
    teams = {name: get_or_create_team(session, name, "EPL") for name in ("Man City", "Arsenal", "Chelsea")}
    session.commit()

    @contextmanager
    def scope():
        yield session
        session.flush()

    ids = {name: team.id for name, team in teams.items()}
    params = DixonColesParams(
        league_code="EPL",
        attack={ids["Man City"]: 1.9, ids["Arsenal"]: 1.5, ids["Chelsea"]: 1.2},
        defense={ids["Man City"]: 0.8, ids["Arsenal"]: 0.7, ids["Chelsea"]: 1.1},
        home_advantage=1.3,
        rho=-0.1,
        xi=0.0018,
        fitted_at="2026-01-01T00:00:00",
        n_matches=100,
    )
    unavailable = {
        "Man City": TeamInjuries(
            [InjuryEntry("Rodri", "defense", 0.4), InjuryEntry("Doku", "attack", 0.2)],
            ["api", "api"],
            [None, None],
            ["FPL: Knee injury - Expected back 25 Nov", "FPL: Hip injury - 50% chance of playing"],
        ),
        "Arsenal": TeamInjuries([InjuryEntry("Saka", "attack", 0.5)], ["api"], [None], ["FPL: Suspended until 29 Nov"]),
    }
    monkeypatch.setattr(
        predictor,
        "get_injuries_for_team",
        lambda _session, _id, name, _day: unavailable.get(name, TeamInjuries([], [], [], [])),
    )
    monkeypatch.setattr(predictor, "session_scope", scope)
    monkeypatch.setattr(predictor, "load_latest_params", lambda *a, **k: params)
    monkeypatch.setattr("soccer_predictor.dashboard.components.resolve_current_attack_strength", lambda *a, **k: None)
    monkeypatch.setattr("soccer_predictor.prediction.service.resolve_current_attack_strength", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    tab_defaults = []
    real_tabs = st.tabs

    def spy_on_tabs(labels, **kwargs):
        tab_defaults.append((tuple(labels), kwargs.get("default")))
        return real_tabs(labels, **kwargs)

    monkeypatch.setattr(st, "tabs", spy_on_tabs)

    def run(**query):
        def app():
            from soccer_predictor.dashboard.views import predictor as view

            view.render()

        at = AppTest.from_function(app, default_timeout=60)
        at.query_params["league"] = "EPL"
        for name, value in query.items():
            at.query_params[name] = value
        return at.run()

    run.tab_defaults = tab_defaults
    yield run, ids
    session.close()


def _table(at, heading: str) -> str | None:
    """The HTML table (a markdown element) that has a column headed `heading`."""
    return next((m.value for m in at.markdown if 'class="fb-table' in m.value and f">{heading}</th>" in m.value), None)


def _players(at) -> str:
    return _table(at, "Chance of playing")


def _impact(at) -> str:
    return _table(at, "Players listed")


def _names(markup: str, column: int) -> list[str]:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", markup.split("<tbody>")[1], re.S)
    return [re.sub(r"<[^>]+>", "", re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)[column]) for row in rows]


def _panel(at) -> str | None:
    return next((m.value for m in at.markdown if m.value.startswith("#### Injured players")), None)


def _panel_table(at) -> str | None:
    return next((m.value for m in at.markdown if 'class="fb-table' in m.value and ">Team</th>" not in m.value
                 and ">Chance of playing</th>" in m.value), None)


def _close_buttons(at) -> list[str]:
    return [b.key for b in at.button if b.key and b.key.startswith("close_injury_breakdown")]


def _pick(at, value):
    at.text_input(key="injury_pick").set_value(value).run()
    return at


def test_both_sub_tabs_are_html_tables_not_canvas_grids(page):
    run, _ = page
    at = run()
    assert not at.exception
    assert _players(at) is not None and _impact(at) is not None
    assert not at.dataframe


def test_the_players_table_lists_everyone_and_each_team_name_opens_its_page(page):
    run, ids = page
    at = run()
    markup = _players(at)
    assert _names(markup, 0) == ["Arsenal", "Man City", "Man City"]  # sorted by team, biggest impact first within one
    assert _names(markup, 1) == ["Saka", "Rodri", "Doku"]
    hrefs = " ".join(re.findall(r'href="([^"]*)"', markup))
    assert f"/team_detail?league=EPL&amp;team={ids['Arsenal']}" in hrefs and f"/team_detail?league=EPL&amp;team={ids['Man City']}" in hrefs


def test_each_injured_player_opens_their_stats_on_the_players_page(page):
    run, _ = page
    hrefs = re.findall(r'class="profile" href="([^"]*)"', _players(run()))
    assert hrefs == [
        "/players?league=EPL&amp;player=Saka%7CArsenal",
        "/players?league=EPL&amp;player=Rodri%7CMan%20City",
        "/players?league=EPL&amp;player=Doku%7CMan%20City",
    ]


def test_the_list_under_the_team_impact_table_links_its_players_too(page):
    run, ids = page
    at = _pick(run(), f"{ids['Man City']}@1000")
    hrefs = re.findall(r'class="profile" href="([^"]*)"', _panel_table(at))
    assert hrefs == ["/players?league=EPL&amp;player=Rodri%7CMan%20City", "/players?league=EPL&amp;player=Doku%7CMan%20City"]


def test_the_filters_still_narrow_the_players_table(page):
    run, _ = page
    at = run()
    at.multiselect(key="inj_teams_EPL").set_value(["Man City"]).run()
    assert _names(_players(at), 1) == ["Rodri", "Doku"]
    at.text_input(key="inj_query_EPL").set_value("dok").run()
    assert _names(_players(at), 1) == ["Doku"]


def test_the_team_impact_table_links_each_team_and_its_players_listed_figure(page):
    run, ids = page
    markup = _impact(run())
    assert _names(markup, 0) == ["Arsenal", "Man City"]  # worst hit first; a team with nobody listed (Chelsea) has no row
    assert _names(markup, 1) == ["1", "2"]
    hrefs = re.findall(r'href="([^"]*)"', markup)
    assert f"/team_detail?league=EPL&amp;team={ids['Man City']}" in hrefs
    assert "?league=EPL&amp;from=" in " ".join(h for h in hrefs if "injuries=" in h)
    assert any(h.endswith(f"&amp;injuries={ids['Arsenal']}") for h in hrefs)


def test_nothing_is_open_until_a_figure_is_clicked(page):
    run, _ = page
    at = run()
    assert _panel(at) is None and not _close_buttons(at) and 'class="selected"' not in _impact(at)
    assert run.tab_defaults[-2:] == [
        (("Upcoming Fixtures", "Team Strength Leaderboard", "Injuries", "AI notes"), None),
        (("Players", "Team impact"), None),
    ]


def test_a_click_on_a_players_listed_figure_opens_who_is_out_under_the_table_without_reloading(page):
    run, ids = page
    at = _pick(run(), f"{ids['Man City']}@1000")
    assert not at.exception
    assert _panel(at) == "#### Injured players: Man City"
    assert _close_buttons(at) == ["close_injury_breakdown_EPL"]
    assert at.query_params["injuries"] == [str(ids["Man City"])]  # the address still names the team
    assert _names(_panel_table(at), 0) == ["Rodri", "Doku"] and ">Team</th>" not in _panel_table(at)
    assert [row == '<tr class="selected">' for row in re.findall(r"<tr[^>]*>", _impact(at).split("<tbody>")[1])] == [False, True]
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "2 listed: 1 out · 1 doubtful · 0 suspended" in text
    assert f'href="/team_detail?league=EPL&amp;team={ids["Man City"]}"' in text and "Open Man City's page" in text


def test_the_link_lands_on_the_injuries_tab_and_its_team_impact_sub_tab(page):
    run, ids = page
    run(injuries=str(ids["Arsenal"]))
    assert run.tab_defaults[-2:] == [
        (("Upcoming Fixtures", "Team Strength Leaderboard", "Injuries", "AI notes"), "Injuries"),
        (("Players", "Team impact"), "Team impact"),
    ]


def test_a_link_to_a_team_opens_its_list_straight_away(page):
    run, ids = page
    at = run(injuries=str(ids["Arsenal"]))
    assert not at.exception
    assert _panel(at) == "#### Injured players: Arsenal" and _names(_panel_table(at), 0) == ["Saka"]


def test_another_click_switches_team_and_close_drops_the_parameter(page):
    run, ids = page
    at = _pick(run(), f"{ids['Man City']}@1000")
    _pick(at, f"{ids['Arsenal']}@2000")
    assert _panel(at) == "#### Injured players: Arsenal"

    at.button(key="close_injury_breakdown_EPL").click().run()
    assert not at.exception
    assert _panel(at) is None and "injuries" not in at.query_params
    at.run()  # a rerun for any other reason must not bring the old click back
    assert _panel(at) is None
    _pick(at, f"{ids['Arsenal']}@3000")  # the script stamps each click with the time, so it can be reopened
    assert _panel(at) == "#### Injured players: Arsenal"


@pytest.mark.parametrize("junk", ["", "nonsense", "99999", "-1", "3.5"])
def test_a_team_that_is_not_listed_opens_nothing_and_the_parameter_is_dropped(page, junk):
    run, ids = page
    for value in (junk, str(ids["Chelsea"])):  # Chelsea is in the league but has nobody listed
        at = run(injuries=value)
        assert not at.exception
        assert _panel(at) is None and "injuries" not in at.query_params
        assert run.tab_defaults[-2][1] is None


def test_a_click_that_names_nobody_listed_is_ignored(page):
    run, ids = page
    for junk in ("nonsense@1", f"{ids['Chelsea']}@2", "99999@3"):
        at = _pick(run(), junk)
        assert not at.exception
        assert _panel(at) is None and "injuries" not in at.query_params


def test_the_players_listed_links_carry_the_timezone_and_date_range_through_a_reload(page):
    run, _ = page
    markup = _impact(run(tz="Europe/London", **{"from": "2026-11-01", "to": "2026-11-08"}))
    link = re.search(r'href="(\?league=EPL[^"]*injuries=[^"]*)"', markup).group(1)
    assert "tz=Europe/London" in link and "from=2026-11-01" in link and "to=2026-11-08" in link


def test_without_any_injuries_there_is_no_table_and_no_hidden_box(page, monkeypatch):
    run, _ = page
    monkeypatch.setattr(predictor, "get_injuries_for_team", lambda *a: TeamInjuries([], [], [], []))
    at = run()
    assert not at.exception
    assert any("No injuries on record" in i.value for i in at.info)
    assert _players(at) is None and not any(t.key == "injury_pick" for t in at.text_input)
