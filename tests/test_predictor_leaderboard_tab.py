"""The Predictor page's Team Strength Leaderboard tab: a sortable HTML table whose team names open
the team's page (not a canvas grid), with the name search kept."""

from __future__ import annotations

import datetime as dt
import re
from contextlib import contextmanager

import pytest
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from streamlit.testing.v1 import AppTest

from soccer_predictor.config import load_leagues
from soccer_predictor.dashboard import rating_workings
from soccer_predictor.dashboard.views import predictor
from soccer_predictor.ingest.player_importance import CurrentAttackDetail
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.model.lineup_adjustment import ExpectedPlayer
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, update_team_crest, upsert_match


@pytest.fixture
def page(monkeypatch):
    """The Predictor page over an in-memory database with three clubs playing this season, one
    with a crest, and a fourth the model rated in an earlier season that has since left the league;
    returns a function that runs it, and the teams' ids."""
    engine = create_engine("sqlite://", future=True, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = Session(engine)
    teams = {name: get_or_create_team(session, name, "EPL") for name in ("Man City", "Arsenal", "Chelsea", "Watford")}
    update_team_crest(session, teams["Man City"].id, "https://crests.example/city.png")
    this_season, last_season = load_leagues()["EPL"].seasons[-1], load_leagues()["EPL"].seasons[-2]
    upsert_match(session, "EPL", this_season, dt.date(2026, 9, 1), teams["Man City"].id, teams["Arsenal"].id, 2, 1)
    upsert_match(session, "EPL", this_season, dt.date(2026, 9, 8), teams["Chelsea"].id, teams["Man City"].id, 0, 0)
    upsert_match(session, "EPL", last_season, dt.date(2026, 3, 1), teams["Watford"].id, teams["Chelsea"].id, 1, 1)
    session.commit()

    @contextmanager
    def scope():
        yield session
        session.flush()

    ids = {name: team.id for name, team in teams.items()}
    params = DixonColesParams(
        league_code="EPL",
        attack={ids["Man City"]: 1.9, ids["Arsenal"]: 1.5, ids["Chelsea"]: 1.2, ids["Watford"]: 1.0},
        defense={ids["Man City"]: 0.8, ids["Arsenal"]: 0.7, ids["Chelsea"]: 1.1, ids["Watford"]: 1.0},
        home_advantage=1.3,
        rho=-0.1,
        xi=0.0018,
        fitted_at="2026-01-01T00:00:00",
        n_matches=100,
    )
    monkeypatch.setattr(predictor, "session_scope", scope)
    monkeypatch.setattr(predictor, "load_latest_params", lambda *a, **k: params)
    monkeypatch.setattr("soccer_predictor.dashboard.components.resolve_current_attack_strength", lambda *a, **k: None)
    monkeypatch.setattr("soccer_predictor.prediction.service.resolve_current_attack_strength", lambda *a, **k: None)
    monkeypatch.setattr(rating_workings, "session_scope", scope)
    monkeypatch.setattr(rating_workings, "resolve_current_attack_detail", lambda *a, **k: None)
    monkeypatch.setattr(rating_workings, "expected_squad", lambda *a, **k: [])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Which tab each run asked to open (st.tabs' `default`), newest last.
    tab_defaults = []
    real_tabs = st.tabs

    def spy_on_tabs(labels, **kwargs):
        tab_defaults.append(kwargs.get("default"))
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


def _table(at) -> str | None:
    return next((m.value for m in at.markdown if 'class="fb-table"' in m.value), None)


def _panel_headings(at) -> list[str]:
    return [m.value for m in at.markdown if m.value.startswith("#### How ") and "calculated" in m.value]


def _team_order(markup: str) -> list[str]:
    return re.findall(r'class="team"[^>]*>([^<]*)</a>', markup)


def test_the_leaderboard_is_a_linked_html_table_ranked_by_net_strength(page):
    run, ids = page
    at = run()
    assert not at.exception
    markup = _table(at)
    assert markup is not None
    assert _team_order(markup) == ["Man City", "Arsenal", "Chelsea"]  # net strength +1.10, +0.80, +0.10
    assert not any("Net strength" in d.value.columns for d in at.dataframe)  # no canvas grid any more


def test_a_club_that_has_left_the_league_is_not_on_the_leaderboard(page):
    run, _ = page
    at = run()
    assert "Watford" not in _table(at)  # rated by the model, but no game or fixture this season


def test_each_name_opens_its_team_page_and_the_crest_is_drawn(page):
    run, ids = page
    markup = _table(run())
    hrefs = " ".join(re.findall(r'href="([^"]*)"', markup))
    for name in ("Man City", "Arsenal", "Chelsea"):
        assert f"/team_detail?league=EPL&amp;team={ids[name]}" in hrefs
    assert markup.count("<img") == 1 and "https://crests.example/city.png" in markup


def test_searching_filters_the_rows_and_keeps_their_rank(page):
    run, _ = page
    at = run()
    at.text_input(key="leaderboard_search_EPL").set_value("chel").run()
    assert not at.exception
    rows = re.findall(r"<tr>.*?</tr>", _table(at).split("<tbody>")[1])
    assert len(rows) == 1 and ">3</td>" in rows[0] and "Chelsea" in rows[0]


def test_a_search_that_matches_nobody_says_so_instead_of_an_empty_table(page):
    run, _ = page
    at = run()
    at.text_input(key="leaderboard_search_EPL").set_value("nobody").run()
    assert any('No team matches "nobody"' in i.value for i in at.info)
    assert _table(at) is None


def test_without_a_rating_in_the_url_no_workings_are_open_and_the_first_tab_shows(page):
    run, _ = page
    at = run()
    assert _panel_headings(at) == []
    assert run.tab_defaults[-1] is None
    assert 'class="selected"' not in _table(at)


@pytest.mark.parametrize("kind, heading", [("attack", "attack"), ("defense", "defense"), ("net", "net strength")])
def test_a_rating_in_the_url_opens_that_figures_workings_on_the_leaderboard_tab(page, kind, heading):
    run, ids = page
    at = run(rating=f"{ids['Man City']}-{kind}")
    assert not at.exception
    assert _panel_headings(at) == [f"#### How Man City's {heading} is calculated"]
    assert run.tab_defaults[-1] == "Team Strength Leaderboard"  # the link reloads the page, which starts on the first tab
    rows = re.findall(r"<tr[^>]*>", _table(at).split("<tbody>")[1])
    assert [row == '<tr class="selected">' for row in rows] == [True, False, False]  # Man City leads the board


def test_the_workings_use_the_teams_own_numbers(page):
    run, ids = page
    text = " ".join(m.value for m in run(rating=f"{ids['Arsenal']}-net").markdown)
    assert "= 1.50 − 0.70 = **+0.80**" in text and "**2nd** of 3 clubs this season" in text


def test_the_attack_workings_say_when_there_is_no_chance_data_to_adjust_with(page):
    run, ids = page
    text = " ".join(m.value for m in run(rating=f"{ids['Man City']}-attack").markdown)
    assert "needs at least 3 matches played this season" in text  # the stub finds no reading for the club
    assert "is **1.90**" in text  # the fitted rating is still explained


def test_the_attack_workings_show_the_chances_and_the_squads_biggest_contributors(page, monkeypatch):
    run, ids = page
    monkeypatch.setattr(
        rating_workings,
        "resolve_current_attack_detail",
        lambda *a, **k: CurrentAttackDetail(team_rate=2.4, league_mean_rate=2.0, matches_played=10, teams_compared=20, ratio=1.2),
    )
    squad = [ExpectedPlayer("Erling Haaland", "attack", 0.5, 900, False, "FWD", 9, 1, 8.0, 1.0)]
    monkeypatch.setattr(rating_workings, "expected_squad", lambda *a, **k: squad)
    at = run(rating=f"{ids['Man City']}-attack")
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "creates **1.20×** the average" in text and "× 15% = **+3.0%**" in text
    assert "Erling Haaland" in text and "<b>8.7</b>" in text


def test_a_team_that_is_not_on_the_leaderboard_opens_nothing_and_the_parameter_is_dropped(page):
    run, ids = page
    at = run(rating=f"{ids['Watford']}-net")  # rated by the model, but gone from this season's league
    assert not at.exception
    assert _panel_headings(at) == [] and "rating" not in at.query_params
    assert run.tab_defaults[-1] is None


@pytest.mark.parametrize("junk", ["", "nonsense", "1-speed", "99999-net", "-1-net"])
def test_a_rating_that_is_not_a_figure_of_a_listed_team_is_ignored(page, junk):
    run, _ = page
    at = run(rating=junk)
    assert not at.exception
    assert _panel_headings(at) == [] and "rating" not in at.query_params


def test_close_drops_the_rating_and_the_workings_go(page):
    run, ids = page
    at = run(rating=f"{ids['Chelsea']}-defense")
    assert len(_panel_headings(at)) == 1
    at.button(key="close_rating_breakdown_EPL").click().run()
    assert not at.exception
    assert _panel_headings(at) == [] and "rating" not in at.query_params


def test_the_workings_stay_open_while_the_name_search_hides_their_row(page):
    run, ids = page
    at = run(rating=f"{ids['Man City']}-net")
    at.text_input(key="leaderboard_search_EPL").set_value("chel").run()
    assert not at.exception
    assert len(_panel_headings(at)) == 1 and 'class="selected"' not in _table(at)


def test_the_figure_links_carry_the_timezone_and_date_range_through_the_reload(page):
    run, _ = page
    markup = _table(run(tz="Europe/London", **{"from": "2026-11-01", "to": "2026-11-08"}))
    link = re.search(r'href="(\?league=EPL[^"]*rating=[^"]*)"', markup).group(1)
    assert "tz=Europe/London" in link and "from=2026-11-01" in link and "to=2026-11-08" in link


# ---- opening the workings in place: the click script types into a hidden box ----


def _pick(at, value):
    at.text_input(key="rating_pick").set_value(value).run()
    return at


def test_a_click_typed_into_the_hidden_box_opens_the_workings_without_reloading(page):
    run, ids = page
    at = _pick(run(), f"{ids['Man City']}-net@1000")
    assert not at.exception
    assert _panel_headings(at) == ["#### How Man City's net strength is calculated"]
    assert at.query_params["rating"] == [f"{ids['Man City']}-net"]  # the address still names it
    assert run.tab_defaults[-1] == "Team Strength Leaderboard"


def test_a_second_click_replaces_the_open_workings(page):
    run, ids = page
    at = _pick(run(), f"{ids['Man City']}-net@1000")
    _pick(at, f"{ids['Arsenal']}-defense@2000")
    assert _panel_headings(at) == ["#### How Arsenal's defense is calculated"]


def test_the_same_figure_can_be_opened_again_after_close_but_only_by_a_new_click(page):
    run, ids = page
    at = _pick(run(), f"{ids['Chelsea']}-attack@1000")
    at.button(key="close_rating_breakdown_EPL").click().run()
    assert _panel_headings(at) == []
    at.run()  # a rerun for any other reason must not bring the old click back
    assert _panel_headings(at) == []
    _pick(at, f"{ids['Chelsea']}-attack@3000")  # the script stamps each click with the time
    assert len(_panel_headings(at)) == 1


def test_a_click_that_names_nothing_listed_is_ignored(page):
    run, ids = page
    for junk in ("nonsense@1", f"{ids['Watford']}-net@2", "1-speed@3"):
        at = _pick(run(), junk)
        assert not at.exception
        assert _panel_headings(at) == [] and "rating" not in at.query_params


def test_the_box_is_not_drawn_when_there_is_no_table_to_click(page):
    run, _ = page
    at = run()
    at.text_input(key="leaderboard_search_EPL").set_value("nobody").run()
    assert not any(t.key == "rating_pick" for t in at.text_input)
