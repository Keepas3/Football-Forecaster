"""A team's games across competitions: linking the same club/country, merged frames,
and the Team Detail page's "Include other competitions" toggle."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League, load_leagues
from soccer_predictor.dashboard import components
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, upsert_fixture, upsert_match, upsert_team_alias
from soccer_predictor.storage.team_links import (
    fixtures_for_teams,
    linked_teams,
    matches_for_teams,
    normalize_team_name,
)

LEAGUES = {
    "LALIGA": League(code="LALIGA", name="La Liga", seasons=["2627"], csv_code="SP1"),
    "EPL": League(code="EPL", name="English Premier League", seasons=["2627"], csv_code="E0"),
    "UCL": League(code="UCL", name="UEFA Champions League", seasons=["2627"], api_competition_id=2001),
    "NL": League(code="NL", name="UEFA Nations League", seasons=["2627"], team_type="national"),
    "WC": League(code="WC", name="FIFA World Cup", seasons=["2026"], team_type="national"),
    "EURO": League(code="EURO", name="European Championship", seasons=["2028"], team_type="national"),
}


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _team(session, name, league, api_alias=None):
    team = get_or_create_team(session, name, league)
    if api_alias:
        upsert_team_alias(session, team, api_alias, "api")
    session.commit()
    return team


def _linked(session, team):
    return [(t.canonical_name, t.league_code) for t in linked_teams(session, team, LEAGUES)]


# --- linking ---------------------------------------------------------------------------------


def test_a_club_links_to_its_champions_league_row_through_the_shared_api_name(session):
    barca = _team(session, "Barcelona", "LALIGA", "FC Barcelona")
    barca_ucl = _team(session, "FC Barcelona", "UCL", "FC Barcelona")
    _team(session, "Real Madrid", "LALIGA", "Real Madrid CF")

    assert _linked(session, barca) == [("FC Barcelona", "UCL")]
    assert _linked(session, barca_ucl) == [("Barcelona", "LALIGA")]


def test_clubs_also_link_by_identical_normalized_name(session):
    leipzig = _team(session, "RB Leipzig", "LALIGA")  # no alias at all
    _team(session, "RB LEIPZIG", "UCL")
    assert _linked(session, leipzig) == [("RB LEIPZIG", "UCL")]


def test_a_club_with_no_other_competition_and_a_lookalike_club_link_to_nothing(session):
    porto = _team(session, "FC Porto", "UCL", "FC Porto")
    _team(session, "Barcelona", "LALIGA", "FC Barcelona")
    assert _linked(session, porto) == []
    # Neither itself nor a same-league namesake.
    assert linked_teams(session, porto, LEAGUES) == []


def test_national_teams_link_across_tournaments_including_spelling_variants(session):
    spain_nl = _team(session, "Spain", "NL")
    _team(session, "Spain", "WC")
    _team(session, "Spain", "EURO")
    turkey_nl, turkey_wc = _team(session, "Türkiye", "NL"), _team(session, "Turkey", "WC")
    czechia = _team(session, "Czechia", "NL")
    _team(session, "Czech Republic", "EURO")
    _team(session, "Republic of Ireland", "NL")
    _team(session, "Ireland", "EURO")

    assert _linked(session, spain_nl) == [("Spain", "EURO"), ("Spain", "WC")]
    assert _linked(session, turkey_nl) == [("Turkey", "WC")] and _linked(session, turkey_wc) == [("Türkiye", "NL")]
    assert _linked(session, czechia) == [("Czech Republic", "EURO")]
    assert _linked(session, _team(session, "Republic of Ireland", "NL")) == [("Ireland", "EURO")]


def test_former_states_stay_separate_and_a_club_never_links_to_a_country(session):
    _team(session, "Germany", "NL")
    _team(session, "West Germany", "WC")
    assert _linked(session, _team(session, "Germany", "NL")) == []
    # A club whose name equals a country's (and vice versa) is not the same team.
    panama_club = _team(session, "Panama", "LALIGA")
    _team(session, "Panama", "WC")
    assert _linked(session, panama_club) == []


def test_name_normalization_ignores_accents_and_case():
    assert normalize_team_name("  Türkiye ") == "turkiye"
    assert normalize_team_name("Curaçao") == "curacao"


def test_the_real_config_marks_the_tournaments_as_national():
    leagues = load_leagues()
    assert {code for code, league in leagues.items() if league.team_type == "national"} == {"WC", "EURO", "NL", "ASIA"}
    assert leagues["UCL"].team_type == "club" and leagues["MLS"].team_type == "club"


# --- merged frames ---------------------------------------------------------------------------


def test_matches_and_fixtures_are_merged_tagged_and_ordered(session):
    barca, ucl = _team(session, "Barcelona", "LALIGA", "FC Barcelona"), _team(session, "FC Barcelona", "UCL", "FC Barcelona")
    rival, ucl_rival = _team(session, "Villarreal", "LALIGA"), _team(session, "Villarreal CF", "UCL")
    upsert_match(session, "LALIGA", "2627", dt.date(2026, 10, 1), barca.id, rival.id, 2, 1)
    upsert_match(session, "UCL", "2627", dt.date(2026, 10, 7), ucl.id, ucl_rival.id, 3, 0)
    today = dt.date.today()
    upsert_fixture(session, "LALIGA", today + dt.timedelta(days=9), rival.id, barca.id, "SCHEDULED")
    upsert_fixture(session, "UCL", today + dt.timedelta(days=3), ucl.id, ucl_rival.id, "SCHEDULED")
    session.commit()

    matches = matches_for_teams(session, [ucl])
    assert list(matches["league_code"]) == ["UCL"] and list(matches["team_id_here"]) == [ucl.id]
    both = matches_for_teams(session, [barca, ucl])
    assert list(both["league_code"]) == ["UCL", "LALIGA"]  # newest first across competitions

    fixtures = fixtures_for_teams(session, [barca, ucl], today, today + dt.timedelta(days=30))
    assert list(fixtures["league_code"]) == ["UCL", "LALIGA"]  # soonest first
    assert fixtures_for_teams(session, [barca, ucl], today + dt.timedelta(days=60), today + dt.timedelta(days=90)).empty
    assert matches_for_teams(session, []).empty


# --- the page --------------------------------------------------------------------------------


def test_fixture_rows_with_and_without_a_prediction_line_up_in_one_table():
    rows = [
        {"Home": "A", "Away": "B", "Home win %": 50.0, "Kickoff": "x", "_predicted_home_goals": 2, "_predicted_away_goals": 1},
        {"Home": "C", "Away": "D", "Kickoff": "y"},  # a competition with no model
    ]
    aligned = components.align_rows(rows)
    assert list(aligned[1]) == list(aligned[0]) and aligned[1]["Home win %"] is None
    html = components.style_fixture_predictions(aligned).to_html()  # styling and formatting cope with the blanks
    assert "50.00" in html and "A" in html and "C" in html


def _team_page(monkeypatch, with_other_games=True, linked_names=("UEFA Champions League",)):
    from streamlit.testing.v1 import AppTest

    from soccer_predictor.dashboard.views import team_detail

    today = dt.date.today()
    other = team_detail.OtherCompetitions(
        matches=pd.DataFrame(
            [{"date": dt.date(2026, 10, 7) - dt.timedelta(days=i), "home_team_id": 500, "away_team_id": 501, "home_goals": 1,
              "away_goals": 0, "league_code": "UCL", "team_id_here": 500} for i in range(3)]
        ),
        fixtures=pd.DataFrame(
            [{"date": today + dt.timedelta(days=2), "home_team_id": 500, "away_team_id": 501, "status": "SCHEDULED",
              "kickoff_utc": None, "league_code": "UCL", "team_id_here": 500}]
        ),
        names_by_league={"UCL": {500: "FC Barcelona", 501: "Arsenal FC"}},
        names=list(linked_names) if with_other_games else [],
    )
    monkeypatch.setattr(team_detail, "_other_competitions", lambda *a, **k: other)

    def app():
        from soccer_predictor.dashboard.views import team_detail as page

        page.render()

    at = AppTest.from_function(app, default_timeout=60)
    at.query_params["team"] = "441"
    return at.run()


def _results_markup(at):
    """The recent-results HTML table (a markdown element), or None when the page has none."""
    return next((m.value for m in at.markdown if "<th>Result</th>" in m.value), None)


def _body_rows(markup):
    return markup.split("<tbody>")[1].count("<tr>")


def _schedule_markup(at):
    """The upcoming-schedule HTML table (a markdown element), or None when the page has none."""
    return next((m.value for m in at.markdown if 'class="fb-fixtures"' in m.value), None)


def test_other_competitions_are_hidden_until_the_toggle_is_on(monkeypatch):
    at = _team_page(monkeypatch)
    assert not at.exception
    assert [t.label for t in at.toggle] == ["Include other competitions (UEFA Champions League)"]
    assert at.toggle[0].value is False
    results = _results_markup(at)
    assert "<th>Competition</th>" not in results and _body_rows(results) == 10
    schedule_off = _schedule_markup(at)  # None when this team has nothing coming up in its own league
    assert schedule_off is None or "Competition" not in schedule_off

    at.toggle[0].set_value(True).run()
    assert not at.exception
    results = _results_markup(at)
    assert "<th>Competition</th>" in results and "UEFA Champions League" in results
    assert "/match_detail?league=UCL&amp;home=500&amp;away=501" in results  # opens that competition's match page
    schedule = _schedule_markup(at)
    assert "<th>Competition</th>" in schedule and "UEFA Champions League" in schedule
    # The other competition's teams link to that competition's own team pages...
    assert "/team_detail?league=UCL&amp;team=500" in schedule and "/team_detail?league=UCL&amp;team=501" in schedule
    # ...and with no model there, its match has no predicted-score link to open.
    assert "fixture=500-501" not in schedule


def test_no_toggle_when_the_team_has_no_other_competitions(monkeypatch):
    at = _team_page(monkeypatch, with_other_games=False)
    assert not at.exception and len(at.toggle) == 0
