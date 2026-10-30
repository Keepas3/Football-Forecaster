"""The upcoming-matches table as HTML: team links, kickoff -> match details,
predicted score -> how it was calculated; and the timezone that travels in the links."""

from __future__ import annotations

import datetime as dt

import pytest

from soccer_predictor.dashboard import components
from soccer_predictor.dashboard.components import FixtureRef, fixture_key, fixtures_table_html, parse_fixture_key

DAY = dt.date(2026, 10, 31)
REF = FixtureRef(7, 17, "Chelsea", "Man United", DAY)


def _row(**overrides):
    row = {
        "Kickoff": "Oct 31, 2026 08:30 AM EDT",
        "Home": "Chelsea",
        "Away": "Man United",
        "Home win %": 45.456,
        "Draw %": 23.87,
        "Away win %": 30.67,
        "Predicted score": "2-1",
        "Prediction": "🔒 Locked in",
        "Over 2.5 goals %": 59.71,
        "Both teams score %": 59.26,
        "_predicted_home_goals": 2,
        "_predicted_away_goals": 1,
    }
    row.update(overrides)
    return row


def _hrefs(markup: str) -> list[str]:
    import re

    return re.findall(r'href="([^"]*)"', markup)


def test_names_kickoff_and_score_are_links_to_the_right_pages():
    markup = fixtures_table_html([_row()], [REF], "EPL", season="2627", tz_name="Europe/London", match_details=True)
    assert _hrefs(markup) == [
        "/match_detail?league=EPL&amp;home=7&amp;away=17&amp;date=2026-10-31",
        "/team_detail?league=EPL&amp;team=7&amp;season=2627&amp;tz=Europe/London",
        "/team_detail?league=EPL&amp;team=17&amp;season=2627&amp;tz=Europe/London",
        "?league=EPL&amp;season=2627&amp;tz=Europe/London&amp;fixture=7-17-2026-10-31",
    ]
    assert markup.count('target="_self"') == 4 and "_blank" not in markup  # same tab, like the standings links


def test_page_params_ride_only_on_the_link_that_reloads_the_same_page():
    markup = fixtures_table_html([_row()], [REF], "EPL", match_details=True, page_params="&from=2026-11-01&to=2026-11-09")
    with_params = [h for h in _hrefs(markup) if "from=" in h]
    assert with_params == ["?league=EPL&amp;from=2026-11-01&amp;to=2026-11-09&amp;fixture=7-17-2026-10-31"]


def test_a_league_without_match_details_leaves_the_kickoff_as_plain_text():
    markup = fixtures_table_html([_row()], [REF], "EPL", match_details=False)
    assert "/match_detail" not in markup and "Oct 31, 2026 08:30 AM EDT" in markup


def test_rows_without_a_model_have_names_but_no_prediction_link_or_columns():
    row = {"Home": "Chelsea", "Away": "Man United", "Kickoff": "Oct 31, 2026"}
    markup = fixtures_table_html([row], [REF], "UCL", match_details=True)
    assert "fixture=" not in markup and "Home win" not in markup
    assert markup.count("/team_detail?") == 2 and "/match_detail?" in markup


def test_percentages_have_two_decimals_and_a_missing_one_is_a_dash():
    markup = fixtures_table_html([_row(**{"Draw %": None})], [REF], "EPL")
    assert ">45.46<" in markup and ">30.67<" in markup and ">59.71<" in markup
    assert ">-<" in markup  # the missing draw figure


def test_team_names_are_coloured_by_the_predicted_scoreline():
    win = fixtures_table_html([_row()], [REF], "EPL")  # 2-1
    assert 'style="color: #2e7d32; font-weight: 600">Chelsea' in win and 'style="color: #c62828">Man United' in win
    draw = fixtures_table_html([_row(_predicted_home_goals=1, _predicted_away_goals=1)], [REF], "EPL")
    assert draw.count("#9e9e9e") == 2
    nothing = fixtures_table_html([_row(_predicted_home_goals=None, _predicted_away_goals=None)], [REF], "EPL")
    assert "#2e7d32" not in nothing and "#c62828" not in nothing and "#9e9e9e" not in nothing


def test_the_open_match_is_highlighted_and_text_from_outside_is_escaped():
    hostile = '"><script>alert(1)</script>'
    row = _row(Home=hostile, Away="A & B")
    markup = fixtures_table_html([row, _row()], [FixtureRef(1, 2, hostile, "A & B", DAY), REF], "EPL", selected=1)
    assert "<script" not in markup and "&lt;script&gt;" in markup and "A &amp; B" in markup
    # Only the second row (index 1) is the open one: its <tr> comes after the hostile first row's text.
    assert markup.count('<tr class="selected">') == 1
    assert markup.index('<tr class="selected">') > markup.index("&lt;script&gt;")


# --- a team's own page: matches from several competitions ------------------------------------


def test_on_a_team_page_the_links_keep_the_team_so_back_returns_to_it():
    markup = fixtures_table_html(
        [_row()], [REF], "EPL", season="2627", tz_name="Europe/London", match_details=True, team_id=7
    )
    assert _hrefs(markup) == [
        "/match_detail?league=EPL&amp;home=7&amp;away=17&amp;date=2026-10-31&amp;team=7&amp;season=2627",
        "/team_detail?league=EPL&amp;team=7&amp;season=2627&amp;tz=Europe/London",
        "/team_detail?league=EPL&amp;team=17&amp;season=2627&amp;tz=Europe/London",
        "?league=EPL&amp;team=7&amp;season=2627&amp;tz=Europe/London&amp;fixture=7-17-2026-10-31",
    ]


def test_a_row_from_another_competition_links_to_that_competitions_pages():
    elsewhere = FixtureRef(500, 501, "FC Barcelona", "Arsenal FC", DAY, league_code="UCL", match_details=True)
    own = FixtureRef(7, 17, "Chelsea", "Man United", DAY, league_code="EPL", match_details=False)
    rows = [_row(Competition="UEFA Champions League", Home="FC Barcelona", Away="Arsenal FC"), _row(Competition="Premier League")]
    markup = fixtures_table_html(rows, [elsewhere, own], "EPL", season="2627", tz_name="UTC", team_id=7)
    hrefs = _hrefs(markup)
    # The other competition: its own league code and match page, the page's season left out.
    assert "/match_detail?league=UCL&amp;home=500&amp;away=501&amp;date=2026-10-31&amp;team=7&amp;season=2627" in hrefs
    assert "/team_detail?league=UCL&amp;team=500&amp;tz=UTC" in hrefs
    assert "/team_detail?league=UCL&amp;team=501&amp;tz=UTC" in hrefs
    # The page's own league keeps the season; its ref says it has no match page.
    assert "/team_detail?league=EPL&amp;team=7&amp;season=2627&amp;tz=UTC" in hrefs
    assert not any(h.startswith("/match_detail?league=EPL") for h in hrefs)
    # The predicted-score link always reopens this team's page, whichever competition the match is in.
    assert hrefs.count("?league=EPL&amp;team=7&amp;season=2627&amp;tz=UTC&amp;fixture=500-501-2026-10-31") == 1


def test_a_mixed_table_puts_the_competition_straight_after_the_kickoff():
    row = {"Home": "A", "Away": "B", "Home win %": 50.0, "Competition": "Premier League", "Kickoff": "x"}
    assert components.fixture_columns_with_kickoff_first(row) == ["Kickoff", "Competition", "Home", "Away", "Home win %"]
    plain_row = {"Home": "A", "Away": "B", "Kickoff": "x"}
    assert components.fixture_columns_with_kickoff_first(plain_row) == ["Kickoff", "Home", "Away"]


def test_the_fixture_parameter_round_trips_and_rejects_anything_else():
    assert parse_fixture_key(fixture_key(REF)) == (7, 17, DAY)
    for bad in (None, "", "7-17", "a-b-2026-10-31", "7-17-notadate", "7-17-2026-13-45", "7--17-2026-10-31"):
        assert parse_fixture_key(bad) is None, bad


# --- the timezone travels in the URL ---------------------------------------------------------


def _timezone_app():
    import streamlit as st

    from soccer_predictor.dashboard import components

    zone = components.selected_timezone()
    st.write(zone.key)
    st.write(components.timezone_selector().key)


@pytest.mark.parametrize(
    ("param", "expected"),
    [("Europe/London", "Europe/London"), ("Not/AZone", "America/New_York"), (None, "America/New_York")],
)
def test_the_timezone_comes_from_the_link_and_defaults_when_it_is_missing_or_unknown(param, expected):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_timezone_app, default_timeout=60)
    if param:
        at.query_params["tz"] = param
    at.run()
    assert not at.exception
    assert [m.value for m in at.markdown] == [expected, expected]
    assert at.query_params["tz"] == [expected]  # written back, so the page's own links carry it


def test_choosing_a_timezone_updates_the_url():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_timezone_app, default_timeout=60).run()
    at.selectbox[0].select("UTC").run()
    assert at.query_params["tz"] == ["UTC"]
    assert [m.value for m in at.markdown] == ["UTC", "UTC"]
