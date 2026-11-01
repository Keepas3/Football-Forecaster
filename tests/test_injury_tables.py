"""The Injuries tab's HTML tables (dashboard/injury_tables.py): team links, the players-listed link that
opens a team's injured players in place, and the formatting the old grid did with column configs."""

from __future__ import annotations

import datetime as dt
import re

from soccer_predictor.dashboard import injury_table as it
from soccer_predictor.dashboard.components import INJURY_LINKS_JS, INJURY_PARAM, INJURY_PICK_KEY, SCROLL_TO_INJURY_JS
from soccer_predictor.dashboard.injury_tables import (
    injury_players_table_html,
    parse_injury_team,
    team_impact_table_html,
)
from soccer_predictor.model.injury_adjustment import InjuryEntry
from soccer_predictor.prediction.service import TeamInjuries

TODAY = dt.date(2026, 10, 29)
IDS = {"Arsenal": 10, "Aston Villa": 20}


def _team(entries, sources, notes, returns=None):
    return TeamInjuries(
        entries=entries, sources=sources, expected_return_dates=returns or [None] * len(entries), notes=notes
    )


ARSENAL = _team(
    [InjuryEntry("Saka", "attack", 0.54), InjuryEntry("Saliba", "defense", 0.10), InjuryEntry("Rice", "attack", 0.03)],
    ["api", "api", "api"],
    ["FPL: Thigh injury - Unknown return date", "FPL: Hip injury - 75% chance of playing", "FPL: Suspended until 29 Nov"],
)
VILLA = _team(
    [InjuryEntry("Onana <b>", "attack", 0.10), InjuryEntry("Watkins", "attack", 0.30)],
    ["manual", "api"],
    ["Knee", "Day-To-Day"],
    [dt.date(2027, 1, 5), None],
)


def _frame():
    return it.players_frame(it.player_rows("Arsenal", ARSENAL, TODAY) + it.player_rows("Aston Villa", VILLA, TODAY))


def _cells(markup, row):
    body = markup.split("<tbody>")[1]
    return re.findall(r"<td[^>]*>(.*?)</td>", re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S)[row], re.S)


def _plain(cell):
    return re.sub(r"<[^>]+>", "", cell)


def _headings(markup):
    return re.findall(r"<th(?: [^>]*)?>(.*?)</th>", markup)


# --- the players table -----------------------------------------------------------------------


def test_each_team_name_opens_its_page_and_a_team_we_cannot_place_is_plain_text():
    markup = injury_players_table_html(_frame(), IDS, "EPL")
    assert 'href="/team_detail?league=EPL&amp;team=10"' in _cells(markup, 0)[0]
    assert 'href="/team_detail?league=EPL&amp;team=20"' in _cells(markup, 3)[0]
    assert _cells(injury_players_table_html(_frame(), {}, "EPL"), 0)[0] == "Arsenal"


def test_each_player_name_opens_their_stats_on_the_players_page_for_their_own_team():
    markup = injury_players_table_html(_frame(), IDS, "EPL")
    saka = _cells(markup, 0)[1]
    assert saka == '<a class="profile" href="/players?league=EPL&amp;player=Saka%7CArsenal" target="_self">Saka</a>'
    # the club in the key is our own team name, so the Players page looks them up at that club
    assert "player=Watkins%7CAston%20Villa" in _cells(markup, 3)[1]
    # a name from a feed is escaped in the link text and encoded in the address
    onana = _cells(markup, 4)[1]
    assert ">Onana &lt;b&gt;</a>" in onana and "player=Onana%20%3Cb%3E%7CAston%20Villa" in onana
    assert "<b>" not in onana


def test_a_single_teams_list_links_its_players_too():
    team = _frame()[lambda f: f["Team"] == "Arsenal"]
    markup = injury_players_table_html(team, IDS, "EPL", show_team=False)
    assert 'href="/players?league=EPL&amp;player=Saka%7CArsenal"' in _cells(markup, 0)[0]


def test_the_players_table_formats_what_the_old_grid_did_with_column_configs():
    markup = injury_players_table_html(_frame(), IDS, "EPL")
    assert _headings(markup) == [
        "Team", "Player", "Status", "Problem", "Chance of playing", "Expected back", "Position", "Impact", "Source",
    ]
    saka, saliba, rice = (_cells(markup, row) for row in range(3))  # biggest impact first
    assert [_plain(saka[1])] + saka[2:5] == ["Saka", "Out", "Thigh injury", "0%"] and saka[5] == "–"
    assert saliba[4] == "75%" and saliba[2] == "Doubtful"
    assert rice[5] == "29 Nov 2026"  # the return date, as "D MMM YYYY"
    assert '<i style="width:54%"></i></span>0.54' in saka[7] and saka[8] == "Automatic feed"
    assert _cells(markup, 4)[5] == "5 Jan 2027" and _cells(markup, 4)[8] == "Manual list"  # Onana, after Watkins


def test_a_missing_chance_or_date_shows_a_dash_and_sorts_last_where_it_should():
    markup = injury_players_table_html(_frame(), IDS, "EPL")
    day_to_day = markup.split("<tbody>")[1].split(">Watkins<")[1].split("</tr>")[0]
    assert "<td data-v=\"-1\">–</td>" in day_to_day  # no chance given: after every real one, biggest first
    assert f'data-v="{10**7}">–' in markup  # no return date: after every real one, soonest first
    assert f'data-v="{dt.date(2026, 11, 29).toordinal()}">29 Nov 2026' in markup


def test_each_table_has_one_style_block_so_the_markdown_never_prints_css_as_text():
    # A <style> block ends the markdown's HTML block at its first </style>: a second one in front of
    # the shared stylesheet left the rest of the CSS showing on the page.
    for markup in (
        injury_players_table_html(_frame(), IDS, "EPL"),
        team_impact_table_html(_impact(), IDS, "EPL"),
    ):
        assert markup.startswith("<style>") and markup.count("<style>") == 1 and markup.count("</style>") == 1
        assert ".impact{" in markup


def test_names_from_a_feed_are_escaped():
    markup = injury_players_table_html(_frame(), IDS, "EPL")
    assert "Onana &lt;b&gt;" in markup and "Onana <b>" not in markup


def test_a_single_teams_list_has_no_team_column():
    team = _frame()[lambda f: f["Team"] == "Arsenal"]
    markup = injury_players_table_html(team, IDS, "EPL", show_team=False)
    assert _headings(markup)[0] == "Player" and "Team" not in _headings(markup)
    assert len(_cells(markup, 0)) == 8


def test_an_empty_frame_still_draws_a_table():
    assert "<tbody></tbody>" in injury_players_table_html(it.players_frame([]), IDS, "EPL")


# --- the team-impact table -------------------------------------------------------------------


def _impact():
    return it.team_impact_frame(
        {
            "Arsenal": (1.5, 0.64, ARSENAL.entries),
            "Aston Villa": (1.2, 1.0, VILLA.entries),
        }
    )


def test_the_team_impact_table_has_the_old_columns_and_a_signed_percentage_for_each_change():
    markup = team_impact_table_html(_impact(), IDS, "EPL")
    assert _headings(markup) == [
        "Team", "Players listed", "Attack", "Attack with injuries", "Attack change",
        "Defence (goals conceded)", "Defence with injuries", "Defence change",
    ]
    arsenal = _cells(markup, 0)
    assert 'href="/team_detail?league=EPL&amp;team=10"' in arsenal[0]
    assert arsenal[2] == "1.50" and re.fullmatch(r"−\d+\.\d%", arsenal[4]) and re.fullmatch(r"\+\d+\.\d%", arsenal[7])


def test_each_players_listed_figure_opens_that_teams_injured_players_in_place():
    markup = team_impact_table_html(_impact(), IDS, "EPL", tz_name="Europe/London", page_params="&from=2026-11-01&to=2026-11-08")
    arsenal = _cells(markup, 0)[1]
    assert 'class="injured"' in arsenal and arsenal.endswith(">3</a>")
    href = re.search(r'href="([^"]*)"', arsenal).group(1)
    # what the click script reads, with the page's timezone and date range carried through a reload
    assert href == f"?league=EPL&amp;tz=Europe/London&amp;from=2026-11-01&amp;to=2026-11-08&amp;{INJURY_PARAM}=10"
    assert f"{INJURY_PARAM}=20" in _cells(markup, 1)[1]


def test_the_open_team_is_marked_in_its_row_and_its_figure():
    markup = team_impact_table_html(_impact(), IDS, "EPL", selected=20)
    assert markup.count('<tr class="selected">') == 1 and 'class="injured on"' in markup
    rows = re.findall(r"<tr[^>]*>", markup.split("<tbody>")[1])
    assert rows == ["<tr>", '<tr class="selected">']  # Arsenal is hit hardest, so it comes first
    assert 'class="selected"' not in team_impact_table_html(_impact(), IDS, "EPL")


def test_a_team_we_cannot_place_has_a_plain_count_not_a_link():
    markup = team_impact_table_html(_impact(), {"Arsenal": 10}, "EPL")
    assert _cells(markup, 1)[0] == "Aston Villa" and _cells(markup, 1)[1] == "2"


# --- the link in the URL and the script that follows it ------------------------------------


def test_the_team_in_the_url_must_be_one_with_someone_listed():
    listed = {"Arsenal", "Aston Villa"}
    assert parse_injury_team("10", IDS, listed) == "Arsenal"
    assert parse_injury_team("10", IDS, {"Aston Villa"}) is None  # nobody listed for it now
    for junk in ("", "x", "10.5", "99", None, "-10"):
        assert parse_injury_team(junk, IDS, listed) is None


def test_the_click_script_looks_for_the_players_listed_link_and_box():
    assert "closest('table.fb-table a.injured')" in INJURY_LINKS_JS
    assert f".st-key-{INJURY_PICK_KEY} input" in INJURY_LINKS_JS
    assert f"searchParams.get('{INJURY_PARAM}')" in INJURY_LINKS_JS and f"__fbInPlace_{INJURY_PARAM}" in INJURY_LINKS_JS
    assert SCROLL_TO_INJURY_JS in INJURY_LINKS_JS and "injury_breakdown" in SCROLL_TO_INJURY_JS
    for placeholder in ("SELECTOR", "BOXKEY", "SCROLL", "PARAM"):
        assert placeholder not in INJURY_LINKS_JS
