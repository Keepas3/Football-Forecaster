"""The Players page's HTML tables: the player and club links, the keys that carry a pick in the
URL, and placing a stats source's club names on our own teams."""

from __future__ import annotations

import re

from soccer_predictor.dashboard.components import PLAYER_LINKS_JS, PLAYER_PARAM, PLAYER_PICK_KEY, SCROLL_TO_PLAYER_JS
from soccer_predictor.dashboard.players_table import (
    club_team_ids,
    parse_player_key,
    player_key,
    search_results_table_html,
    top_players_table_html,
)
from soccer_predictor.ingest.squad import PlayerSearchResult, SquadPlayer

TEAMS = {1: "Arsenal", 2: "Man United", 3: "Newcastle", 4: "Leeds", 5: "Real Betis"}


# --- the key in the URL ----------------------------------------------------------------------


def test_a_key_round_trips_and_a_blank_one_names_nobody():
    assert parse_player_key(player_key("Mohamed Salah", "Liverpool")) == ("Mohamed Salah", "Liverpool")
    assert parse_player_key("Haaland|") == ("Haaland", "")  # a source that gives no club
    assert parse_player_key("Salah") == ("Salah", "")  # a hand-typed link still finds someone
    assert parse_player_key("") is None and parse_player_key(None) is None and parse_player_key("|Arsenal") is None


# --- placing a source's club names on our teams ----------------------------------------------


def test_clubs_are_placed_by_name_alias_and_close_spelling():
    placed = club_team_ids(
        ["Arsenal", "Manchester United", "Newcastle United", "Leeds United", "Real Betis Balompié", "Arsenal FC"],
        TEAMS,
        aliases=[("Real Betis Balompié", 5), ("Arsenal FC", 1)],
    )
    assert placed == {
        "Arsenal": 1,
        "Manchester United": 2,  # Understat's spelling of "Man United"
        "Newcastle United": 3,
        "Leeds United": 4,
        "Real Betis Balompié": 5,
        "Arsenal FC": 1,
    }


def test_accents_and_case_do_not_matter():
    assert club_team_ids(["ARSENAL", "real betis"], TEAMS) == {"ARSENAL": 1, "real betis": 5}
    assert club_team_ids(["Atlético"], {9: "Atletico"}) == {"Atlético": 9}


def test_a_club_that_cannot_be_placed_is_left_out_rather_than_linked_to_a_lookalike():
    placed = club_team_ids(["Manchester City", "Totally Different FC", ""], TEAMS)
    assert placed == {}  # "Manchester City" is not "Man United" however alike the names look


def test_an_alias_of_a_team_that_is_not_in_the_league_is_ignored():
    assert club_team_ids(["Some Club"], TEAMS, aliases=[("Some Club", 99)]) == {}


# --- the top-players table -------------------------------------------------------------------

ROWS = [
    {"#": 1, "Player": "Bukayo Saka", "Club": "Arsenal", "Goals": 9, "Assists": 4, "G + A": 13},
    {"#": 2, "Player": "Bruno <b>Fernandes</b>", "Club": "Mystery FC", "Goals": 5, "Assists": 6, "G + A": 11},
]


def _cells(markup, row):
    body = markup.split("<tbody>")[1]
    return re.findall(r"<td[^>]*>(.*?)</td>", re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S)[row], re.S)


def test_the_top_table_links_each_player_in_place_and_each_known_club_to_its_team_page():
    markup = top_players_table_html(ROWS, "EPL", {"Arsenal": 1})
    player_cell, club_cell = _cells(markup, 0)[1], _cells(markup, 0)[2]
    assert 'class="player"' in player_cell and "Bukayo Saka" in player_cell
    assert f"&amp;{PLAYER_PARAM}=Bukayo%20Saka%7CArsenal" in player_cell  # what the click script reads
    assert 'href="/team_detail?league=EPL&amp;team=1"' in club_cell and 'target="_self"' in club_cell
    assert _cells(markup, 1)[2] == "Mystery FC"  # not placed: plain text, no link
    assert re.findall(r"<th(?: [^>]*)?>(.*?)</th>", markup) == ["#", "Player", "Club", "Goals", "Assists", "G + A"]


def test_the_top_table_escapes_names_and_sorts_by_the_numbers_not_the_text():
    markup = top_players_table_html(ROWS, "EPL", {})
    assert "<b>Fernandes</b>" not in markup and "Bruno &lt;b&gt;Fernandes&lt;/b&gt;" in markup
    first = re.findall(r"<td[^>]*>", markup.split("<tbody>")[1])[:6]
    assert 'data-v="1"' in first[0] and 'data-v="9"' in first[3] and 'data-v="13"' in first[5]
    assert 'data-sort="num"' in markup and 'data-sort="text"' in markup


def test_a_salary_or_rate_column_sorts_as_a_number_and_a_dash_sorts_last():
    rows = [
        {"#": 1, "Player": "A", "Club": "X", "Goals": 1, "Assists": 0, "Salary (guaranteed)": "$1,200,000"},
        {"#": 2, "Player": "B", "Club": "X", "Goals": 1, "Assists": 0, "Salary (guaranteed)": "–"},
    ]
    markup = top_players_table_html(rows, "MLS", {})
    assert 'data-v="1200000"' in markup and 'data-v="-1"' in markup
    assert "$" not in markup  # written as &#36; so a table in st.markdown never reads two of them as a formula
    assert "&#36;1,200,000" in markup


def test_the_picked_player_is_marked_in_the_row_and_the_name():
    selected = player_key("Bukayo Saka", "Arsenal")
    markup = top_players_table_html(ROWS, "EPL", {}, selected=selected)
    assert markup.count('<tr class="selected">') == 1 and 'class="player on"' in markup
    nobody = top_players_table_html(ROWS, "EPL", {}, selected="Nobody|Nowhere")
    assert 'class="selected"' not in nobody and 'class="player on"' not in nobody


def test_an_empty_list_still_draws_a_table():
    assert "<tbody></tbody>" in top_players_table_html([], "EPL", {})


# --- the search results ----------------------------------------------------------------------


def _result(team_id, team, name, position="Forward"):
    return PlayerSearchResult(team_id, team, SquadPlayer(name, position, "England", None))


def test_search_results_link_the_player_and_the_club_by_the_squad_they_came_from():
    results = [_result(1, "Arsenal", "Bukayo Saka", "Winger"), _result(2, "Man United", "Rasmus H&ojlund")]
    markup = search_results_table_html(results, "EPL", selected=player_key("Rasmus H&ojlund", "Man United"))
    first, second = _cells(markup, 0), _cells(markup, 1)
    assert "Bukayo%20Saka%7CArsenal" in first[0] and first[1] == "Winger"
    assert 'href="/team_detail?league=EPL&amp;team=1"' in first[2]
    assert "H&amp;ojlund" in second[0]  # escaped, in the link text and the URL
    assert markup.count('<tr class="selected">') == 1


def test_the_click_script_looks_for_the_players_link_and_box():
    assert "closest('table.fb-table a.player')" in PLAYER_LINKS_JS
    assert f".st-key-{PLAYER_PICK_KEY} input" in PLAYER_LINKS_JS
    assert f"searchParams.get('{PLAYER_PARAM}')" in PLAYER_LINKS_JS and f"__fbInPlace_{PLAYER_PARAM}" in PLAYER_LINKS_JS
    assert SCROLL_TO_PLAYER_JS in PLAYER_LINKS_JS and "player_detail" in SCROLL_TO_PLAYER_JS
    for placeholder in ("SELECTOR", "BOXKEY", "SCROLL", "PARAM"):
        assert placeholder not in PLAYER_LINKS_JS
