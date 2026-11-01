"""The sortable HTML tables (dashboard/components.py::sortable_table_html) and the team-strength
leaderboard built on them: names that open the team's page, real ranks that survive a filter,
and escaping of everything that came from a database or an API."""

from __future__ import annotations

import re

import pandas as pd

from soccer_predictor.dashboard.components import (
    LEADERBOARD_DEFENSE_COLUMN,
    RATING_LINKS_JS,
    RATING_PARAM,
    RATING_PICK_KEY,
    SCROLL_TO_RATING_JS,
    TableCell,
    TableColumn,
    leaderboard_table_html,
    sortable_table_html,
)


def _board(*teams: tuple[int, str, float, float, float]) -> pd.DataFrame:
    """(team_id, name, attack, attack from results only, defense) as leaderboard_dataframe lays it out."""
    rows = [
        {
            "team_id": team_id,
            "Team": name,
            "Attack": attack,
            "Attack (from results only)": results_only,
            LEADERBOARD_DEFENSE_COLUMN: defense,
            "Net strength": attack - defense,
        }
        for team_id, name, attack, results_only, defense in teams
    ]
    return pd.DataFrame(rows).sort_values("Net strength", ascending=False).reset_index(drop=True)


BOARD = _board(
    (1, "Man City", 1.91, 1.75, 0.78),
    (2, "Arsenal", 1.56, 1.52, 0.64),
    (3, "Chelsea", 1.40, 1.38, 1.09),
    (4, "Spurs", 1.14, 1.18, 1.19),
)


def _rows(markup: str) -> list[list[str]]:
    body = markup.split("<tbody>")[1]
    return [re.findall(r"<td[^>]*>(.*?)</td>", row) for row in re.findall(r"<tr[^>]*>.*?</tr>", body)]


def _hrefs(markup: str) -> list[str]:
    return re.findall(r'href="([^"]*)"', markup)


def _text(cell: str) -> str:
    return re.sub("<.*?>", "", cell)


def test_each_team_name_opens_that_teams_page():
    markup = leaderboard_table_html(BOARD, "EPL", {})
    assert [href for href in _hrefs(markup) if href.startswith("/team_detail")] == [
        "/team_detail?league=EPL&amp;team=1",
        "/team_detail?league=EPL&amp;team=2",
        "/team_detail?league=EPL&amp;team=3",
        "/team_detail?league=EPL&amp;team=4",
    ]
    assert 'target="_self"' in markup


def test_the_table_starts_ranked_by_net_strength_with_signed_two_decimal_figures():
    rows = _rows(leaderboard_table_html(BOARD, "EPL", {}))
    assert [row[0] for row in rows] == ["1", "2", "3", "4"]
    assert [_text(row[2]) for row in rows] == ["Man City", "Arsenal", "Chelsea", "Spurs"]
    assert [_text(cell) for cell in rows[0][3:]] == ["1.91", "1.75", "0.78", "+1.13"]
    assert "<b>+1.13</b>" in rows[0][6]
    assert _text(rows[3][6]) == "−0.05"  # a real minus sign, like the rest of the app


def test_attack_defense_and_net_strength_each_link_to_their_own_workings():
    rows = _rows(leaderboard_table_html(BOARD, "EPL", {}))
    man_city = rows[0]
    assert [re.search(r'href="([^"]*)"', cell).group(1) for cell in man_city[3:]] == [
        "?league=EPL&amp;rating=1-attack",
        "?league=EPL&amp;rating=1-attack",  # results-only attack is the first step of the same workings
        "?league=EPL&amp;rating=1-defense",
        "?league=EPL&amp;rating=1-net",
    ]
    assert all('target="_self"' in cell and 'class="stat"' in cell for cell in man_city[3:])
    assert re.search(r"rating=4-net", rows[3][6])  # each row names its own team


def test_the_figure_links_keep_the_timezone_and_the_pages_filters():
    markup = leaderboard_table_html(
        BOARD.head(1), "EPL", {}, tz_name="Europe/London", page_params="&from=2026-11-01&to=2026-11-08"
    )
    figure_links = [href for href in _hrefs(markup) if "rating=" in href]
    assert figure_links[0] == "?league=EPL&amp;tz=Europe/London&amp;from=2026-11-01&amp;to=2026-11-08&amp;rating=1-attack"
    assert len(figure_links) == 4


def test_the_open_teams_row_and_figure_are_marked():
    markup = leaderboard_table_html(BOARD, "EPL", {}, selected=(2, "defense"))
    rows = re.findall(r"<tr[^>]*>", markup.split("<tbody>")[1])
    assert [row == '<tr class="selected">' for row in rows] == [False, True, False, False]  # Arsenal is second
    assert markup.count('class="stat on"') == 1 and re.search(r'class="stat on" href="[^"]*rating=2-defense"', markup)


def test_a_filter_that_hides_the_open_team_marks_no_row():
    markup = leaderboard_table_html(BOARD, "EPL", {}, name_filter="chel", selected=(1, "net"))
    assert 'class="selected"' not in markup and 'class="stat on"' not in markup


def test_the_open_row_is_still_the_right_one_after_a_filter():
    markup = leaderboard_table_html(BOARD, "EPL", {}, name_filter="C", selected=(3, "attack"))  # Man City, Chelsea
    rows = re.findall(r"<tr[^>]*>", markup.split("<tbody>")[1])
    assert [row == '<tr class="selected">' for row in rows] == [False, True]


def test_filtering_by_name_keeps_each_teams_real_rank():
    rows = _rows(leaderboard_table_html(BOARD, "EPL", {}, name_filter="C"))  # Man City and Chelsea
    assert [(row[0], re.sub("<.*?>", "", row[2])) for row in rows] == [("1", "Man City"), ("3", "Chelsea")]
    assert _rows(leaderboard_table_html(BOARD, "EPL", {}, name_filter="zzz")) == []


def test_numeric_columns_sort_by_value_and_defense_starts_with_the_best_lowest_first():
    markup = leaderboard_table_html(BOARD, "EPL", {})
    head = markup.split("<thead>")[1].split("</thead>")[0]
    assert 'data-sort="num" data-first="asc" title="Click to sort">#' in head
    assert re.search(r'data-sort="num" data-first="asc"[^>]*>Defense \(lower is better\)', head)
    assert re.search(r'data-sort="num" data-first="desc"[^>]*>Net strength', head)
    assert 'data-sort="text" data-first="asc"' in head  # the team column sorts A to Z first
    assert re.findall(r'data-v="([^"]*)"', markup)[:4] == ["1", "man city", "1.91", "1.75"]


def test_a_team_name_from_the_database_is_escaped():
    board = _board((1, '<img src=x onerror="alert(1)">', 1.0, 1.0, 1.0))
    markup = leaderboard_table_html(board, "EPL", {})
    assert "<img src=x" not in markup and "&lt;img src=x" in markup
    assert 'data-v="&lt;img' in markup  # the sort value too


def test_only_web_crests_are_drawn():
    markup = leaderboard_table_html(BOARD, "EPL", {1: "https://crests.example/city.png", 2: "javascript:alert(1)", 3: "data:text/html,x"})
    assert markup.count("<img") == 1 and 'src="https://crests.example/city.png"' in markup


def test_the_league_code_is_quoted_in_the_link():
    assert "league=A%26B" in leaderboard_table_html(BOARD.head(1), "A&B", {})


def test_a_cell_with_no_sort_value_sorts_after_every_real_one_on_a_numeric_column():
    columns = [TableColumn("Name", left=True, sort="text", first="asc"), TableColumn("Cut", sort="num")]
    markup = sortable_table_html(columns, [[TableCell("a", "a"), TableCell("–")], [TableCell("b", "b"), TableCell("5%", 5)]])
    assert re.findall(r'data-v="([^"]*)"', markup) == ["a", "-1", "b", "5"]


def test_a_column_that_cannot_be_sorted_has_no_sort_attributes_or_values():
    markup = sortable_table_html([TableColumn(""), TableColumn("Name", sort="text", first="asc")], [[TableCell("x"), TableCell("n", "n")]])
    assert markup.count("data-sort=") == 1 and markup.count("data-v=") == 1


def test_compact_only_when_asked():
    columns = [TableColumn("A")]
    assert 'class="fb-table"' in sortable_table_html(columns, [])
    assert 'class="fb-table compact"' in sortable_table_html(columns, [], compact=True)


def test_the_click_script_finds_the_figure_links_the_hidden_box_and_the_parameter_by_the_names_the_page_uses():
    # Plain text on both sides, so a rename on one side shows up here rather than as a dead link.
    assert "table.fb-table a.stat" in RATING_LINKS_JS
    assert f".st-key-{RATING_PICK_KEY} input" in RATING_LINKS_JS
    assert f"searchParams.get('{RATING_PARAM}')" in RATING_LINKS_JS
    assert SCROLL_TO_RATING_JS in RATING_LINKS_JS and ".st-key-rating_breakdown" in SCROLL_TO_RATING_JS
    assert 'class="stat' in leaderboard_table_html(BOARD, "EPL", {})
