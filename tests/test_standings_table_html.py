"""The standings table (HTML, so each form square can have its own hover text)."""

from __future__ import annotations

import datetime as dt
import re

import pandas as pd

from soccer_predictor.config import TableZone
from soccer_predictor.dashboard.components import form_game_tooltip, standings_table_html
from soccer_predictor.model.standings import FormGame, TeamStanding, compute_standings

ZONES = [
    TableZone(start=1, end=4, kind="qualify", label="Champions League"),
    TableZone(start=6, end=6, kind="playoff", label="Relegation playoff"),
    TableZone(start=7, end=8, kind="relegation", label="Relegation"),
]


def _standing(team_id: int, name: str, points: int = 0) -> TeamStanding:
    s = TeamStanding(team_id=team_id, team_name=name)
    s.won = points // 3
    return s


def _rows(markup):
    return re.findall(r"<tr>.*?</tr>", markup.split("<tbody>")[1])


def _game(home="Spain", away="Argentina", hg=1, ag=0, day=dt.date(2026, 10, 5)):
    return FormGame(day, home, away, hg, ag)


# --- zones --------------------------------------------------------------------------------


def test_pos_is_colored_by_zone():
    standings = [_standing(i, f"Team {i}") for i in range(1, 9)]
    rows = _rows(standings_table_html(standings, {}, {}, zones=ZONES))

    assert "#2e7d32" in rows[0] and "#2e7d32" in rows[3]  # qualify -> green
    assert "color" not in rows[4].split("</td>")[0]  # 5th place: no zone
    assert "#f9a825" in rows[5]  # playoff -> amber
    assert "#c62828" in rows[6] and "#c62828" in rows[7]  # relegation -> red


def test_no_zones_means_no_colors():
    markup = standings_table_html([_standing(1, "Team 1")], {}, {})
    assert "#2e7d32" not in markup and "#f9a825" not in markup and "#c62828" not in markup


# --- form hover ----------------------------------------------------------------------------


def test_each_form_square_shows_the_game_behind_it_on_hover():
    team = _standing(1, "Spain")
    team.form = ["W", "L"]
    team.form_games = [_game("Spain", "Argentina", 1, 0), _game("France", "Spain", 2, 0, dt.date(2026, 10, 9))]

    markup = standings_table_html([team], {}, {})

    assert 'title="Spain vs Argentina · Score 1-0 · Oct 5 (Win)">🟩' in markup
    assert 'title="France vs Spain · Score 2-0 · Oct 9 (Loss)">🟥' in markup


def test_tooltip_wording_for_each_result():
    assert form_game_tooltip(_game(hg=1, ag=0), "W") == "Spain vs Argentina · Score 1-0 · Oct 5 (Win)"
    assert form_game_tooltip(_game(hg=2, ag=2), "D") == "Spain vs Argentina · Score 2-2 · Oct 5 (Draw)"
    assert form_game_tooltip(_game(hg=0, ag=3), "L") == "Spain vs Argentina · Score 0-3 · Oct 5 (Loss)"


def test_a_missing_date_is_just_left_out_of_the_tooltip():
    assert form_game_tooltip(_game(day=None), "W") == "Spain vs Argentina · Score 1-0 (Win)"


def test_form_squares_still_render_when_no_games_are_known():
    team = _standing(1, "Spain")
    team.form = ["W", "D"]
    body = standings_table_html([team], {}, {}).split("<tbody>")[1]
    assert "🟩" in body and "⬜" in body and "title=" not in body


def test_compute_standings_records_the_game_behind_each_form_square():
    matches = pd.DataFrame(
        [
            (dt.date(2026, 10, 1), "2627", 1, 2, 1, 0),
            (dt.date(2026, 10, 5), "2627", 3, 1, 2, 2),
        ],
        columns=["date", "season", "home_team_id", "away_team_id", "home_goals", "away_goals"],
    )
    names = {1: "Spain", 2: "Argentina", 3: "France"}

    spain = next(s for s in compute_standings(matches, names, "2627") if s.team_id == 1)

    assert spain.form == ["W", "D"]
    assert [(g.home_name, g.away_name, g.home_goals, g.away_goals) for g in spain.form_games] == [
        ("Spain", "Argentina", 1, 0),
        ("France", "Spain", 2, 2),
    ]


def test_form_games_are_trimmed_to_the_same_length_as_form():
    rows = [(dt.date(2026, 9, d), "2627", 1, 2, 1, 0) for d in range(1, 12)]
    matches = pd.DataFrame(rows, columns=["date", "season", "home_team_id", "away_team_id", "home_goals", "away_goals"])
    spain = next(s for s in compute_standings(matches, {1: "Spain", 2: "X"}, "2627") if s.team_id == 1)
    assert len(spain.form) == len(spain.form_games) == 7
    assert spain.form_games[-1].date == dt.date(2026, 9, 11)


# --- table content --------------------------------------------------------------------------


def test_the_team_name_links_to_its_page():
    markup = standings_table_html([_standing(412, "Greece")], {}, {}, league_code="NL")
    assert '<a class="team" href="/team_detail?league=NL&amp;team=412" target="_self">Greece</a>' in markup


def test_team_links_carry_the_season_so_back_returns_to_it():
    with_season = standings_table_html([_standing(412, "Greece")], {}, {}, league_code="NL", season="2223")
    assert "team=412&amp;season=2223" in with_season
    assert "season" not in standings_table_html([_standing(412, "Greece")], {}, {}, league_code="NL")


def test_columns_and_numbers():
    team = _standing(1, "Spain", points=6)
    team.played, team.drawn, team.lost, team.goals_for, team.goals_against = 4, 0, 2, 7, 3
    markup = standings_table_html([team], {}, {})
    for header in ("Pos", "Team", "Pl", "W", "D", "L", "GF", "GA", "GD", "Pts", "Form", "Next", "Search"):
        assert f"{header}</th>" in markup
    row = _rows(markup)[0]
    for number in (4, 7, 3):
        assert f">{number}</td>" in row
    assert "<b>6</b>" in row


# --- click-to-sort data ----------------------------------------------------------------------


def _headers(markup):
    return {re.sub("<.*?>", "", h): h for h in re.findall(r"<th.*?</th>", markup)}


def test_stat_headers_are_sortable_and_next_and_search_are_not():
    heads = _headers(standings_table_html([_standing(1, "Spain")], {}, {}))
    for name in ("Pos", "Team", "Pl", "W", "D", "L", "GF", "GA", "GD", "Pts", "Form"):
        assert "data-sort=" in heads[name], name
    assert "data-sort" not in heads["Next"] and "data-sort" not in heads["Search"]


def test_first_click_direction_is_ascending_for_pos_and_team_and_highest_first_for_stats():
    heads = _headers(standings_table_html([_standing(1, "Spain")], {}, {}))
    assert 'data-first="asc"' in heads["Pos"] and 'data-first="asc"' in heads["Team"]
    for name in ("Pl", "W", "D", "L", "GF", "GA", "GD", "Pts", "Form"):
        assert 'data-first="desc"' in heads[name], name
    assert 'data-sort="text"' in heads["Team"] and 'data-sort="num"' in heads["Pts"]


def test_cells_carry_the_value_to_sort_by():
    team = _standing(7, "Côte d'Ivoire", points=7)
    team.played, team.won, team.drawn, team.lost = 4, 2, 1, 1
    team.goals_for, team.goals_against = 5, 3
    team.form = ["W", "D", "L", "W"]
    row = _rows(standings_table_html([team, _standing(8, "B")], {}, {}))[0]

    cells = re.findall(r'data-v="([^"]*)"', row)
    # Pos, Team (lowercased), Pl, W, D, L, GF, GA, GD, Pts, Form (3 per win + 1 per draw)
    assert cells == ["1", "côte d&#x27;ivoire", "4", "2", "1", "1", "5", "3", "2", "7", "7"]


def test_a_negative_goal_difference_sorts_as_a_negative_number():
    team = _standing(1, "Spain")
    team.goals_for, team.goals_against = 1, 4
    assert 'data-v="-3"' in _rows(standings_table_html([team], {}, {}))[0]


def test_sort_values_are_html_escaped():
    row = _rows(standings_table_html([_standing(1, '"><script>alert(1)</script>')], {}, {}))[0]
    assert "<script>" not in row and 'data-v=""' not in row


def test_the_sort_script_is_static_and_installs_itself_once():
    from soccer_predictor.dashboard.components import STANDINGS_SORT_JS

    assert "__fbSortInstalled" in STANDINGS_SORT_JS  # guard: a rerun doesn't stack listeners
    assert "table.fb-table th[data-sort]" in STANDINGS_SORT_JS  # only ever acts on these headers
    # No placeholders/formatting: nothing from the database can end up inside the script.
    assert "{" in STANDINGS_SORT_JS and "%s" not in STANDINGS_SORT_JS and "{{" not in STANDINGS_SORT_JS


def test_search_filter_keeps_each_teams_real_position():
    standings = [_standing(1, "Spain", 9), _standing(2, "France", 6), _standing(3, "Spain B", 3)]
    markup = standings_table_html(standings, {}, {}, name_filter="spa")
    rows = _rows(markup)
    assert len(rows) == 2 and ">1</td>" in rows[0] and ">3</td>" in rows[1]
    assert "France" not in markup


def test_next_label_and_search_link():
    markup = standings_table_html(
        [_standing(1, "Spain")], {}, {1: "Norway · Oct 06"}, match_links={1: "https://www.google.com/search?q=a+b"}
    )
    assert "Norway · Oct 06" in markup
    assert 'href="https://www.google.com/search?q=a+b" target="_blank" rel="noopener noreferrer"' in markup


def test_no_search_link_cell_when_no_match_is_scheduled():
    assert 'class="search"' not in standings_table_html([_standing(1, "Spain")], {}, {})


def test_crest_is_shown_only_for_plain_web_urls():
    ok = standings_table_html([_standing(1, "Spain")], {1: "https://crests.example/es.png"}, {})
    assert '<img class="crest" src="https://crests.example/es.png"' in ok
    bad = standings_table_html([_standing(1, "Spain")], {1: "javascript:alert(1)"}, {})
    assert "javascript" not in bad and "<img" not in bad.split("<tbody>")[1]


def test_everything_from_the_database_is_escaped():
    team = _standing(1, '<img src=x onerror=alert(1)>"')
    team.form = ["W"]
    team.form_games = [FormGame(dt.date(2026, 10, 5), "<script>alert(2)</script>", 'A"B', 1, 0)]
    markup = standings_table_html([team], {}, {1: "<b>x</b>"}, league_code='"><script>')

    body = markup.split("<tbody>")[1]
    assert "<script>" not in body and "<img src=x" not in body and "<b>x</b>" not in body
    assert 'onerror=alert(1)>"' not in body


def test_the_table_has_no_blank_or_indented_lines_so_markdown_does_not_split_it():
    markup = standings_table_html([_standing(1, "Spain"), _standing(2, "France")], {}, {})
    # A blank line ends a markdown HTML block, and a line indented 4+ spaces
    # becomes a code block; the style sheet's own lines are unindented and the
    # table itself is a single line.
    assert "\n\n" not in markup
    assert not any(line.startswith("    ") for line in markup.split("\n"))


def test_the_sort_arrows_are_the_real_characters_not_a_mangled_escape():
    # A backslash-25B2 escape written inside a normal Python string is an octal
    # escape (chr(21) + "B2"), which the browser showed as a missing-glyph box
    # next to "BC".
    css = standings_table_html([_standing(1, "Spain")], {}, {}).split("</style>")[0]
    assert 'content:" ▲"' in css and 'content:" ▼"' in css
    assert not any(ord(ch) < 32 and ch != chr(10) for ch in css)  # no stray control characters
