"""The team page's Squad strength (dashboard/squad_strength.py): what each player counts for in the
team's attack and defence, the usual XI to draw, the team's net strength and its recent results."""

from __future__ import annotations

import datetime as dt
import re

import pandas as pd
import pytest

from soccer_predictor.dashboard import squad_strength as ss
from soccer_predictor.dashboard.components import LEADERBOARD_DEFENSE_COLUMN
from soccer_predictor.model.injury_adjustment import MAX_MULTIPLIER, MIN_MULTIPLIER
from soccer_predictor.model.lineup_adjustment import LINEUP_ATTACK_FACTOR, LINEUP_DEFENSE_FACTOR, ExpectedPlayer


def _player(name, position="attack", weight=0.5, minutes=900.0, role="MID", goals=0, assists=0, keeper=False):
    return ExpectedPlayer(name, position, weight, minutes, keeper, role, goals, assists)


def _squad(extra_outfield=1):
    """A keeper, ten usual outfield players and `extra_outfield` fringe players."""
    squad = [_player("Keeper", "defense", 0.9, 3000, "GK", keeper=True)]
    squad += [_player(f"Regular {i}", "attack", 0.3, 2000 - i * 10, "MID") for i in range(10)]
    squad += [_player(f"Fringe {i}", "attack", 0.1, 100 - i, "FWD") for i in range(extra_outfield)]
    return squad


# --- what each player is worth ---------------------------------------------------------------------


def test_an_absent_attacker_costs_the_attack_and_an_absent_defender_the_defence():
    squad = [_player("Striker", "attack", 0.6, role="FWD", goals=5, assists=2), _player("Centre back", "defense", 0.8, role="DEF")]
    rows = {row.name: row for row in ss.player_strengths(squad)}
    assert rows["Striker"].attack_cut == pytest.approx(0.6 * LINEUP_ATTACK_FACTOR) and rows["Striker"].defense_rise is None
    assert rows["Centre back"].defense_rise == pytest.approx(0.8 * LINEUP_DEFENSE_FACTOR) and rows["Centre back"].attack_cut is None
    assert (rows["Striker"].goals, rows["Striker"].assists, rows["Striker"].role) == (5, 2, "FWD")


def test_the_figures_stop_where_the_model_stops_them():
    assert ss.attack_cut(5.0) == pytest.approx(1 - MIN_MULTIPLIER)
    assert ss.defense_rise(50.0) == pytest.approx(MAX_MULTIPLIER - 1)


def test_regulars_come_first_and_only_the_usual_xi_is_starred():
    rows = ss.player_strengths(_squad())
    assert [row.name for row in rows][:2] == ["Keeper", "Regular 0"]
    assert rows[-1].name == "Fringe 0" and not rows[-1].in_xi
    assert sum(row.in_xi for row in rows) == 11


def test_the_usual_xi_goes_to_the_pitch_most_important_first_with_what_he_costs_the_team():
    squad = [
        _player("Keeper", "defense", 0.9, 3000, "GK", keeper=True),
        _player("Centre back", "defense", 0.8, 2900, "DEF"),
        _player("Striker", "attack", 0.6, 2800, "FWD"),
        _player("Mystery", "attack", 0.2, 2700, ""),
    ]
    slots = ss.usual_xi_slots(squad)
    assert [(s.name, s.role, s.figure, s.defensive) for s in slots] == [
        ("Keeper", "GK", "+9%", True),
        ("Centre back", "DEF", "+8%", True),
        ("Striker", "FWD", "−18%", False),
        ("Mystery", "MID", "−6%", False),  # a position nobody recognises is drawn with the midfielders
    ]


def test_a_fringe_player_is_not_on_the_pitch():
    assert "Fringe 0" not in [s.name for s in ss.usual_xi_slots(_squad())]
    assert len(ss.usual_xi_slots(_squad())) == 11


# --- the table ------------------------------------------------------------------------------------


def _cells(markup: str) -> list[list[str]]:
    return [re.findall(r"<td[^>]*>(.*?)</td>", row) for row in re.findall(r"<tr>.*?</tr>", markup.split("<tbody>")[1])]


def test_the_squad_table_has_a_row_per_player_with_the_cut_in_the_column_that_counts():
    squad = [
        _player("Striker", "attack", 0.6, 900, "FWD", goals=7, assists=2),
        _player("Keeper", "defense", 0.9, 1000, "GK", keeper=True),
    ]
    markup = ss.squad_table_html(ss.player_strengths(squad))
    keeper, striker = _cells(markup)
    assert [re.sub("<.*?>", "", c) for c in keeper] == ["Keeper", "GK", "★", "1,000", "0", "0", "–", "+9.0%"]
    assert [re.sub("<.*?>", "", c) for c in striker] == ["Striker", "FWD", "★", "900", "7", "2", "−18.0%", "–"]
    assert 'class="fb-table compact"' in markup


def test_a_cell_the_model_does_not_use_sorts_last_and_a_cut_sorts_by_its_size():
    squad = [_player("A", "attack", 0.6, role="FWD"), _player("B", "defense", 0.5, role="DEF")]
    markup = ss.squad_table_html(ss.player_strengths(squad))
    values = re.findall(r'data-v="([^"]*)"', markup)
    assert values[6:8] == ["18.0", "-1"] and values[14:16] == ["-1", "5.0"]


def test_a_player_name_from_the_stats_source_is_escaped():
    markup = ss.squad_table_html(ss.player_strengths([_player("<script>alert(1)</script>")]))
    assert "<script>" not in markup and "&lt;script&gt;" in markup


def test_a_player_with_no_known_position_shows_a_dash():
    assert _cells(ss.squad_table_html(ss.player_strengths([_player("Sub", role="")])))[0][1] == "–"


# --- the team's own line --------------------------------------------------------------------------


def _board():
    rows = [
        {"team_id": 7, "Team": "Strong", "Attack": 1.9, "Attack (from results only)": 1.8, LEADERBOARD_DEFENSE_COLUMN: 0.8, "Net strength": 1.1},
        {"team_id": 3, "Team": "Weak", "Attack": 0.8, "Attack (from results only)": 0.9, LEADERBOARD_DEFENSE_COLUMN: 1.3, "Net strength": -0.5},
    ]
    return pd.DataFrame(rows)


def test_the_teams_strength_and_rank_come_from_its_leaderboard_row():
    strong, weak = ss.team_strength(_board(), 7), ss.team_strength(_board(), 3)
    assert (strong.rank, strong.attack, strong.attack_results_only, strong.defense, strong.net) == (1, 1.9, 1.8, 0.8, 1.1)
    assert weak.rank == 2 and weak.net == -0.5
    assert ss.team_strength(_board(), 99) is None


def test_the_strength_line_names_the_net_strength_and_the_rank():
    assert ss.strength_line(ss.team_strength(_board(), 7)) == (
        "**Net strength +1.10** (attack minus defense) · 1st on the Predictor's Team Strength Leaderboard"
    )
    assert "**Net strength −0.50**" in ss.strength_line(ss.team_strength(_board(), 3))


@pytest.mark.parametrize(
    ("number", "text"),
    [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"), (111, "111th")],
)
def test_ordinals(number, text):
    assert ss.ordinal(number) == text


# --- recent results -------------------------------------------------------------------------------

NAMES = {1: "Us", 2: "Them", 3: "<b>Odd</b> FC"}


def _matches(*games):
    """(day, home, away, home goals, away goals) in any order."""
    rows = [(dt.date(2026, 10, day), home, away, hg, ag) for day, home, away, hg, ag in games]
    return pd.DataFrame(rows, columns=["date", "home_team_id", "away_team_id", "home_goals", "away_goals"])


def test_recent_form_is_the_last_results_oldest_first_from_the_teams_own_side():
    matches = _matches((20, 2, 1, 3, 1), (1, 1, 2, 2, 0), (10, 1, 3, 1, 1), (15, 3, 1, 0, 2))
    form = ss.recent_form(matches, 1, NAMES)
    assert [result for result, _ in form] == ["W", "D", "W", "L"]  # a win at home, a draw, a win away, a loss
    assert [game.date.day for _, game in form] == [1, 10, 15, 20]


def test_recent_form_keeps_only_the_latest_few():
    matches = _matches(*[(day, 1, 2, 1, 0) for day in range(1, 11)])
    form = ss.recent_form(matches, 1, NAMES, length=3)
    assert [game.date.day for _, game in form] == [8, 9, 10]
    assert ss.recent_form(matches.iloc[0:0], 1, NAMES) == []


def test_the_form_squares_say_which_game_each_is_and_the_record():
    form = ss.recent_form(_matches((1, 1, 2, 2, 0), (2, 3, 1, 1, 1), (3, 2, 1, 4, 0)), 1, NAMES)
    markup = ss.form_html(form)
    assert markup.count("<span") == 3
    assert "Us vs Them · Score 2-0 · Oct 1 (Win)" in markup
    assert markup.endswith("1W 1D 1L")


def test_a_team_name_in_a_form_square_is_escaped():
    markup = ss.form_html(ss.recent_form(_matches((1, 3, 1, 0, 0)), 1, NAMES))
    assert "<b>Odd" not in markup and "&lt;b&gt;Odd" in markup
