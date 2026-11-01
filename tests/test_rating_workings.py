"""The workings panel under the Predictor's leaderboard: what each of a team's attack, defense and
net strength figures is built from. The builders are plain functions of the numbers, so these
tests pin the figures and the sentences that carry them."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from soccer_predictor.dashboard.components import LEADERBOARD_DEFENSE_COLUMN, parse_rating_key, rating_key
from soccer_predictor.dashboard.rating_workings import (
    RatingWorkings,
    SeasonRecord,
    attack_workings,
    defense_workings,
    net_workings,
    season_record,
)
from soccer_predictor.dashboard.squad_strength import TeamStrength
from soccer_predictor.ingest.player_importance import CurrentAttackDetail
from soccer_predictor.model.current_form_adjustment import adjust_for_current_attack_form
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.model.lineup_adjustment import ExpectedPlayer
from soccer_predictor.model.standings import TeamStanding

PARAMS = DixonColesParams(
    league_code="EPL",
    attack={1: 1.4},
    defense={1: 0.8},
    home_advantage=1.3,
    rho=-0.1,
    xi=0.0018,
    fitted_at="2026-10-01T00:00:00",
    n_matches=1234,
)


def _text(workings: RatingWorkings) -> str:
    parts = [workings.heading]
    for step in workings.steps:
        parts += [step.heading, *step.paragraphs]
    return "\n".join(parts)


def _detail(ratio: float) -> CurrentAttackDetail:
    return CurrentAttackDetail(team_rate=ratio * 2.0, league_mean_rate=2.0, matches_played=10, teams_compared=20, ratio=ratio)


def _strength(ratio: float | None = 1.2, base: float = 1.4, defense: float = 0.8, rank: int | None = 1) -> TeamStrength:
    attack = adjust_for_current_attack_form(base, ratio)
    return TeamStrength(attack=attack, attack_results_only=base, defense=defense, net=attack - defense, rank=rank)


def _rate_player(name: str, xg: float, xa: float) -> ExpectedPlayer:
    return ExpectedPlayer(name, "attack", 0.3, 1500, False, "FWD", 0, 0, xg, xa)


RECORD = SeasonRecord(played=10, scored_per_game=2.0, conceded_per_game=1.0, league_per_game=1.5)


# ---- the rating= URL parameter ----


def test_a_rating_key_round_trips():
    for kind in ("attack", "defense", "net"):
        assert parse_rating_key(rating_key(42, kind)) == (42, kind)


@pytest.mark.parametrize("text", [None, "", "x-attack", "3-speed", "3", "-3-attack", "0-net", "3-", "attack-3", "3-attack-net"])
def test_anything_that_is_not_a_rating_key_is_rejected(text):
    assert parse_rating_key(text) is None


# ---- the season's record, for the sanity check against the fit ----


def test_the_season_record_compares_the_team_with_the_leagues_average_side():
    standings = [
        TeamStanding(1, "A", played=10, goals_for=20, goals_against=10),
        TeamStanding(2, "B", played=10, goals_for=10, goals_against=20),
    ]
    record = season_record(standings, 1)
    assert record == SeasonRecord(played=10, scored_per_game=2.0, conceded_per_game=1.0, league_per_game=1.5)


def test_there_is_no_record_for_a_team_that_has_not_played_or_is_not_listed():
    standings = [TeamStanding(1, "A", played=0), TeamStanding(2, "B", played=4, goals_for=4, goals_against=4)]
    assert season_record(standings, 1) is None
    assert season_record(standings, 99) is None
    assert season_record([], 1) is None


# ---- attack ----


def test_attack_walks_from_the_fitted_rating_to_the_adjusted_one_with_the_real_figures():
    strength = _strength(ratio=1.2)  # 1.4 x (1 + 0.2 x 0.15) = 1.442
    workings = attack_workings("Man City", strength, PARAMS, RECORD, _detail(1.2), True)
    text = _text(workings)
    assert workings.heading == "How Man City's attack is calculated"
    assert [step.heading for step in workings.steps] == [
        "1. Fitted from results",
        "2. Chances created this season",
        "3. The adjustment",
    ]
    assert "**1,234 matches**" in text and "about 385 days old counts half as much" in text
    assert "is **1.40**, so it scores about 40% more goals" in text
    assert "scored **2.00** goals a game in 10 games this season, and the average team in the league 1.50" in text
    assert "**24.0** over 10 matches = **2.40 a match**" in text
    assert "average of the league's 20 clubs is **2.00**" in text and "creates **1.20×**" in text
    assert "(1.20 − 1) = +0.20" in text and "× 15% = **+3.0%**" in text
    assert "1.40 × (1 + 0.030) = **1.44**" in text


def test_the_adjusted_figure_the_panel_ends_on_is_the_one_the_leaderboard_shows():
    for ratio in (0.7, 1.0, 1.2, 1.9):
        strength = _strength(ratio=ratio)
        text = _text(attack_workings("A", strength, PARAMS, None, _detail(ratio), True))
        assert f"= **{strength.attack:.2f}**" in text


@pytest.mark.parametrize("ratio, nudge", [(3.0, "+15.0%"), (0.0, "−15.0%"), (0.2, "−12.0%")])
def test_a_big_swing_in_chances_is_held_to_the_cap(ratio, nudge):  # the ratio's distance from 1 stops counting at 1
    text = _text(attack_workings("A", _strength(ratio=ratio), PARAMS, None, _detail(ratio), True))
    assert f"= **{nudge}**" in text


def test_a_below_average_attack_is_adjusted_down_with_a_real_minus_sign():
    text = _text(attack_workings("A", _strength(ratio=0.8), PARAMS, None, _detail(0.8), True))
    assert "(0.80 − 1) = −0.20" in text and "**−3.0%**" in text and "1.40 × (1 − 0.030) = **1.36**" in text


def test_the_biggest_contributors_are_listed_with_everyone_else_added_up():
    squad = [_rate_player(f"Player {i}", xg=float(i), xa=1.0) for i in range(1, 9)]
    workings = attack_workings("A", _strength(), PARAMS, None, _detail(1.2), True, squad)
    table = workings.steps[1].table_html
    names = re.findall(r"<td class=\"left\">(Player \d|The other \d+ players)</td>", table)
    assert names == ["Player 8", "Player 7", "Player 6", "Player 5", "Player 4", "The other 3 players"]
    assert "<b>8.7</b>" in table  # 8 xG + 0.7 x 1 xA
    other = re.findall(r"<tr>.*?</tr>", table)[-1]
    assert ">6.0<" in other and ">3.0<" in other and "<b>8.1</b>" in other  # 3+2+1 xG, 3 xA, 6 + 2.1
    assert "The biggest contributors:" in workings.steps[1].paragraphs[1]


def test_no_squad_means_no_table_and_no_dangling_introduction():
    workings = attack_workings("A", _strength(), PARAMS, None, _detail(1.2), True, [])
    assert workings.steps[1].table_html == ""
    assert "biggest contributors" not in _text(workings)


def test_a_squad_with_no_chances_at_all_has_no_table():
    workings = attack_workings("A", _strength(), PARAMS, None, _detail(1.2), True, [_rate_player("Keeper", 0.0, 0.0)])
    assert workings.steps[1].table_html == ""


def test_without_chance_data_the_attack_is_just_the_fitted_rating_and_the_panel_says_why():
    strength = _strength(ratio=None)
    uncovered = attack_workings("Real Madrid", strength, PARAMS, None, None, chances_covered=False)
    assert [step.heading for step in uncovered.steps] == ["1. Fitted from results", "2. Adjusted for chances created"]
    assert "no per-player chance data for this competition" in _text(uncovered)
    assert "not adjusted: it is the same as the results-only figure, **1.40**" in _text(uncovered)

    too_early = attack_workings("Arsenal", strength, PARAMS, None, None, chances_covered=True)
    assert "needs at least 3 matches played this season" in _text(too_early)


def test_the_check_against_this_seasons_goals_is_left_out_when_there_is_no_record():
    assert "As a check on the fit" not in _text(attack_workings("A", _strength(), PARAMS, None, _detail(1.2), True))


def test_a_team_name_cannot_inject_markdown_into_the_panel():
    text = _text(attack_workings("[x](http://evil.example)", _strength(), PARAMS, None, None, True))
    assert "[x](http" not in text and r"\[x\]\(http" in text


def test_the_player_names_in_the_table_are_html_escaped():
    squad = [_rate_player("<b>x</b>", 5.0, 0.0)]
    table = attack_workings("A", _strength(), PARAMS, None, _detail(1.2), True, squad).steps[1].table_html
    assert "&lt;b&gt;x&lt;/b&gt;" in table and "<b>x</b>" not in table


# ---- defense ----


def test_defense_is_the_fitted_goals_conceded_rating_and_nothing_adjusts_it():
    workings = defense_workings("Man City", _strength(defense=0.8), PARAMS, RECORD)
    text = _text(workings)
    assert workings.heading == "How Man City's defense is calculated"
    assert [step.heading for step in workings.steps] == [
        "1. Fitted from goals conceded",
        "2. Nothing is added to it",
        "3. How it is used",
    ]
    assert "**lower is better**" in text and "is **0.80**, so opponents score about 20% fewer goals against it" in text
    assert "conceded **1.00** goals a game in 10 games this season, and the average team in the league 1.50" in text
    assert "not adjusted for the chances a team creates" in text
    assert "its attack × **0.80**, and × 1.30 (the home advantage) as well when the opponent is the home side" in text


def test_a_leaky_defense_is_described_as_conceding_more():
    assert "about 25% more goals against it" in _text(defense_workings("A", _strength(defense=1.25), PARAMS, None))


# ---- net strength ----


def _board(count: int) -> pd.DataFrame:
    rows = [
        {
            "team_id": team_id,
            "Team": f"Team {team_id}",
            "Attack": 2.0 - team_id * 0.1,
            "Attack (from results only)": 2.0 - team_id * 0.1,
            LEADERBOARD_DEFENSE_COLUMN: 1.0,
            "Net strength": 1.0 - team_id * 0.1,
        }
        for team_id in range(1, count + 1)
    ]
    return pd.DataFrame(rows)  # already best first


def _strength_of(board: pd.DataFrame, team_id: int) -> TeamStrength:
    from soccer_predictor.dashboard.squad_strength import team_strength

    return team_strength(board, team_id)


def _shown_teams(workings: RatingWorkings) -> list[str]:
    return re.findall(r">(Team \d+)</a>", workings.steps[1].table_html)


def test_net_strength_is_attack_minus_defense_and_ranks_the_team():
    board = _board(6)
    workings = net_workings("Team 3", _strength_of(board, 3), board, "EPL")
    text = _text(workings)
    assert workings.heading == "How Team 3's net strength is calculated"
    assert "= 1.70 − 1.00 = **+0.70**" in text
    assert "**3rd** of 6 clubs this season" in text


def test_the_clubs_just_above_and_below_are_shown_with_the_gap():
    board = _board(6)
    workings = net_workings("Team 3", _strength_of(board, 3), board, "EPL")
    table = workings.steps[1].table_html
    assert _shown_teams(workings) == ["Team 1", "Team 2", "Team 3", "Team 4", "Team 5"]
    rows = re.findall(r"<tr[^>]*>.*?</tr>", table.split("<tbody>")[1])
    assert 'class="selected"' in rows[2] and "<b>Team 3</b>" not in rows[2]  # the team's own row is the marked one
    assert "+0.20" in rows[0] and "+0.10" in rows[1] and "−0.10" in rows[3] and "−0.20" in rows[4]


def test_at_the_top_and_the_bottom_there_are_just_fewer_neighbours():
    board = _board(6)
    assert _shown_teams(net_workings("Team 1", _strength_of(board, 1), board, "EPL")) == ["Team 1", "Team 2", "Team 3"]
    assert _shown_teams(net_workings("Team 6", _strength_of(board, 6), board, "EPL")) == ["Team 4", "Team 5", "Team 6"]


def test_a_one_club_board_still_has_workings():
    board = _board(1)
    workings = net_workings("Team 1", _strength_of(board, 1), board, "EPL")
    assert _shown_teams(workings) == ["Team 1"] and "**1st** of 1 clubs" in _text(workings)


def test_each_neighbour_links_to_its_team_page():
    board = _board(3)
    table = net_workings("Team 2", _strength_of(board, 2), board, "EPL").steps[1].table_html
    assert 'href="/team_detail?league=EPL&amp;team=1"' in table and 'href="/team_detail?league=EPL&amp;team=3"' in table
