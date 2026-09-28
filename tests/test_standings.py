from __future__ import annotations

import datetime as dt

import pandas as pd

from soccer_predictor.model.standings import compute_standings

TEAM_NAMES = {1: "Arsenal", 2: "Aston Villa", 3: "Chelsea"}


def _matches(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["date", "season", "home_team_id", "away_team_id", "home_goals", "away_goals"],
    )


def test_points_and_goal_diff_computed_correctly():
    matches = _matches(
        [
            (dt.date(2025, 8, 1), "2425", 1, 2, 2, 0),  # Arsenal beat Villa
            (dt.date(2025, 8, 8), "2425", 3, 1, 1, 1),  # Chelsea drew Arsenal
        ]
    )
    standings = {s.team_id: s for s in compute_standings(matches, TEAM_NAMES, "2425")}

    arsenal = standings[1]
    assert arsenal.played == 2
    assert arsenal.won == 1
    assert arsenal.drawn == 1
    assert arsenal.lost == 0
    assert arsenal.goals_for == 3
    assert arsenal.goals_against == 1
    assert arsenal.goal_diff == 2
    assert arsenal.points == 4

    villa = standings[2]
    assert villa.played == 1
    assert villa.lost == 1
    assert villa.points == 0


def test_sorted_by_points_then_goal_diff_then_goals_for():
    matches = _matches(
        [
            (dt.date(2025, 8, 1), "2425", 1, 2, 3, 0),  # Arsenal 3-0 Villa
            (dt.date(2025, 8, 8), "2425", 3, 2, 1, 0),  # Chelsea 1-0 Villa
        ]
    )
    standings = compute_standings(matches, TEAM_NAMES, "2425")
    # Arsenal and Chelsea both have 3 pts, but Arsenal has better GD (+3 vs +1)
    assert [s.team_name for s in standings] == ["Arsenal", "Chelsea", "Aston Villa"]


def test_form_is_chronological_and_truncated_to_length():
    matches = _matches(
        [
            (dt.date(2025, 8, 1), "2425", 1, 2, 1, 0),  # W
            (dt.date(2025, 8, 8), "2425", 3, 1, 1, 1),  # D
            (dt.date(2025, 8, 15), "2425", 1, 2, 0, 2),  # L
            (dt.date(2025, 8, 22), "2425", 1, 3, 3, 0),  # W
        ]
    )
    standings = {s.team_id: s for s in compute_standings(matches, TEAM_NAMES, "2425", form_length=3)}
    assert standings[1].form == ["D", "L", "W"]


def test_only_matches_for_the_requested_season_are_counted():
    matches = _matches(
        [
            (dt.date(2024, 8, 1), "2324", 1, 2, 5, 0),
            (dt.date(2025, 8, 1), "2425", 1, 2, 1, 1),
        ]
    )
    standings = {s.team_id: s for s in compute_standings(matches, TEAM_NAMES, "2425")}
    assert standings[1].played == 1
    assert standings[1].goals_for == 1


def test_team_with_no_matches_in_season_is_excluded():
    # Chelsea (id 3) is a known team (in team_names) but didn't play this
    # season -- e.g. it was relegated/promoted in a different season. It
    # must not show up as a zero-played padding row alongside real entries.
    matches = _matches([(dt.date(2025, 8, 1), "2425", 1, 2, 1, 0)])
    standings = {s.team_id: s for s in compute_standings(matches, TEAM_NAMES, "2425")}
    assert 3 not in standings
    assert set(standings.keys()) == {1, 2}
