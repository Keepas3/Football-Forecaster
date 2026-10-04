from __future__ import annotations

import datetime as dt

import pandas as pd

from soccer_predictor.model.standings import compute_group_standings

TEAM_NAMES = {1: "Spain", 2: "Croatia", 3: "England", 4: "Italy"}


def _matches(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["date", "season", "home_team_id", "away_team_id", "home_goals", "away_goals", "group_name"],
    )


def test_matches_split_into_their_own_group_tables():
    matches = _matches(
        [
            (dt.date(2024, 9, 5), "2425", 1, 2, 2, 0, "Group A1"),
            (dt.date(2024, 9, 8), "2425", 3, 4, 1, 1, "Group A2"),
        ]
    )
    groups = compute_group_standings(matches, TEAM_NAMES, "2425")

    assert list(groups) == ["Group A1", "Group A2"]
    assert [s.team_name for s in groups["Group A1"]] == ["Spain", "Croatia"]
    # England and Italy never appear in Group A1's table, only their own.
    assert {s.team_id for s in groups["Group A2"]} == {3, 4}


def test_knockout_matches_with_no_group_are_excluded():
    matches = _matches(
        [
            (dt.date(2024, 9, 5), "2425", 1, 2, 2, 0, "Group A1"),
            (dt.date(2025, 3, 20), "2425", 1, 3, 5, 0, None),  # quarter-final
        ]
    )
    groups = compute_group_standings(matches, TEAM_NAMES, "2425")

    spain = next(s for s in groups["Group A1"] if s.team_id == 1)
    assert spain.played == 1
    assert spain.goals_for == 2
    assert 3 not in {s.team_id for s in groups["Group A1"]}


def test_only_the_requested_season_is_used():
    matches = _matches(
        [
            (dt.date(2022, 9, 5), "2223", 1, 2, 4, 0, "Group A4"),
            (dt.date(2024, 9, 5), "2425", 1, 2, 0, 1, "Group A1"),
        ]
    )
    assert list(compute_group_standings(matches, TEAM_NAMES, "2425")) == ["Group A1"]


def test_teams_rank_by_points_then_goal_difference_then_goals_for():
    matches = _matches(
        [
            (dt.date(2024, 9, 5), "2425", 1, 2, 3, 0, "G"),  # Spain +3
            (dt.date(2024, 9, 6), "2425", 3, 4, 1, 0, "G"),  # England +1, same points
            (dt.date(2024, 9, 9), "2425", 2, 4, 0, 0, "G"),
        ]
    )
    table = compute_group_standings(matches, TEAM_NAMES, "2425")["G"]
    # Spain and England both won once; Spain's goal difference is better.
    assert [s.team_name for s in table[:2]] == ["Spain", "England"]


def test_empty_when_the_season_has_no_group_data():
    matches = _matches([(dt.date(2024, 9, 5), "2425", 1, 2, 2, 0, None)])
    assert compute_group_standings(matches, TEAM_NAMES, "2425") == {}
    # A frame from before the column existed behaves the same.
    assert compute_group_standings(matches.drop(columns=["group_name"]), TEAM_NAMES, "2425") == {}


def test_teams_with_only_fixtures_so_far_are_listed_with_zero_games():
    matches = _matches([(dt.date(2026, 9, 5), "2627", 1, 2, 2, 0, "Group A1")])
    fixture_groups = pd.DataFrame(
        [(3, 4, "Group A2"), (1, 2, "Group A1")],
        columns=["home_team_id", "away_team_id", "group_name"],
    )
    groups = compute_group_standings(matches, TEAM_NAMES, "2627", fixture_groups=fixture_groups)

    assert set(groups) == {"Group A1", "Group A2"}
    unplayed = {s.team_id: s for s in groups["Group A2"]}
    assert set(unplayed) == {3, 4}
    assert all(s.played == 0 for s in unplayed.values())
    # A team that already played isn't duplicated or reset by its fixture.
    assert len(groups["Group A1"]) == 2
    assert next(s for s in groups["Group A1"] if s.team_id == 1).played == 1


def test_a_group_with_fixtures_but_no_results_still_shows_when_nothing_has_been_played():
    matches = _matches([])
    fixture_groups = pd.DataFrame([(1, 2, "Group A")], columns=["home_team_id", "away_team_id", "group_name"])
    groups = compute_group_standings(matches, TEAM_NAMES, "2627", fixture_groups=fixture_groups)
    assert {s.team_id for s in groups["Group A"]} == {1, 2}
