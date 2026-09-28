from __future__ import annotations

from soccer_predictor.model.player_importance import (
    MAX_IMPORTANCE_WEIGHT,
    MIN_IMPORTANCE_WEIGHT,
    compute_attack_importance,
    compute_defense_importance,
    goal_contribution_value,
)


def test_goal_contribution_values_goals_above_assists():
    assert goal_contribution_value(1, 0) > goal_contribution_value(0, 1)


def test_attack_importance_none_on_zero_team_total():
    assert compute_attack_importance(goals=5, assists=2, team_goal_contribution_total=0) is None


def test_attack_importance_scales_with_share_of_team_output():
    low_share = compute_attack_importance(goals=1, assists=0, team_goal_contribution_total=100)
    high_share = compute_attack_importance(goals=20, assists=10, team_goal_contribution_total=100)
    assert low_share < high_share


def test_attack_importance_clipped_to_bounds():
    # A player who IS the entire team's attacking output should hit the
    # ceiling, not exceed [MIN, MAX].
    dominant = compute_attack_importance(goals=50, assists=20, team_goal_contribution_total=64)
    assert MIN_IMPORTANCE_WEIGHT <= dominant <= MAX_IMPORTANCE_WEIGHT
    assert dominant == MAX_IMPORTANCE_WEIGHT

    negligible = compute_attack_importance(goals=0, assists=0, team_goal_contribution_total=100)
    assert negligible == MIN_IMPORTANCE_WEIGHT


def test_defense_importance_none_on_zero_max_minutes():
    assert compute_defense_importance(minutes=900, max_minutes_on_roster=0, rating=7.0) is None


def test_defense_importance_scales_with_minutes_share():
    bench_player = compute_defense_importance(minutes=90, max_minutes_on_roster=3000, rating=None)
    undisputed_starter = compute_defense_importance(minutes=2900, max_minutes_on_roster=3000, rating=None)
    assert bench_player < undisputed_starter


def test_defense_importance_rating_nudge_direction():
    baseline = compute_defense_importance(minutes=1500, max_minutes_on_roster=3000, rating=None)
    good_rating = compute_defense_importance(minutes=1500, max_minutes_on_roster=3000, rating=8.0)
    poor_rating = compute_defense_importance(minutes=1500, max_minutes_on_roster=3000, rating=4.0)
    assert good_rating > baseline > poor_rating


def test_defense_importance_clipped_to_bounds():
    assert compute_defense_importance(minutes=0, max_minutes_on_roster=3000, rating=None) == MIN_IMPORTANCE_WEIGHT
    maxed = compute_defense_importance(minutes=3000, max_minutes_on_roster=3000, rating=9.0)
    assert maxed == MAX_IMPORTANCE_WEIGHT
