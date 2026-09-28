from __future__ import annotations

import datetime as dt

import pandas as pd

from soccer_predictor.model.team_facts import compute_head_to_head, compute_team_facts

TEAM = 1
RIVAL_A = 2
RIVAL_B = 3


def _df(rows: list[tuple]) -> pd.DataFrame:
    # (date, home_id, away_id, home_goals, away_goals)
    return pd.DataFrame(
        rows, columns=["date", "home_team_id", "away_team_id", "home_goals", "away_goals"]
    )


def test_returns_none_for_empty_history():
    assert compute_team_facts(TEAM, _df([])) is None


def test_totals_and_biggest_win_loss():
    matches = _df(
        [
            (dt.date(2026, 1, 1), TEAM, RIVAL_A, 5, 0),  # win, home, +5
            (dt.date(2026, 1, 8), RIVAL_A, TEAM, 4, 0),  # loss, away, -4
            (dt.date(2026, 1, 15), TEAM, RIVAL_A, 1, 1),  # draw
        ]
    )
    facts = compute_team_facts(TEAM, matches)

    assert facts is not None
    assert facts.total_matches == 3
    assert facts.total_goals_for == 5 + 0 + 1
    assert facts.total_goals_against == 0 + 4 + 1

    assert facts.biggest_win is not None
    assert (facts.biggest_win.goals_for, facts.biggest_win.goals_against) == (5, 0)
    assert facts.biggest_win.is_home is True

    assert facts.biggest_loss is not None
    assert (facts.biggest_loss.goals_for, facts.biggest_loss.goals_against) == (0, 4)
    assert facts.biggest_loss.is_home is False


def test_no_loss_on_record_leaves_biggest_loss_none():
    matches = _df([(dt.date(2026, 1, 1), TEAM, RIVAL_A, 2, 0)])
    facts = compute_team_facts(TEAM, matches)
    assert facts.biggest_win is not None
    assert facts.biggest_loss is None


def test_win_and_unbeaten_streaks():
    matches = _df(
        [
            (dt.date(2026, 1, 1), TEAM, RIVAL_A, 1, 0),  # W
            (dt.date(2026, 1, 8), TEAM, RIVAL_A, 2, 0),  # W
            (dt.date(2026, 1, 15), RIVAL_A, TEAM, 1, 1),  # D -- breaks win streak, extends unbeaten
            (dt.date(2026, 1, 22), TEAM, RIVAL_A, 0, 3),  # L -- breaks both
            (dt.date(2026, 1, 29), TEAM, RIVAL_A, 1, 0),  # W -- new streak starts
        ]
    )
    facts = compute_team_facts(TEAM, matches)
    assert facts.longest_win_streak == 2
    assert facts.longest_unbeaten_streak == 3


def test_most_played_opponent_and_head_to_head_record():
    matches = _df(
        [
            (dt.date(2026, 1, 1), TEAM, RIVAL_A, 1, 0),  # vs A: W
            (dt.date(2026, 1, 8), RIVAL_A, TEAM, 1, 1),  # vs A: D
            (dt.date(2026, 1, 15), TEAM, RIVAL_A, 0, 2),  # vs A: L
            (dt.date(2026, 1, 22), TEAM, RIVAL_B, 3, 0),  # vs B: W (only 1 match)
        ]
    )
    facts = compute_team_facts(TEAM, matches)
    assert facts.most_played_opponent_id == RIVAL_A
    assert facts.most_played_opponent_matches == 3
    assert facts.most_played_opponent_record == (1, 1, 1)


def test_chronological_order_of_input_does_not_affect_streaks():
    # Same matches as test_win_and_unbeaten_streaks but shuffled -- the
    # function must sort by date itself rather than trust input order.
    matches = _df(
        [
            (dt.date(2026, 1, 22), TEAM, RIVAL_A, 0, 3),  # L
            (dt.date(2026, 1, 1), TEAM, RIVAL_A, 1, 0),  # W
            (dt.date(2026, 1, 29), TEAM, RIVAL_A, 1, 0),  # W
            (dt.date(2026, 1, 15), RIVAL_A, TEAM, 1, 1),  # D
            (dt.date(2026, 1, 8), TEAM, RIVAL_A, 2, 0),  # W
        ]
    )
    facts = compute_team_facts(TEAM, matches)
    assert facts.longest_win_streak == 2
    assert facts.longest_unbeaten_streak == 3


def test_head_to_head_returns_none_when_never_played():
    matches = _df([(dt.date(2026, 1, 1), TEAM, RIVAL_A, 1, 0)])
    assert compute_head_to_head(TEAM, RIVAL_B, matches) is None


def test_head_to_head_isolates_just_the_one_opponent():
    matches = _df(
        [
            (dt.date(2026, 1, 1), TEAM, RIVAL_A, 2, 0),  # vs A: W
            (dt.date(2026, 1, 8), RIVAL_A, TEAM, 1, 1),  # vs A: D
            (dt.date(2026, 1, 15), TEAM, RIVAL_B, 3, 0),  # vs B: W -- must be excluded
        ]
    )
    h2h = compute_head_to_head(TEAM, RIVAL_A, matches)

    assert h2h is not None
    assert h2h.opponent_id == RIVAL_A
    assert h2h.total_matches == 2
    assert (h2h.wins, h2h.draws, h2h.losses) == (1, 1, 0)
    assert h2h.goals_for == 3  # 2 + 1
    assert h2h.goals_against == 1  # 0 + 1


def test_head_to_head_recent_matches_are_newest_first_and_capped():
    matches = _df(
        [
            (dt.date(2026, 1, 1), TEAM, RIVAL_A, 1, 0),
            (dt.date(2026, 2, 1), TEAM, RIVAL_A, 2, 0),
            (dt.date(2026, 3, 1), TEAM, RIVAL_A, 3, 0),
        ]
    )
    h2h = compute_head_to_head(TEAM, RIVAL_A, matches, recent_n=2)

    assert len(h2h.recent_matches) == 2
    assert h2h.recent_matches[0].date == dt.date(2026, 3, 1)
    assert h2h.recent_matches[1].date == dt.date(2026, 2, 1)
