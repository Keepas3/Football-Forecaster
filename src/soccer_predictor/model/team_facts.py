""""Fun facts" about one team derived purely from its own historical match
results already in our DB -- no external API, just analysis of data we
already have (as opposed to ingest/squad.py's TeamInfo, which comes from
football-data.org).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd


@dataclass
class NotableResult:
    opponent_id: int
    date: dt.date
    goals_for: int
    goals_against: int
    is_home: bool

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against


@dataclass
class TeamFacts:
    total_matches: int
    total_goals_for: int
    total_goals_against: int
    biggest_win: NotableResult | None
    biggest_loss: NotableResult | None
    longest_win_streak: int
    longest_unbeaten_streak: int
    most_played_opponent_id: int
    most_played_opponent_matches: int
    most_played_opponent_record: tuple[int, int, int]  # (wins, draws, losses) vs that opponent


@dataclass
class HeadToHeadRecord:
    opponent_id: int
    total_matches: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    recent_matches: list[NotableResult]  # newest first, capped at `recent_n`


def compute_head_to_head(
    team_id: int, opponent_id: int, matches_df: pd.DataFrame, recent_n: int = 5
) -> HeadToHeadRecord | None:
    """Head-to-head record between `team_id` and one specific `opponent_id`
    -- e.g. clicking a fixture on Team Detail's own schedule to see that
    team's history against just this upcoming opponent, not every opponent.
    `matches_df` is `team_id`'s own match history (see compute_team_facts),
    filtered down here to games against `opponent_id` specifically. None if
    they've never played (in whatever history we have -- see the "how much
    historical data" note in Team Detail).
    """
    relevant = matches_df[
        (matches_df["home_team_id"] == opponent_id) | (matches_df["away_team_id"] == opponent_id)
    ]
    if relevant.empty:
        return None

    wins = draws = losses = goals_for = goals_against = 0
    matches: list[NotableResult] = []
    for row in relevant.sort_values("date", ascending=False).itertuples(index=False):
        is_home = row.home_team_id == team_id
        scored = row.home_goals if is_home else row.away_goals
        conceded = row.away_goals if is_home else row.home_goals
        goals_for += scored
        goals_against += conceded
        if scored > conceded:
            wins += 1
        elif scored == conceded:
            draws += 1
        else:
            losses += 1
        matches.append(
            NotableResult(opponent_id=opponent_id, date=row.date, goals_for=scored, goals_against=conceded, is_home=is_home)
        )

    return HeadToHeadRecord(
        opponent_id=opponent_id,
        total_matches=len(relevant),
        wins=wins,
        draws=draws,
        losses=losses,
        goals_for=goals_for,
        goals_against=goals_against,
        recent_matches=matches[:recent_n],
    )


def compute_team_facts(team_id: int, matches_df: pd.DataFrame) -> TeamFacts | None:
    """`matches_df` needs columns: date, home_team_id, away_team_id,
    home_goals, away_goals (see storage.repository.matches_for_team) --
    every match on record for this team, any league/season, chronological
    order not required (sorted here).
    """
    if matches_df.empty:
        return None

    total_gf = total_ga = 0
    biggest_win: NotableResult | None = None
    biggest_loss: NotableResult | None = None
    opponent_counts: dict[int, int] = {}
    opponent_record: dict[int, list[int]] = {}  # opponent_id -> [wins, draws, losses]

    longest_win_streak = current_win_streak = 0
    longest_unbeaten_streak = current_unbeaten_streak = 0

    for row in matches_df.sort_values("date").itertuples(index=False):
        is_home = row.home_team_id == team_id
        opponent_id = row.away_team_id if is_home else row.home_team_id
        goals_for = row.home_goals if is_home else row.away_goals
        goals_against = row.away_goals if is_home else row.home_goals
        total_gf += goals_for
        total_ga += goals_against

        if goals_for > goals_against:
            current_win_streak += 1
            current_unbeaten_streak += 1
            outcome_idx = 0
        elif goals_for == goals_against:
            current_win_streak = 0
            current_unbeaten_streak += 1
            outcome_idx = 1
        else:
            current_win_streak = 0
            current_unbeaten_streak = 0
            outcome_idx = 2
        longest_win_streak = max(longest_win_streak, current_win_streak)
        longest_unbeaten_streak = max(longest_unbeaten_streak, current_unbeaten_streak)

        result = NotableResult(
            opponent_id=opponent_id,
            date=row.date,
            goals_for=goals_for,
            goals_against=goals_against,
            is_home=is_home,
        )
        if result.goal_diff > 0 and (biggest_win is None or result.goal_diff > biggest_win.goal_diff):
            biggest_win = result
        if result.goal_diff < 0 and (biggest_loss is None or result.goal_diff < biggest_loss.goal_diff):
            biggest_loss = result

        opponent_counts[opponent_id] = opponent_counts.get(opponent_id, 0) + 1
        opponent_record.setdefault(opponent_id, [0, 0, 0])[outcome_idx] += 1

    most_played_id, most_played_count = max(opponent_counts.items(), key=lambda kv: kv[1])

    return TeamFacts(
        total_matches=len(matches_df),
        total_goals_for=total_gf,
        total_goals_against=total_ga,
        biggest_win=biggest_win,
        biggest_loss=biggest_loss,
        longest_win_streak=longest_win_streak,
        longest_unbeaten_streak=longest_unbeaten_streak,
        most_played_opponent_id=most_played_id,
        most_played_opponent_matches=most_played_count,
        most_played_opponent_record=tuple(opponent_record[most_played_id]),
    )
