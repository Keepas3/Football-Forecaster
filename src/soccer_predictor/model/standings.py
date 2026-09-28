"""Computes classic league-table standings (Played/W/D/L/GF/GA/GD/Pts/Form)
straight from actual results -- independent of the Dixon-Coles model. This
is what the Leagues page shows; the model's attack/defense ratings are a
separate, per-team detail shown only on Team Detail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# 7, not 5 -- a single-tournament competition like the World Cup only has
# ~7-8 games total for a finalist, so 5 was clipping off their early group
# games; 7 covers a full run for every league without over-cluttering the
# usual weekly domestic form strip.
DEFAULT_FORM_LENGTH = 7


@dataclass
class TeamStanding:
    team_id: int
    team_name: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0
    form: list[str] = field(default_factory=list)  # "W"/"D"/"L", oldest first

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def points(self) -> int:
        return self.won * 3 + self.drawn


def compute_standings(
    matches: pd.DataFrame,
    team_names: dict[int, str],
    season: str,
    form_length: int = DEFAULT_FORM_LENGTH,
) -> list[TeamStanding]:
    """`matches` needs columns: date, season, home_team_id, away_team_id,
    home_goals, away_goals (see storage.repository.matches_for_league).
    Returns standings sorted by points/goal-diff/goals-for, best first.

    Only teams that actually played a match in `season` are included --
    `team_names` (and the underlying `matches` DataFrame) span every season
    ever ingested for the league, so promoted/relegated clubs from other
    seasons must not show up as padding rows with zero games played.
    """
    season_matches = matches[matches["season"] == season].sort_values("date")

    standings: dict[int, TeamStanding] = {}

    for row in season_matches.itertuples(index=False):
        home = standings.setdefault(
            row.home_team_id,
            TeamStanding(row.home_team_id, team_names.get(row.home_team_id, f"team#{row.home_team_id}")),
        )
        away = standings.setdefault(
            row.away_team_id,
            TeamStanding(row.away_team_id, team_names.get(row.away_team_id, f"team#{row.away_team_id}")),
        )

        home.played += 1
        away.played += 1
        home.goals_for += row.home_goals
        away.goals_for += row.away_goals
        home.goals_against += row.away_goals
        away.goals_against += row.home_goals

        if row.home_goals > row.away_goals:
            home.won += 1
            away.lost += 1
            home.form.append("W")
            away.form.append("L")
        elif row.home_goals < row.away_goals:
            away.won += 1
            home.lost += 1
            home.form.append("L")
            away.form.append("W")
        else:
            home.drawn += 1
            away.drawn += 1
            home.form.append("D")
            away.form.append("D")

    for standing in standings.values():
        standing.form = standing.form[-form_length:]

    return sorted(
        standings.values(),
        key=lambda s: (-s.points, -s.goal_diff, -s.goals_for, s.team_name),
    )
