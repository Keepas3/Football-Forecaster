"""The team page's "Squad strength": how each player counts toward the team's
attack and defence ratings, the usual XI to draw on a pitch, and the team's net
strength and recent results. No Streamlit calls, no database, no network.

A player is priced the way the model prices him when he is injured or left out
of a named lineup (model/lineup_adjustment.py): an attacker or midfielder who is
absent takes a share off the team's attack, a defender or goalkeeper who is
absent adds to the goals it concedes. The table shows those same figures.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from soccer_predictor.dashboard.components import (
    _FORM_EMOJI,
    LEADERBOARD_DEFENSE_COLUMN,
    TableCell,
    TableColumn,
    _signed_number,
    form_game_tooltip,
    sortable_table_html,
)
from soccer_predictor.dashboard.pitch import ROLE_ORDER, SquadSlot
from soccer_predictor.model.injury_adjustment import MAX_MULTIPLIER, MIN_MULTIPLIER
from soccer_predictor.model.lineup_adjustment import (
    LINEUP_ATTACK_FACTOR,
    LINEUP_DEFENSE_FACTOR,
    ExpectedPlayer,
    expected_xi,
)
from soccer_predictor.model.standings import DEFAULT_FORM_LENGTH, FormGame

ROLE_LABELS = {"GK": "Goalkeeper", "DEF": "Defender", "MID": "Midfielder", "FWD": "Forward"}


@dataclass(frozen=True)
class PlayerStrength:
    name: str
    role: str  # "GK" | "DEF" | "MID" | "FWD" | ""
    minutes: float
    goals: int
    assists: int
    attack_cut: float | None  # share of the team's attack lost if he is out; None for a defender or keeper
    defense_rise: float | None  # extra goals conceded, as a share, if he is out; None for everyone else
    in_xi: bool


def attack_cut(weight: float) -> float:
    return min(weight * LINEUP_ATTACK_FACTOR, 1 - MIN_MULTIPLIER)


def defense_rise(weight: float) -> float:
    return min(weight * LINEUP_DEFENSE_FACTOR, MAX_MULTIPLIER - 1)


def player_strengths(squad: Sequence[ExpectedPlayer]) -> list[PlayerStrength]:
    """Every player in `squad`, the regulars (most minutes) first."""
    xi = {player.name for player in expected_xi(squad)}
    rows = [
        PlayerStrength(
            name=player.name,
            role=player.role,
            minutes=player.minutes,
            goals=player.goals,
            assists=player.assists,
            attack_cut=None if player.position == "defense" else attack_cut(player.weight),
            defense_rise=defense_rise(player.weight) if player.position == "defense" else None,
            in_xi=player.name in xi,
        )
        for player in squad
    ]
    return sorted(rows, key=lambda row: (-row.minutes, row.name))


def usual_xi_slots(squad: Sequence[ExpectedPlayer]) -> list[SquadSlot]:
    """The team's usual XI, most important first, as the pitch graphic draws it."""
    slots = []
    for player in sorted(expected_xi(squad), key=lambda p: (-p.weight, p.name)):
        defensive = player.position == "defense"
        figure = f"+{defense_rise(player.weight) * 100:.0f}%" if defensive else f"−{attack_cut(player.weight) * 100:.0f}%"
        slots.append(SquadSlot(player.name, player.role if player.role in ROLE_ORDER else "MID", figure, defensive))
    return slots


def _cut_cell(value: float | None, sign: str) -> TableCell:
    if value is None:
        return TableCell("–")
    return TableCell(f"{sign}{value * 100:.1f}%", value * 100)


SQUAD_COLUMNS = [
    TableColumn("Player", left=True, sort="text", first="asc"),
    TableColumn("Role", left=True, sort="text", first="asc", hint="Goalkeeper, defender, midfielder or forward"),
    TableColumn("XI", left=True, sort="num", hint="★ = one of the 11 who have played the most minutes (the usual XI)"),
    TableColumn("Mins", sort="num"),
    TableColumn("Goals", sort="num"),
    TableColumn("Assists", sort="num"),
    TableColumn("Attack", sort="num", hint="How much the team's attack rating drops if he is out"),
    TableColumn("Defense", sort="num", hint="How much more the team concedes if he is out"),
]


def squad_table_html(players: Sequence[PlayerStrength]) -> str:
    """The squad as a sortable table (call inject_standings_sort() once on the page)."""
    rows = []
    for player in players:
        rows.append(
            [
                TableCell(html.escape(player.name), player.name.casefold()),
                TableCell(
                    f'<span title="{ROLE_LABELS[player.role]}">{player.role}</span>' if player.role in ROLE_LABELS else "–",
                    player.role,
                ),
                TableCell("★" if player.in_xi else "", 1 if player.in_xi else 0),
                TableCell(f"{player.minutes:,.0f}", player.minutes),
                TableCell(str(player.goals), player.goals),
                TableCell(str(player.assists), player.assists),
                _cut_cell(player.attack_cut, "−"),
                _cut_cell(player.defense_rise, "+"),
            ]
        )
    return sortable_table_html(SQUAD_COLUMNS, rows, compact=True)


@dataclass(frozen=True)
class TeamStrength:
    attack: float  # adjusted for this season's chances created
    attack_results_only: float
    defense: float  # goals conceded against an average team: lower is better
    net: float  # attack minus defense
    rank: int | None  # 1 = strongest; None for a club that is not on the current leaderboard


def team_strength(leaderboard: pd.DataFrame, team_id: int) -> TeamStrength | None:
    """The team's row of leaderboard_dataframe (already ranked by net strength)."""
    for rank, row in enumerate(leaderboard.to_dict("records"), start=1):
        if row["team_id"] == team_id:
            return TeamStrength(
                attack=row["Attack"],
                attack_results_only=row["Attack (from results only)"],
                defense=row[LEADERBOARD_DEFENSE_COLUMN],
                net=row["Net strength"],
                rank=rank,
            )
    return None


def ordinal(number: int) -> str:
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def strength_line(strength: TeamStrength) -> str:
    where = (
        f"{ordinal(strength.rank)} on the Predictor's Team Strength Leaderboard"
        if strength.rank is not None
        else "not on the Predictor's Team Strength Leaderboard, which lists this season's clubs"
    )
    return f"**Net strength {_signed_number(strength.net)}** (attack minus defense) · {where}"


def recent_form(
    matches: pd.DataFrame, team_id: int, team_names: dict[int, str], length: int = DEFAULT_FORM_LENGTH
) -> list[tuple[str, FormGame]]:
    """The team's last `length` results from `matches` (date, home_team_id,
    away_team_id, home_goals, away_goals; any order), oldest first, each as
    ("W" | "D" | "L", the game)."""
    form = []
    for row in matches.sort_values("date", kind="stable").tail(length).itertuples(index=False):
        is_home = row.home_team_id == team_id
        scored, conceded = (row.home_goals, row.away_goals) if is_home else (row.away_goals, row.home_goals)
        game = FormGame(
            row.date,
            team_names.get(row.home_team_id, f"team#{row.home_team_id}"),
            team_names.get(row.away_team_id, f"team#{row.away_team_id}"),
            int(row.home_goals),
            int(row.away_goals),
        )
        form.append(("W" if scored > conceded else "D" if scored == conceded else "L", game))
    return form


def form_html(form: Sequence[tuple[str, FormGame]]) -> str:
    """Coloured squares, oldest to newest, each with the game as its hover text,
    then the record they add up to. HTML-escaped."""
    squares = "".join(
        f'<span title="{html.escape(form_game_tooltip(game, result), quote=True)}" style="cursor:help">'
        f"{_FORM_EMOJI[result]}</span>"
        for result, game in form
    )
    results = [result for result, _ in form]
    return f"{squares} &nbsp;{results.count('W')}W {results.count('D')}D {results.count('L')}L"
