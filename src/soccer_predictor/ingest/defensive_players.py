"""The Players page's Defenders and Goalkeepers lists: who defends and keeps goal best.

The default Top players list ranks by goals, assists and expected goals, which says nothing
about a defender or a keeper. Only two leagues have a free source that does:

* Premier League -- the Fantasy Premier League feed (ingest/fpl_client.py, already downloaded
  for injuries): tackles, clearances, blocks, interceptions, clean sheets, goals conceded, expected
  goals conceded and saves. Clean sheets and goals conceded count the whole team while the player
  was on the pitch, and its expected goals conceded is the team's, so "goals prevented" there is a
  rough guide, not a shot-stopping measure.
* MLS -- American Soccer Analysis (ingest/asa_client.py): goals added above average for each kind of
  defensive action, and for keepers the expected goals of the shots they faced against the goals
  they let in (a proper goals-prevented number).

La Liga, the Bundesliga, Serie A and Ligue 1 (Understat gives only attacking numbers) and the
cup competitions have no such free source, so their lists are empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from soccer_predictor.config import League
from soccer_predictor.ingest import asa_client, fpl_client
from soccer_predictor.ingest.top_players import MIN_MINUTES_PER_90, TOP_N, find_top_player, per_90

DEFENDER = "defender"
GOALKEEPER = "goalkeeper"
ROLES = (DEFENDER, GOALKEEPER)
ROLE_LABELS = {DEFENDER: "Defenders", GOALKEEPER: "Goalkeepers"}

# The league code -> the source that has defender and keeper numbers for it.
SOURCES = {"EPL": "Fantasy Premier League", "MLS": "American Soccer Analysis"}

ACTIONS_P90 = "actions_90"
INTERRUPTING_P90 = "interrupting_90"
TOTAL_P90 = "total_90"
CLEAN_SHEETS = "clean_sheets"
CONCEDED_P90 = "conceded_90"
GOALS_PLUS_ASSISTS = "goals_assists"
PREVENTED = "prevented"
SAVE_PCT = "save_pct"
SAVES = "saves"
PENALTIES_SAVED = "penalties_saved"
SHOTS_FACED = "shots_faced"


@dataclass(frozen=True)
class Metric:
    label: str
    decimals: int = 0
    lower_is_better: bool = False  # goals conceded: the fewest first
    needs_minutes: bool = False  # a rate or a percentage: left out below MIN_MINUTES_PER_90
    signed: bool = False  # shown with + or -, for a difference
    suffix: str = ""


METRICS = {
    ACTIONS_P90: Metric("Defensive actions per 90", 1, needs_minutes=True),
    INTERRUPTING_P90: Metric("Interrupting g+ per 90", 2, needs_minutes=True, signed=True),
    TOTAL_P90: Metric("Total g+ per 90", 2, needs_minutes=True, signed=True),
    CLEAN_SHEETS: Metric("Clean sheets"),
    CONCEDED_P90: Metric("Goals conceded per 90", 2, lower_is_better=True, needs_minutes=True),
    GOALS_PLUS_ASSISTS: Metric("Goals + assists"),
    PREVENTED: Metric("Goals prevented", 1, signed=True),
    SAVE_PCT: Metric("Save %", 1, needs_minutes=True, suffix="%"),
    SAVES: Metric("Saves"),
    PENALTIES_SAVED: Metric("Penalties saved"),
    SHOTS_FACED: Metric("Shots faced"),
}

# The order a role's metrics are offered in; the first one its source has is the default.
ROLE_METRICS = {
    DEFENDER: [ACTIONS_P90, INTERRUPTING_P90, TOTAL_P90, CLEAN_SHEETS, CONCEDED_P90, GOALS_PLUS_ASSISTS],
    GOALKEEPER: [PREVENTED, SAVE_PCT, SAVES, CLEAN_SHEETS, CONCEDED_P90, PENALTIES_SAVED, SHOTS_FACED],
}


@dataclass
class DefensivePlayer:
    name: str
    team: str
    role: str
    minutes: int
    values: dict[str, float | None] = field(default_factory=dict)  # metric id -> value


@dataclass
class DefensivePlayers:
    players: list[DefensivePlayer]
    source: str  # shown under the table; "" when the league has none
    season: int | None


def available_metrics(role: str, players: list[DefensivePlayer]) -> list[str]:
    """The metrics this league's data can rank `role` by, in the order they are offered."""
    return [m for m in ROLE_METRICS[role] if any(p.values.get(m) is not None for p in players)]


def _eligible(player: DefensivePlayer, metric: str) -> bool:
    if player.values.get(metric) is None:
        return False
    return not METRICS[metric].needs_minutes or player.minutes >= MIN_MINUTES_PER_90


def rank_players(players: list[DefensivePlayer], metric: str, limit: int = TOP_N) -> list[DefensivePlayer]:
    """Best first by `metric` (fewest first for goals conceded). Players with no value for it,
    or too few minutes for a rate, are left out; ties go to the one with more minutes."""
    sign = 1 if METRICS[metric].lower_is_better else -1
    ranked = [p for p in players if _eligible(p, metric)]
    ranked.sort(key=lambda p: (sign * p.values[metric], -p.minutes, p.name))
    return ranked[:limit]


def format_value(metric: str, value: float | None) -> str:
    if value is None:
        return "–"
    spec = METRICS[metric]
    sign = "+" if spec.signed else ""
    shown = round(value, spec.decimals) or 0.0  # a tiny negative would otherwise print as "-0.0"
    return f"{shown:{sign}.{spec.decimals}f}{spec.suffix}"


def table_rows(players: list[DefensivePlayer], role: str, metric: str) -> list[dict]:
    """The page's table: rank, player, club, the ranked metric, the role's other metrics this
    source has, and the minutes played."""
    columns = [metric] + [m for m in available_metrics(role, players) if m != metric]
    rows = []
    for rank, player in enumerate(rank_players(players, metric), start=1):
        row = {"#": rank, "Player": player.name, "Club": player.team}
        for column in columns:
            row[METRICS[column].label] = format_value(column, player.values.get(column))
        row["Minutes"] = player.minutes
        rows.append(row)
    return rows


def league_rankings(players: list[DefensivePlayer], player: DefensivePlayer) -> list[tuple[str, int, int]]:
    """Where this player stands among the league's players of his role, as (label, rank, how
    many are ranked) for each metric his source has. Ties share a rank (1 + the number strictly
    ahead)."""
    standings = []
    for metric in available_metrics(player.role, players):
        if not _eligible(player, metric):
            continue
        spec = METRICS[metric]
        mine = player.values[metric]
        others = [p.values[metric] for p in players if p.role == player.role and _eligible(p, metric)]
        ahead = sum(v < mine if spec.lower_is_better else v > mine for v in others)
        standings.append((spec.label, 1 + ahead, len(others)))
    return standings


def find_defensive_player(players: list[DefensivePlayer], name: str, team: str = "") -> DefensivePlayer | None:
    """The listing's entry for a player by name (same club preferred)."""
    return find_top_player(players, name, team)  # type: ignore[arg-type]  # same name/team attributes


def _fpl_players(role: str) -> list[DefensivePlayer]:
    players = []
    for p in fpl_client.fetch_players():
        if p.role != role:
            continue
        values: dict[str, float | None] = {
            CLEAN_SHEETS: p.clean_sheets,
            CONCEDED_P90: per_90(p.goals_conceded, p.minutes),
        }
        if role == DEFENDER:
            values[ACTIONS_P90] = per_90(p.defensive_actions, p.minutes)
            values[GOALS_PLUS_ASSISTS] = p.goals + p.assists
        else:
            faced = p.saves + p.goals_conceded
            values[PREVENTED] = p.expected_goals_conceded - p.goals_conceded if p.minutes else None
            values[SAVE_PCT] = 100 * p.saves / faced if faced else None
            values[SAVES] = p.saves
            values[PENALTIES_SAVED] = p.penalties_saved
        players.append(DefensivePlayer(p.name, p.team_name, role, p.minutes, values))
    return players


def _asa_players(role: str, season: int) -> list[DefensivePlayer]:
    if role == DEFENDER:
        return [
            DefensivePlayer(
                d.name,
                d.team,
                role,
                d.minutes,
                {INTERRUPTING_P90: per_90(d.interrupting, d.minutes), TOTAL_P90: per_90(d.total, d.minutes)},
            )
            for d in asa_client.fetch_league_defenders(season)
        ]
    return [
        DefensivePlayer(
            k.name,
            k.team,
            role,
            k.minutes,
            {
                PREVENTED: k.goals_prevented,
                SAVE_PCT: 100 * k.saves / k.shots_faced if k.shots_faced else None,
                SAVES: k.saves,
                SHOTS_FACED: k.shots_faced,
                CONCEDED_P90: per_90(k.goals_conceded, k.minutes),
            },
        )
        for k in asa_client.fetch_league_goalkeepers(season)
    ]


def fetch_defensive_players(league: League, role: str) -> DefensivePlayers:
    """The league's defenders or goalkeepers with their numbers, from whichever source covers
    it (see the module docstring). Empty when there is none or it fails."""
    season = league.api_season_year(league.seasons[-1])
    if league.code == "EPL":
        return DefensivePlayers(_fpl_players(role), SOURCES["EPL"], season)
    if league.code == "MLS":
        return DefensivePlayers(_asa_players(role, season), SOURCES["MLS"], season)
    return DefensivePlayers([], "", None)
