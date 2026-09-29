"""Glue between a team's real historical player stats
(ingest/player_stats.py) and model/player_importance.py's pure formulas --
resolves one injured player's name to their own stats and turns that into a
real per-player importance_weight for ingest/injuries.py, instead of the
flat DEFAULT_API_IMPORTANCE_WEIGHT every API-sourced injury used to get.

DEFAULT_API_IMPORTANCE_WEIGHT lives here (not injuries.py, which imports it
back) so injuries.py can depend on this module without a circular import.

Optionally also takes Understat data (ingest/understat_client.py) for the
attack side: its xG/xA reflects the CURRENT season (unlike API-Football's
free-plan goals/assists, stuck on 2022-2024 -- see player_stats.py's
AVAILABLE_SEASONS), so it's preferred when a name match exists there, with
API-Football goals/assists as the fallback. The two are never mixed in one
ratio -- a player's xG against an xG-space team total, or their goals
against a goals-space team total, never one against the other's denominator.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz, process

from soccer_predictor.ingest.player_stats import HistoricalPlayerStats, fetch_team_player_stats
from soccer_predictor.ingest.understat_client import (
    UNDERSTAT_LEAGUE_SLUG,
    UnderstatPlayerStats,
    fetch_team_season,
)
from soccer_predictor.model.player_importance import (
    compute_attack_importance,
    compute_defense_importance,
)

DEFAULT_API_IMPORTANCE_WEIGHT = 0.5

# Looser than team_mapper.py's 90 -- API-Football's injuries endpoint and its
# player-stats endpoint sometimes format the same player's name differently
# ("N. Kante" vs "N'Golo Kante"), similar to why note_parser.py's chat-note
# matching (85) is also looser than team-name matching.
PLAYER_NAME_FUZZY_THRESHOLD = 80


@dataclass
class TeamOutputTotals:
    goal_contribution_total: float
    max_minutes: int


@dataclass
class UnderstatTeamTotals:
    xg_contribution_total: float


def aggregate_team_output(players: list[HistoricalPlayerStats]) -> TeamOutputTotals:
    total = sum((p.goals or 0) + (p.assists or 0) * 0.7 for p in players)
    max_minutes = max((p.minutes or 0 for p in players), default=0)
    return TeamOutputTotals(goal_contribution_total=total, max_minutes=max_minutes)


def aggregate_understat_team_output(players: list[UnderstatPlayerStats]) -> UnderstatTeamTotals:
    total = sum((p.xg or 0) + (p.xa or 0) * 0.7 for p in players)
    return UnderstatTeamTotals(xg_contribution_total=total)


def find_player_stats(player_name: str, team_stats: list[HistoricalPlayerStats]) -> HistoricalPlayerStats | None:
    """Fuzzy-matches `player_name` against a team's API-Football historical
    stats list. Public: also used by dashboard/views/players.py to look up
    one specific player's historical row, not just internally here.
    """
    if not team_stats:
        return None
    names = [p.name for p in team_stats]
    match = process.extractOne(player_name, names, scorer=fuzz.WRatio)
    if match is None:
        return None
    _, score, index = match
    if score < PLAYER_NAME_FUZZY_THRESHOLD:
        return None
    return team_stats[index]


def find_understat_player(
    player_name: str, understat_players: list[UnderstatPlayerStats]
) -> UnderstatPlayerStats | None:
    """Fuzzy-matches `player_name` against a team's Understat player list.
    Public: also used by dashboard/views/players.py to look up one specific
    player's current-season xG/xA row, not just internally here.
    """
    if not understat_players:
        return None
    names = [p.name for p in understat_players]
    match = process.extractOne(player_name, names, scorer=fuzz.WRatio)
    if match is None:
        return None
    _, score, index = match
    if score < PLAYER_NAME_FUZZY_THRESHOLD:
        return None
    return understat_players[index]


def _resolve_real_importance_weight(
    player_name: str,
    position: str,
    team_stats: list[HistoricalPlayerStats],
    team_totals: TeamOutputTotals,
    understat_players: list[UnderstatPlayerStats] | None = None,
    understat_totals: UnderstatTeamTotals | None = None,
) -> float | None:
    """Returns None (rather than a default) when there's no real stats match
    -- used by resolve_api_importance_weight (which then falls back to
    DEFAULT_API_IMPORTANCE_WEIGHT) and by resolve_team_importance_weights
    (which excludes the player entirely instead, see its own docstring).
    """
    if position != "defense" and understat_players:
        understat_player = find_understat_player(player_name, understat_players)
        if understat_player is not None and understat_totals is not None:
            weight = compute_attack_importance(
                understat_player.xg, understat_player.xa, understat_totals.xg_contribution_total
            )
            if weight is not None:
                return weight

    player = find_player_stats(player_name, team_stats)
    if player is None:
        return None

    if position == "defense":
        return compute_defense_importance(player.minutes, team_totals.max_minutes, player.rating)
    return compute_attack_importance(player.goals, player.assists, team_totals.goal_contribution_total)


def resolve_api_importance_weight(
    player_name: str,
    position: str,
    team_stats: list[HistoricalPlayerStats],
    team_totals: TeamOutputTotals,
    understat_players: list[UnderstatPlayerStats] | None = None,
    understat_totals: UnderstatTeamTotals | None = None,
) -> float:
    """Never returns None -- falls back to DEFAULT_API_IMPORTANCE_WEIGHT
    whenever there's no name match or not enough data to compute a real
    number (new signing, unresolved team, empty roster fetch), same
    degrade-gracefully contract as the rest of this app's API integrations.
    """
    weight = _resolve_real_importance_weight(
        player_name, position, team_stats, team_totals, understat_players, understat_totals
    )
    return weight if weight is not None else DEFAULT_API_IMPORTANCE_WEIGHT


def resolve_team_importance_weights(
    squad: list,
    team_stats: list[HistoricalPlayerStats],
    team_totals: TeamOutputTotals,
    understat_players: list[UnderstatPlayerStats] | None = None,
    understat_totals: UnderstatTeamTotals | None = None,
) -> dict[str, float]:
    """A real importance_weight per squad member with actual stats coverage
    -- unlike resolve_api_importance_weight, players with no stats match are
    left out entirely rather than defaulted to DEFAULT_API_IMPORTANCE_WEIGHT,
    since a flat 0.5 for every unmatched player would be meaningless noise
    for star-player detection (dashboard/components.py::compute_automatic_stars).

    `squad` is a list of ingest.squad.SquadPlayer (not imported here to
    avoid a circular import -- squad.py doesn't depend on this module).
    """
    weights: dict[str, float] = {}
    for player in squad:
        position = "defense" if player.position in ("Goalkeeper", "Defence") else "attack"
        weight = _resolve_real_importance_weight(
            player.name, position, team_stats, team_totals, understat_players, understat_totals
        )
        if weight is not None:
            weights[player.name] = weight
    return weights


def resolve_team_historical_stats(team_name: str, league_code: str, season: int) -> tuple[str, list]:
    """Picks the best available per-player season-stats source for a team --
    Understat (goals/assists/xG/xA/npxG/shots/key passes/cards/appearances)
    preferred for its 5 covered leagues (fresher, no season restriction --
    see understat_client.py's module docstring), falling back to
    API-Football (adds saves/tackles/rating/nationality, but capped to
    player_stats.AVAILABLE_SEASONS and currently degraded by an account
    suspension) when Understat has nothing for this team/season or doesn't
    cover the league at all.

    Returns (source, players) where source is "understat" or
    "api_football" -- players may be empty either way; callers treat an
    empty list as "no stats found" regardless of which source produced it,
    same degrade-gracefully contract as both underlying fetch functions.
    """
    if league_code in UNDERSTAT_LEAGUE_SLUG:
        understat_players = fetch_team_season(team_name, league_code, season)
        if understat_players:
            return "understat", understat_players

    return "api_football", fetch_team_player_stats(team_name, league_code, season)
