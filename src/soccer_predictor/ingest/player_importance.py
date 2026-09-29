"""Turns a player's own real stats into an injury importance_weight
(model/injury_adjustment.py's per-player [0, 1] multiplier input) -- so
losing a team's top scorer hurts a prediction more than losing a rarely-used
squad player, instead of every injury counting identically.

DEFAULT_API_IMPORTANCE_WEIGHT lives here (not injuries.py, which imports it
back) so injuries.py can depend on this module without a circular import.

Two possible sources, never mixed for one player: Understat
(ingest/understat_client.py, 5 European leagues) or American Soccer
Analysis (ingest/asa_client.py, MLS only) -- whichever one covers the
league in question. API-Football used to be a third source here (and the
primary one, at that), but that account is gone, not just temporarily
suspended, so it was removed entirely rather than left as permanent dead
code. UCL/EURO/World Cup have no per-player stats source at all now -- every
weight for those leagues falls back to DEFAULT_API_IMPORTANCE_WEIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz, process

from soccer_predictor.ingest.asa_client import AsaPlayerStats
from soccer_predictor.ingest.asa_client import fetch_team_season as fetch_asa_team_season
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

# Looser than team_mapper.py's 90 -- these sources' own name spellings can
# differ slightly from this app's canonical names ("Alexey" vs "Aleksey"),
# similar to why note_parser.py's chat-note matching (85) is also looser
# than team-name matching.
PLAYER_NAME_FUZZY_THRESHOLD = 80


@dataclass
class UnderstatTeamTotals:
    xg_contribution_total: float
    max_minutes: int


@dataclass
class AsaTeamTotals:
    xg_contribution_total: float
    max_minutes: int


def aggregate_understat_team_output(players: list[UnderstatPlayerStats]) -> UnderstatTeamTotals:
    total = sum((p.xg or 0) + (p.xa or 0) * 0.7 for p in players)
    max_minutes = max((p.minutes or 0 for p in players), default=0)
    return UnderstatTeamTotals(xg_contribution_total=total, max_minutes=max_minutes)


def aggregate_asa_team_output(players: list[AsaPlayerStats]) -> AsaTeamTotals:
    total = sum((p.xg or 0) + (p.xa or 0) * 0.7 for p in players)
    max_minutes = max((p.minutes or 0 for p in players), default=0)
    return AsaTeamTotals(xg_contribution_total=total, max_minutes=max_minutes)


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


def find_asa_player(player_name: str, asa_players: list[AsaPlayerStats]) -> AsaPlayerStats | None:
    """Fuzzy-matches `player_name` against a team's American Soccer Analysis
    player list. Public: also used by dashboard/views/players.py, same
    pattern as find_understat_player.
    """
    if not asa_players:
        return None
    names = [p.name for p in asa_players]
    match = process.extractOne(player_name, names, scorer=fuzz.WRatio)
    if match is None:
        return None
    _, score, index = match
    if score < PLAYER_NAME_FUZZY_THRESHOLD:
        return None
    return asa_players[index]


def _resolve_real_importance_weight(
    player_name: str,
    position: str,
    understat_players: list[UnderstatPlayerStats] | None = None,
    understat_totals: UnderstatTeamTotals | None = None,
    asa_players: list[AsaPlayerStats] | None = None,
    asa_totals: AsaTeamTotals | None = None,
) -> float | None:
    """Returns None (rather than a default) when there's no real stats match
    -- used by resolve_api_importance_weight (which then falls back to
    DEFAULT_API_IMPORTANCE_WEIGHT) and by resolve_team_importance_weights
    (which excludes the player entirely instead, see its own docstring).

    Unlike the old API-Football-backed version, defense positions ARE
    covered here now: compute_defense_importance only strictly needs
    minutes (rating is optional, see model/player_importance.py), and both
    Understat and ASA carry that.
    """
    if understat_players and understat_totals is not None:
        player = find_understat_player(player_name, understat_players)
        if player is not None:
            if position == "defense":
                weight = compute_defense_importance(player.minutes, understat_totals.max_minutes, None)
            else:
                weight = compute_attack_importance(player.xg, player.xa, understat_totals.xg_contribution_total)
            if weight is not None:
                return weight

    if asa_players and asa_totals is not None:
        player = find_asa_player(player_name, asa_players)
        if player is not None:
            if position == "defense":
                weight = compute_defense_importance(player.minutes, asa_totals.max_minutes, None)
            else:
                weight = compute_attack_importance(player.xg, player.xa, asa_totals.xg_contribution_total)
            if weight is not None:
                return weight

    return None


def resolve_api_importance_weight(
    player_name: str,
    position: str,
    understat_players: list[UnderstatPlayerStats] | None = None,
    understat_totals: UnderstatTeamTotals | None = None,
    asa_players: list[AsaPlayerStats] | None = None,
    asa_totals: AsaTeamTotals | None = None,
) -> float:
    """Never returns None -- falls back to DEFAULT_API_IMPORTANCE_WEIGHT
    whenever there's no name match or not enough data to compute a real
    number (new signing, unresolved team, empty roster fetch, or a league
    with no stats source at all -- UCL/EURO/WC), same degrade-gracefully
    contract as the rest of this app's integrations.
    """
    weight = _resolve_real_importance_weight(
        player_name, position, understat_players, understat_totals, asa_players, asa_totals
    )
    return weight if weight is not None else DEFAULT_API_IMPORTANCE_WEIGHT


def resolve_team_importance_weights(
    squad: list,
    understat_players: list[UnderstatPlayerStats] | None = None,
    understat_totals: UnderstatTeamTotals | None = None,
    asa_players: list[AsaPlayerStats] | None = None,
    asa_totals: AsaTeamTotals | None = None,
) -> dict[str, float]:
    """A real importance_weight per squad member with actual stats coverage
    -- unlike resolve_api_importance_weight, players with no stats match are
    left out entirely rather than defaulted to DEFAULT_API_IMPORTANCE_WEIGHT,
    since a flat 0.5 for every unmatched player would be meaningless noise.

    `squad` is a list of ingest.squad.SquadPlayer (not imported here to
    avoid a circular import -- squad.py doesn't depend on this module).
    """
    weights: dict[str, float] = {}
    for player in squad:
        position = "defense" if player.position in ("Goalkeeper", "Defence") else "attack"
        weight = _resolve_real_importance_weight(
            player.name, position, understat_players, understat_totals, asa_players, asa_totals
        )
        if weight is not None:
            weights[player.name] = weight
    return weights


def resolve_team_goals_and_assists(
    squad: list,
    understat_players: list[UnderstatPlayerStats] | None = None,
    asa_players: list[AsaPlayerStats] | None = None,
) -> dict[str, tuple[int, int]]:
    """Maps each squad member (keyed by their own SquadPlayer.name, not the
    stats source's spelling of it) to their season (goals, assists) --
    used by dashboard/components.py::compute_top_scorer_and_assister for
    the automatic star/top-assister badges. Fuzzy-matched the same way
    resolve_team_importance_weights is, and for the same reason: a stats
    source's own name spelling can differ slightly from the squad's
    (confirmed live -- Understat's "Alexey Miranchuk" vs this app's
    ESPN-sourced "Aleksey Miranchuk"). Players with no stats match are left
    out entirely, not defaulted to (0, 0), so a whole team with no stats
    coverage correctly produces no badges rather than a false "0 goals is
    the most goals" winner.
    """
    result: dict[str, tuple[int, int]] = {}
    for player in squad:
        stats_row = None
        if understat_players:
            stats_row = find_understat_player(player.name, understat_players)
        elif asa_players:
            stats_row = find_asa_player(player.name, asa_players)
        if stats_row is not None:
            result[player.name] = (stats_row.goals or 0, stats_row.assists or 0)
    return result


def resolve_team_historical_stats(team_name: str, league_code: str, season: int) -> tuple[str, list]:
    """Picks the best available per-player season-stats source for a team --
    Understat (goals/assists/xG/xA/npxG/shots/key passes/cards/appearances)
    for its 5 covered leagues (fresher, no season restriction -- see
    understat_client.py's module docstring); American Soccer Analysis
    (goals/assists/xG/xA/shots/key passes/points_added, keyless -- see
    asa_client.py) for MLS specifically. UCL/EURO/World Cup have no stats
    source at all -- API-Football used to fill that gap but that account is
    gone, not just suspended, so there's no fallback left to attempt.

    Returns (source, players) where source is "understat", "asa", or
    "none" -- players may be empty either way; callers treat an empty list
    as "no stats found" regardless of which source produced it, same
    degrade-gracefully contract as every underlying fetch function.
    """
    if league_code in UNDERSTAT_LEAGUE_SLUG:
        return "understat", fetch_team_season(team_name, league_code, season)

    if league_code == "MLS":
        return "asa", fetch_asa_team_season(team_name, season)

    return "none", []
