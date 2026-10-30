"""The starting XI named for an upcoming match, and the squad's usual XI it is
compared with (model/lineup_adjustment.py).

The named XI comes from ESPN's match summary, the only lineup source this app
has -- unofficial like the rest of ESPN's API, and published only about an hour
before kickoff, so every function here degrades to None / [] rather than
raising, and the prediction simply stays on its injury-based estimate.

The usual XI comes from the same per-player season stats that price injuries
(Understat for the five European leagues, and for a Champions League club the
domestic league it plays in; American Soccer Analysis for MLS). Other
competitions have no player stats, so a lineup there cannot be rated.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from soccer_predictor.config import League
from soccer_predictor.ingest import asa_client, espn_client, injuries, player_importance, understat_client
from soccer_predictor.model.lineup_adjustment import ExpectedPlayer
from soccer_predictor.model.player_importance import (
    MIN_IMPORTANCE_WEIGHT,
    compute_attack_importance,
    compute_defense_importance,
)

STARTERS_PER_SIDE = 11
# Understat's position strings are space separated ("D M S"): the first is the
# player's main position, a trailing "S" marks appearances as a substitute.
_UNDERSTAT_DEFENSE = {"GK", "D"}
_ASA_DEFENSE = {"GK", "CB", "FB"}


@dataclass(frozen=True)
class ConfirmedXi:
    event_id: str
    home: list[str]
    away: list[str]
    home_formation: str | None = None
    away_formation: str | None = None
    # The spellings ESPN gave the two clubs, to tell which of our teams each XI belongs to.
    names: tuple[str, str] = field(default=("", ""))


def _starters(info: espn_client.TeamMatchInfo) -> list[str]:
    starters = [p for p in info.players if p.starter]
    # ESPN numbers the formation slots 1 (goalkeeper) upward; keep that order.
    starters.sort(key=lambda p: p.formation_place if p.formation_place else 99)
    return [p.name for p in starters]


def confirmed_xi(detail: espn_client.MatchDetail | None, home_name: str, away_name: str) -> ConfirmedXi | None:
    """The XI of each of OUR home and away teams, or None unless ESPN lists
    exactly eleven starters for both sides (a half-published sheet is not a lineup).
    ESPN's home/away can differ from ours at a neutral venue, so the sides are
    matched to our team names rather than trusted."""
    if detail is None:
        return None
    side_a, side_b = _starters(detail.home), _starters(detail.away)
    if len(side_a) != STARTERS_PER_SIDE or len(side_b) != STARTERS_PER_SIDE:
        return None
    straight = espn_client._name_score(home_name, detail.home.name) + espn_client._name_score(away_name, detail.away.name)
    swapped = espn_client._name_score(home_name, detail.away.name) + espn_client._name_score(away_name, detail.home.name)
    if swapped > straight:
        return ConfirmedXi(
            detail.event_id, side_b, side_a, detail.away.formation, detail.home.formation,
            (detail.away.name, detail.home.name),
        )
    return ConfirmedXi(
        detail.event_id, side_a, side_b, detail.home.formation, detail.away.formation,
        (detail.home.name, detail.away.name),
    )


def find_event(
    league: League, match_date: dt.date, home_name: str, away_name: str, stored_event_id: str | None = None
) -> str | None:
    """ESPN's id for this match: the one already stored, else found by team names
    on the scoreboards around `match_date` (kept short-lived: it must see a match
    turn from scheduled to under way)."""
    if stored_event_id:
        return stored_event_id
    if not league.espn_league_slug:
        return None
    found = espn_client.find_event_by_names(
        league.espn_league_slug, match_date, home_name, away_name, cache_ttl_seconds=600
    )
    return found[0] if found else None


def fetch_confirmed_xi(league: League, event_id: str, home_name: str, away_name: str) -> ConfirmedXi | None:
    """The named XI right now (never from the cache), or None if it is not out yet."""
    if not league.espn_league_slug:
        return None
    detail = espn_client.fetch_match_detail(league.espn_league_slug, event_id, state="pre", cache_ttl_seconds=0)
    return confirmed_xi(detail, home_name, away_name)


def _bucket(position: str | None, defense_positions: set[str]) -> tuple[str, bool]:
    """("attack" | "defense", is_keeper) from a stats source's position string."""
    tokens = [t for t in (position or "").upper().split() if t != "S"]
    main = tokens[0] if tokens else ""
    return ("defense" if main in defense_positions else "attack"), main == "GK"


def _expected_player(player, bucket: tuple[str, bool], totals) -> ExpectedPlayer:
    position, is_keeper = bucket
    if position == "defense":
        weight = compute_defense_importance(player.minutes, totals.max_minutes, None)
    else:
        weight = compute_attack_importance(player.xg, player.xa, totals.xg_contribution_total)
    return ExpectedPlayer(
        name=player.name,
        position=position,
        weight=weight if weight is not None else MIN_IMPORTANCE_WEIGHT,
        minutes=player.minutes or 0,
        is_keeper=is_keeper,
    )


def expected_squad(team_name: str, league: League) -> list[ExpectedPlayer]:
    """Every player with stats for `team_name` this season, priced the way an
    injury to him would be; [] when the league has no player-stats source or the
    club cannot be found, in which case a lineup there cannot be rated."""
    season = league.api_season_year(league.seasons[-1])
    if league.code == "MLS":
        players = asa_client.fetch_team_season(team_name, season)
        totals = player_importance.aggregate_asa_team_output(players)
        return [_expected_player(p, _bucket(p.position, _ASA_DEFENSE), totals) for p in players]
    if league.code in understat_client.UNDERSTAT_LEAGUE_SLUG or league.code == "UCL":
        players = injuries.understat_squad(team_name, team_name, league, season)
        totals = player_importance.aggregate_understat_team_output(players)
        return [_expected_player(p, _bucket(p.position, _UNDERSTAT_DEFENSE), totals) for p in players]
    return []
