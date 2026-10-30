"""Injury sync -- the bonus source layered on top of the reliable
config/injuries.yaml manual list.

Automatic injury detection exists for these, all unofficial and so treated as
best effort: the Premier League (the Fantasy Premier League feed, see
ingest/fpl_client.py), La Liga and the Champions League (Biwenger's fantasy
feed, see ingest/biwenger_client.py), and MLS, Bundesliga, Serie A and Ligue 1
(starting11.com's injury pages, see ingest/starting11_client.py). ESPN's roster
injuries (_sync_injuries_to_db_espn) remain for any other ESPN-backed league,
but they come back empty in practice, so MLS uses starting11.com instead.
football-data.org (this app's source for the other leagues) has no injuries
endpoint of its own, and the API-Football-based fallback this used to have
was removed once that account became permanently unusable -- see
ingest/player_importance.py's module docstring. Every other league relies
solely on config/injuries.yaml (manual) and AI-parsed chat notes (see
prediction/service.py::get_injuries_for_team).

For MLS, the importance_weight each detected injury gets is looked up from
American Soccer Analysis (ingest/asa_client.py) via
ingest/player_importance.py, falling back to a flat default when there's
truly no data to compute from (new signing, unresolved name, fetch
failure) -- same "best-effort bonus, never a hard dependency" contract as
everywhere else in this app's API integrations.
"""

from __future__ import annotations

import datetime as dt

from rapidfuzz import fuzz, process
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import (
    asa_client,
    biwenger_client,
    club_names,
    espn_client,
    fpl_client,
    player_importance,
    starting11_client,
    understat_client,
)
from soccer_predictor.storage.models import Team
from soccer_predictor.storage.repository import replace_injuries


def sync_injuries_to_db(session: Session, league: League, season_year: int) -> tuple[int, int]:
    """Fetches + stores injuries for every team in `league`. Returns
    (teams_synced, teams_skipped) -- a deliberate (0, 0) no-op for any
    league without an automatic injury source (see this module's docstring).
    """
    if league.code in STARTING11_LEAGUES:  # before ESPN: MLS is ESPN-backed, but ESPN's injury fields are empty
        return sync_starting11_injuries(session, league)
    if league.data_source == "espn":
        return _sync_injuries_to_db_espn(session, league)
    if league.code in FPL_LEAGUES:
        return sync_fpl_injuries(session, league)
    if league.code in BIWENGER_LEAGUES:
        return sync_biwenger_injuries(session, league)
    return 0, 0


# The Fantasy Premier League feed (ingest/fpl_client.py) covers this league only.
FPL_LEAGUES = {"EPL"}
# Biwenger's fantasy feed (ingest/biwenger_client.py) covers these.
BIWENGER_LEAGUES = set(biwenger_client.COMPETITIONS)
# starting11.com (ingest/starting11_client.py) covers these. It is a web page, not a
# feed, so it is read in the daily full refresh only, not the frequent quick one.
STARTING11_LEAGUES = set(starting11_client.COMPETITIONS)
# Leagues whose whole feed is one cached request, cheap enough for the frequent quick refresh.
SINGLE_REQUEST_LEAGUES = FPL_LEAGUES | BIWENGER_LEAGUES


def has_automatic_feed(league: League) -> bool:
    return (
        league.data_source == "espn"
        or league.code in SINGLE_REQUEST_LEAGUES
        or league.code in STARTING11_LEAGUES
    )
# A doubtful player's weight is scaled by how likely he is to miss the game; below
# this it would not move a prediction, so the entry is dropped.
MIN_INJURY_WEIGHT = 0.02
# A player Understat has no record of who has barely played this season (a
# reserve goalkeeper, a youngster) would otherwise get the flat default weight,
# far more than he matters; he gets this instead.
UNPLAYED_WEIGHT = 0.1
UNPLAYED_MINUTES = 300


def sync_fpl_injuries(session: Session, league: League) -> tuple[int, int]:
    """Premier League injuries, doubts and suspensions from the Fantasy Premier
    League feed. Returns (clubs synced, clubs skipped); (0, 0) and nothing
    changed if the feed can't be reached -- yesterday's data beats wiping it.

    Each player's importance_weight comes from his Understat output (the same
    way MLS injuries are priced from ASA), falling back to a flat default for
    players it can't match; a doubtful player's weight is scaled by his chance
    of missing the game. Stored as source "api" and replaced on every sync.
    """
    from soccer_predictor.storage.repository import teams_for_league

    by_club = fpl_client.fetch_injuries_by_club()
    if by_club is None:
        return 0, 0
    teams = teams_for_league(session, league.code)
    id_by_name = {name: team_id for team_id, name in teams.items()}
    season = league.api_season_year(league.seasons[-1])

    synced = skipped = 0
    for fpl_club, injuries in by_club.items():
        our_name = fpl_client.resolve_team_name(fpl_club, list(id_by_name))
        if our_name is None:
            skipped += 1
            continue
        entries = []
        if injuries:  # only priced when the club has someone to price
            players = understat_client.fetch_team_season(our_name, league.code, season)
            totals = player_importance.aggregate_understat_team_output(players)
            for injury in injuries:
                if (
                    player_importance.find_understat_player(injury.player_name, players) is None
                    and injury.minutes < UNPLAYED_MINUTES
                ):
                    base_weight = UNPLAYED_WEIGHT
                else:
                    base_weight = player_importance.resolve_api_importance_weight(
                        injury.player_name,
                        injury.position_bucket,
                        understat_players=players,
                        understat_totals=totals,
                    )
                weight = base_weight * (1 - injury.chance_of_playing / 100)
                if weight < MIN_INJURY_WEIGHT:
                    continue
                doubtful = injury.status == "d"
                news = injury.news or ("Doubtful" if doubtful else "Out")
                # FPL's own news usually already says "75% chance of playing".
                chance = (
                    f" ({injury.chance_of_playing}% chance of playing)"
                    if doubtful and "chance of playing" not in news
                    else ""
                )
                entries.append(
                    {
                        "player_name": injury.player_name,
                        "position": injury.position_bucket,
                        "importance_weight": weight,
                        "note": f"FPL: {news}{chance}",
                    }
                )
        replace_injuries(session, id_by_name[our_name], source="api", injuries=entries)
        synced += 1
    return synced, skipped


# A club must match an Understat club this closely to borrow its players' stats;
# below it (a Portuguese or Turkish side Understat doesn't track) the lookalike
# is more likely wrong than right.
UNDERSTAT_CLUB_MATCH_THRESHOLD = 85


def _understat_squad(
    biwenger_club: str, our_name: str, league: League, season: int
) -> list[understat_client.UnderstatPlayerStats]:
    """The Understat players of a club, or [] if Understat doesn't track it.
    La Liga is covered directly; the Champions League is not, so a club's stats
    are looked up in the domestic league it plays in (the five Understat covers),
    which is where its players' output is recorded."""
    if league.code in understat_client.UNDERSTAT_LEAGUE_SLUG:
        return understat_client.fetch_team_season(our_name, league.code, season)
    best_score, best_squad = 0.0, []
    for code in understat_client.UNDERSTAT_LEAGUE_SLUG:
        squads = understat_client.fetch_all_teams_totals(code, season)
        for name in (biwenger_club, our_name):  # one of the two spellings is usually Understat's own
            match = process.extractOne(
                name, list(squads), scorer=fuzz.WRatio, processor=biwenger_client.fold_name
            )
            if match is not None and match[1] > best_score:
                best_score, best_squad = match[1], squads[match[0]]
    return best_squad if best_score >= UNDERSTAT_CLUB_MATCH_THRESHOLD else []


def _biwenger_base_weight(injury, players, totals) -> float:
    """How much his club loses without him, before any doubt is applied: his
    Understat output share where Understat knows him; otherwise a guess from the
    feed itself -- none if he hasn't played, else the default scaled by how
    close his market price is to his club's most expensive player."""
    if players and player_importance.find_understat_player(injury.player_name, players) is not None:
        return player_importance.resolve_api_importance_weight(
            injury.player_name, injury.position_bucket, understat_players=players, understat_totals=totals
        )
    if injury.games_played == 0:
        return UNPLAYED_WEIGHT
    return max(UNPLAYED_WEIGHT, player_importance.DEFAULT_API_IMPORTANCE_WEIGHT * injury.price_share)


def _biwenger_note(injury) -> str:
    news = injury.news
    if injury.status == "doubt":
        chance = f"{injury.chance_of_playing}% chance of playing"
        news = f"{news} - {chance}" if news else chance
    return f"Biwenger: {news or 'Out'}"


def sync_biwenger_injuries(session: Session, league: League) -> tuple[int, int]:
    """La Liga or Champions League injuries, doubts and suspensions from
    Biwenger's feed. Returns (clubs synced, clubs skipped); (0, 0) and nothing
    changed if the feed can't be reached -- yesterday's data beats wiping it.

    Priced like the Premier League's (see sync_fpl_injuries): by Understat
    output where it knows the player, else a guess from the feed; a doubtful
    player's weight is scaled by his chance of missing the game. Stored as
    source "api" and replaced on every sync.
    """
    from soccer_predictor.storage.repository import teams_for_league

    by_club = biwenger_client.fetch_injuries_by_club(league.code)
    if by_club is None:
        return 0, 0
    teams = teams_for_league(session, league.code)
    id_by_name = {name: team_id for team_id, name in teams.items()}
    season = league.api_season_year(league.seasons[-1])

    synced = skipped = 0
    claimed: set[str] = set()
    for club, injuries in by_club.items():
        our_name = biwenger_client.resolve_team_name(club, list(id_by_name))
        if our_name is None or our_name in claimed:  # unknown club, or two of theirs for one of ours
            skipped += 1
            continue
        claimed.add(our_name)
        entries = []
        if injuries:  # only priced when the club has someone to price
            players = _understat_squad(club, our_name, league, season)
            totals = player_importance.aggregate_understat_team_output(players)
            for injury in injuries:
                weight = _biwenger_base_weight(injury, players, totals) * (1 - injury.chance_of_playing / 100)
                if weight < MIN_INJURY_WEIGHT:
                    continue
                entries.append(
                    {
                        "player_name": injury.player_name,
                        "position": injury.position_bucket,
                        "importance_weight": weight,
                        "note": _biwenger_note(injury),
                    }
                )
        replace_injuries(session, id_by_name[our_name], source="api", injuries=entries)
        synced += 1
    return synced, skipped


def _starting11_note(injury) -> str:
    note = injury.problem or ("Suspended" if injury.status == "Suspended" else "Out")
    if injury.status == "Suspended":
        note = f"{note} - Suspended"
    elif injury.status == "Doubtful":
        note = f"{note} - {injury.chance_of_playing}% chance of playing"
    return f"Starting XI: {note}"


def sync_starting11_injuries(session: Session, league: League) -> tuple[int, int]:
    """MLS, Bundesliga, Serie A or Ligue 1 injuries, doubts and suspensions from
    starting11.com. Returns (clubs synced, clubs skipped); (0, 0) and nothing
    changed if the page can't be fetched or read -- yesterday's data beats
    wiping it. Clubs the page doesn't list are left as they are.

    Priced from the player's output where his club's stats are available --
    American Soccer Analysis for MLS, Understat for the European leagues -- and
    otherwise by the small UNPLAYED_WEIGHT: the page gives only a surname, so a
    player the stats don't know is far more likely a fringe player than a star,
    and a wrongly large penalty hurts more than a small one. A doubtful player's
    weight is scaled by his chance of missing the game. Stored as source "api"
    and replaced on every sync.
    """
    from soccer_predictor.storage.repository import teams_for_league

    by_club = starting11_client.fetch_injuries_by_club(league.code)
    if by_club is None:
        return 0, 0
    teams = teams_for_league(session, league.code)
    id_by_name = {name: team_id for team_id, name in teams.items()}
    season = league.api_season_year(league.seasons[-1])

    synced = skipped = 0
    claimed: set[str] = set()
    for club, injuries in by_club.items():
        our_name = starting11_client.resolve_team_name(club, list(id_by_name))
        if our_name is None or our_name in claimed:  # unknown club, or two of theirs for one of ours
            skipped += 1
            continue
        claimed.add(our_name)
        entries = []
        if injuries:  # only priced when the club has someone to price
            if league.code == "MLS":
                asa_players = asa_client.fetch_team_season(our_name, dt.date.today().year)
                asa_totals = player_importance.aggregate_asa_team_output(asa_players)
                understat_players, understat_totals = [], None
            else:
                understat_players = understat_client.fetch_team_season(our_name, league.code, season)
                understat_totals = player_importance.aggregate_understat_team_output(understat_players)
                asa_players, asa_totals = [], None
            for injury in injuries:
                # Starting 11 spells "Pongračić" where a stats site may say "Pongracic".
                spellings = dict.fromkeys((injury.player_name, club_names.strip_accents(injury.player_name)))
                known_as = next(
                    (
                        spelling
                        for spelling in spellings
                        if (understat_players and player_importance.find_understat_player(spelling, understat_players))
                        or (asa_players and player_importance.find_asa_player(spelling, asa_players))
                    ),
                    None,
                )
                if known_as is None:
                    base_weight = UNPLAYED_WEIGHT
                else:
                    base_weight = player_importance.resolve_api_importance_weight(
                        known_as,
                        injury.position_bucket,
                        understat_players=understat_players,
                        understat_totals=understat_totals,
                        asa_players=asa_players,
                        asa_totals=asa_totals,
                    )
                weight = base_weight * (1 - injury.chance_of_playing / 100)
                if weight < MIN_INJURY_WEIGHT:
                    continue
                entries.append(
                    {
                        "player_name": injury.player_name,
                        "position": injury.position_bucket,
                        "importance_weight": weight,
                        "note": _starting11_note(injury),
                    }
                )
        replace_injuries(session, id_by_name[our_name], source="api", injuries=entries)
        synced += 1
    return synced, skipped


def _sync_injuries_to_db_espn(session: Session, league: League) -> tuple[int, int]:
    """ESPN-backed injury sync (e.g. MLS). No position-resolution step is
    needed here -- ESPN's own roster entries already carry a real position
    bucket per player.
    """
    from soccer_predictor.storage.repository import teams_for_league

    synced = 0
    skipped = 0
    for team_id, team_name in teams_for_league(session, league.code).items():
        team = session.get(Team, team_id)
        if team is None or team.espn_team_id is None:
            skipped += 1
            continue

        roster, espn_injuries = espn_client.fetch_team_roster(league.espn_league_slug, str(team.espn_team_id))
        if not espn_injuries:
            replace_injuries(session, team_id, source="api", injuries=[])
            synced += 1
            continue

        # Only fetched when this team actually has an injury to price --
        # most teams have none on a given sync, so this keeps the extra
        # call rare rather than doubling every sync's cost.
        asa_players = asa_client.fetch_team_season(team_name, dt.date.today().year)
        asa_totals = player_importance.aggregate_asa_team_output(asa_players)

        entries = []
        for injury in espn_injuries:
            importance_weight = player_importance.resolve_api_importance_weight(
                injury.player_name,
                injury.position_bucket,
                asa_players=asa_players,
                asa_totals=asa_totals,
            )
            entries.append(
                {
                    "player_name": injury.player_name,
                    "position": injury.position_bucket,
                    "importance_weight": importance_weight,
                    "note": injury.note,
                }
            )
        replace_injuries(session, team_id, source="api", injuries=entries)
        synced += 1
    return synced, skipped
