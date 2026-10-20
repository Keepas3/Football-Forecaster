"""Injury sync -- the bonus source layered on top of the reliable
config/injuries.yaml manual list.

Automatic injury detection exists for two places only: MLS (ESPN's roster
data) and the Premier League (the Fantasy Premier League feed, see
ingest/fpl_client.py -- unofficial, so treated as best effort).
football-data.org (this app's source for the other leagues) has no
injuries endpoint of its own, ESPN's injury fields are empty for them, and
the API-Football-based fallback this used to have was removed once that
account became permanently unusable -- see ingest/player_importance.py's
module docstring. Every other league relies solely on config/injuries.yaml
(manual) and AI-parsed chat notes (see prediction/service.py::get_injuries_for_team).

For MLS, the importance_weight each detected injury gets is looked up from
American Soccer Analysis (ingest/asa_client.py) via
ingest/player_importance.py, falling back to a flat default when there's
truly no data to compute from (new signing, unresolved name, fetch
failure) -- same "best-effort bonus, never a hard dependency" contract as
everywhere else in this app's API integrations.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import asa_client, espn_client, fpl_client, player_importance, understat_client
from soccer_predictor.storage.models import Team
from soccer_predictor.storage.repository import replace_injuries


def sync_injuries_to_db(session: Session, league: League, season_year: int) -> tuple[int, int]:
    """Fetches + stores injuries for every team in `league`. Returns
    (teams_synced, teams_skipped) -- a deliberate (0, 0) no-op for any
    league that isn't ESPN-backed, since there's no automatic injury source
    for those anymore (see this module's docstring).
    """
    if league.data_source == "espn":
        return _sync_injuries_to_db_espn(session, league)
    if league.code in FPL_LEAGUES:
        return sync_fpl_injuries(session, league)
    return 0, 0


# The Fantasy Premier League feed (ingest/fpl_client.py) covers this league only.
FPL_LEAGUES = {"EPL"}
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
