"""Injury sync -- the bonus source layered on top of the reliable
config/injuries.yaml manual list.

Automatic injury detection only exists for ESPN-backed leagues (MLS):
football-data.org (this app's source for every other league) has no
injuries endpoint of its own, and the API-Football-based fallback this used
to have for the other leagues was removed once that account became
permanently unusable, not just temporarily suspended -- see
ingest/player_importance.py's module docstring. Every other league relies
solely on config/injuries.yaml (manual) and AI-parsed chat notes (see
prediction/service.py::get_injuries_for_team) for injuries now.

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
from soccer_predictor.ingest import asa_client, espn_client, player_importance
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
    return 0, 0


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
