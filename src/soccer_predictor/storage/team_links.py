"""The same club or country across competitions.

Teams are stored per competition (storage.models.Team): "Barcelona" in La Liga and
"FC Barcelona" in the Champions League are separate rows, so each competition's
games sit under a different team id. This links them back up, so a team's page can
offer its games from every competition the app tracks.

* Clubs link through the football-data.org name both rows were created from (a
  shared "api" alias such as "FC Barcelona"), or an identical normalized name.
* National teams link through an identical normalized name, after a small map of
  the spellings that differ between the sources.
* A club never links to a country (see League.team_type), and a team is never
  linked to itself or to another team of its own competition.
"""

from __future__ import annotations

import datetime as dt
import unicodedata

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.storage.models import Team, TeamAlias
from soccer_predictor.storage.repository import fixtures_for_team, matches_for_team

# Normalized spellings that name the same national team in different sources
# (the real mismatches in the data). Former states -- West Germany, Yugoslavia,
# Czechoslovakia -- are separate teams on purpose and are not mapped.
NATIONAL_NAME_VARIANTS = {
    "turkiye": "turkey",
    "czech republic": "czechia",
    "bosnia and herzegovina": "bosnia-herzegovina",
    "ireland": "republic of ireland",
}


def normalize_team_name(name: str) -> str:
    """Lowercase, without accents, so "Türkiye" and "Turkiye" compare equal."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().casefold().strip()


def _key(name: str, team_type: str) -> str:
    normalized = normalize_team_name(name)
    return NATIONAL_NAME_VARIANTS.get(normalized, normalized) if team_type == "national" else normalized


def linked_teams(session: Session, team: Team, leagues: dict[str, League]) -> list[Team]:
    """This team's rows in the other competitions of the same kind, ordered by
    competition code. Empty for a team that plays in only one."""
    own = leagues.get(team.league_code)
    team_type = own.team_type if own else "club"
    other_codes = [
        code for code, league in leagues.items() if league.team_type == team_type and code != team.league_code
    ]
    if not other_codes:
        return []
    candidates = list(session.scalars(select(Team).where(Team.league_code.in_(other_codes))))

    wanted_key = _key(team.canonical_name, team_type)
    wanted_aliases: set[str] = set()
    candidate_aliases: dict[int, set[str]] = {}
    if team_type == "club":
        wanted_aliases = set(
            session.scalars(select(TeamAlias.alias_text).where(TeamAlias.team_id == team.id, TeamAlias.source == "api"))
        )
        for alias_text, team_id in session.execute(
            select(TeamAlias.alias_text, TeamAlias.team_id).where(
                TeamAlias.source == "api", TeamAlias.league_code.in_(other_codes)
            )
        ):
            candidate_aliases.setdefault(team_id, set()).add(alias_text)

    linked = []
    for candidate in candidates:
        if _key(candidate.canonical_name, team_type) == wanted_key or (
            wanted_aliases and wanted_aliases & candidate_aliases.get(candidate.id, set())
        ):
            linked.append(candidate)
    return sorted(linked, key=lambda t: (t.league_code, t.id))


def _tagged(frame: pd.DataFrame, team: Team) -> pd.DataFrame:
    frame = frame.copy()
    frame["league_code"] = team.league_code
    frame["team_id_here"] = team.id
    return frame


def matches_for_teams(session: Session, teams: list[Team]) -> pd.DataFrame:
    """Results of every given team, newest first, each row tagged with its
    competition (`league_code`) and the team's own id there (`team_id_here`)."""
    frames = [_tagged(matches_for_team(session, t.id), t) for t in teams]
    columns = ["date", "home_team_id", "away_team_id", "home_goals", "away_goals", "league_code", "team_id_here"]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).sort_values("date", ascending=False, kind="stable").reset_index(drop=True)


def fixtures_for_teams(session: Session, teams: list[Team], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Upcoming fixtures of every given team, in date order, tagged like matches_for_teams."""
    frames = [_tagged(fixtures_for_team(session, t.id, start, end), t) for t in teams]
    columns = ["date", "home_team_id", "away_team_id", "status", "kickoff_utc", "league_code", "team_id_here"]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)
