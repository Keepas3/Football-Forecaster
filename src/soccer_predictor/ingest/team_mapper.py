"""Resolves team-name strings from any data source to one canonical team.

The same club is spelled differently in the historical CSVs ("Man United")
vs the live API ("Manchester United FC"). `config/team_aliases.yaml` seeds
the known spellings; `resolve()` does exact lookup first, then a fuzzy
fallback, and never silently guesses on a low-confidence match — those are
logged to `data/unmatched_teams.log` for manual review instead.
"""

from __future__ import annotations

import datetime as dt

import requests
from rapidfuzz import fuzz, process
from sqlalchemy.orm import Session

from soccer_predictor.config import DATA_DIR, League, load_team_aliases
from soccer_predictor.ingest import api_client, espn_client
from soccer_predictor.storage.models import Team
from soccer_predictor.storage.repository import (
    all_alias_texts,
    find_team_by_alias,
    get_or_create_team,
    update_espn_team_id,
    update_team_conference,
    update_team_crest,
    upsert_team_alias,
)

FUZZY_MATCH_THRESHOLD = 90  # 0-100; below this, treat as unmatched rather than guess.
UNMATCHED_LOG = DATA_DIR / "unmatched_teams.log"
TEAM_LIST_CACHE_TTL_SECONDS = 24 * 3600  # a competition's entrant list changes slowly


def seed_teams_and_aliases(session: Session, league_code: str) -> None:
    """Populates teams/team_aliases from config/team_aliases.yaml.

    Only processes rows scoped to `league_code` -- config/team_aliases.yaml
    is one flat file shared by every league, so without this filter seeding
    league A would also (mis-)tag every other league's clubs with league A.
    """
    for alias in load_team_aliases():
        if alias.league != league_code:
            continue
        team = get_or_create_team(session, alias.canonical_name, league_code)
        upsert_team_alias(session, team, alias.csv_name, source="csv")
        upsert_team_alias(session, team, alias.api_name, source="api")


def seed_teams_from_api(session: Session, league: League) -> int:
    """Populates teams/aliases for a league with no config/team_aliases.yaml
    entries (no CSV source to hand-map against -- see League.csv_code) by
    reading football-data.org's own team list for the competition instead.

    Each team's API name becomes both its canonical_name and its "api"
    alias, so later resolve(source="api", ...) calls exact-match immediately.
    Returns the number of teams seeded; 0 (never raises) if the API key is
    missing or the request fails, same degrade-gracefully pattern as every
    other live-API call in this app.
    """
    try:
        data = api_client.get(
            f"/competitions/{league.api_competition_id}/teams",
            cache_ttl_seconds=TEAM_LIST_CACHE_TTL_SECONDS,
        )
    except (api_client.MissingApiKey, requests.RequestException):
        return 0

    count = 0
    for entry in data.get("teams", []):
        name = entry.get("name")
        if not name:
            continue
        team = get_or_create_team(session, name, league.code)
        upsert_team_alias(session, team, name, source="api")
        update_team_crest(session, team.id, entry.get("crest"))
        count += 1
    return count


def seed_teams_from_espn(session: Session, league: League) -> int:
    """Populates teams for a league with data_source == "espn" (e.g. MLS --
    no football-data.org coverage at all, so seed_teams_from_api can't be
    used). ESPN keys everything by a stable numeric id, so this caches
    Team.espn_team_id directly rather than creating a "api" TeamAlias --
    fixture/result resolution for these leagues never needs name-based
    fuzzy matching (see ingest/fixtures.py's *_espn functions).

    Returns the number of teams seeded; 0 (never raises) on any failure.
    """
    count = 0
    for entry in espn_client.fetch_teams(league.espn_league_slug):
        team = get_or_create_team(session, entry.display_name, league.code)
        update_team_crest(session, team.id, entry.logo_url)
        update_espn_team_id(session, team.id, int(entry.espn_id))
        # Shares its on-disk cache with fetch_team_roster (same request) --
        # this doesn't cost an extra network call once squad/injuries have
        # already been synced for this team in the same refresh run, or
        # vice versa.
        # Conference ids are MLS-specific ("1"/"2"); in a group competition
        # the same field is a group id, which would be mislabeled as one
        # of those conferences -- and costs a request per team for nothing.
        if not league.has_groups:
            conference = espn_client.fetch_team_conference(league.espn_league_slug, entry.espn_id)
            update_team_conference(session, team.id, conference)
        count += 1
    return count


def seed_teams_from_names(session: Session, league_code: str, team_names: set[str]) -> int:
    """Populates teams/aliases for a league whose team names come straight
    from a static archive source (World Cup/Euro historical data -- see
    ingest/worldcup_archive.py, ingest/euro_archive.py) rather than
    config/team_aliases.yaml or a live API's own team-list endpoint.

    Each name becomes both its canonical_name and its "csv" alias, matching
    historical_csv.py's own alias source convention -- resolve(source="csv",
    ...) then exact-matches immediately once this has run. This bootstrap
    step is required: resolve()'s fuzzy fallback needs at least one existing
    alias to fuzzy-match against, so it can never seed a brand-new league
    from nothing on its own (see resolve()'s own docstring).
    """
    count = 0
    for name in sorted(team_names):
        team = get_or_create_team(session, name, league_code)
        upsert_team_alias(session, team, name, source="csv")
        count += 1
    return count


class UnresolvedTeamName(Exception):
    def __init__(self, name: str, source: str, best_match: str | None, score: float | None):
        self.name = name
        self.source = source
        self.best_match = best_match
        self.score = score
        message = f"Unresolved team name '{name}' (source={source})"
        if best_match is not None:
            message += f"; closest known alias '{best_match}' scored {score:.0f}/100"
        super().__init__(message)


def resolve(session: Session, name: str, source: str, league_code: str) -> int:
    """Returns a team_id for a team-name string from `source` ("csv"|"api").

    `league_code` scopes the fuzzy fallback to teams in that league only --
    with several leagues sharing one aliases table, an unscoped fuzzy match
    could resolve a genuinely new/misspelled team to a similarly-spelled
    team from a different league instead of correctly logging it as
    unresolved. Exact matches (checked first) don't need this: find_team_by_alias
    is already scoped to (alias_text, source, league_code).

    Raises UnresolvedTeamName (and logs it) if there's no exact match and no
    fuzzy match clears FUZZY_MATCH_THRESHOLD -- callers should treat this as
    a data-quality issue to fix in config/team_aliases.yaml, not swallow it.
    """
    exact = find_team_by_alias(session, name, source, league_code)
    if exact is not None:
        return exact.id

    candidates = all_alias_texts(session, source, league_code)
    if not candidates:
        _log_unmatched(name, source, None, None)
        raise UnresolvedTeamName(name, source, None, None)

    texts = [text for text, _ in candidates]
    match = process.extractOne(name, texts, scorer=fuzz.WRatio)
    if match is None:
        _log_unmatched(name, source, None, None)
        raise UnresolvedTeamName(name, source, None, None)

    matched_text, score, index = match
    if score < FUZZY_MATCH_THRESHOLD:
        _log_unmatched(name, source, matched_text, score)
        raise UnresolvedTeamName(name, source, matched_text, score)

    _, team_id = candidates[index]
    # Learn this spelling for next time so it becomes an exact match.
    team = session.get(Team, team_id)
    upsert_team_alias(session, team, name, source)
    return team_id


def _log_unmatched(name: str, source: str, best_match: str | None, score: float | None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(UNMATCHED_LOG, "a", encoding="utf-8") as fh:
        fh.write(
            f"{dt.datetime.now(dt.UTC).isoformat()} source={source} name={name!r} "
            f"best_match={best_match!r} score={score}\n"
        )
