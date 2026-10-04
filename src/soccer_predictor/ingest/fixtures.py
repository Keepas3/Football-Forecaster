"""Fetches upcoming fixtures from football-data.org and syncs them to SQLite."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import api_client, espn_client
from soccer_predictor.ingest.team_mapper import UnresolvedTeamName, resolve
from soccer_predictor.storage.models import Team
from soccer_predictor.storage.repository import (
    get_or_create_team,
    team_by_espn_id,
    teams_for_league,
    update_espn_team_id,
    update_team_crest,
    upsert_fixture,
    upsert_match,
)

FIXTURES_CACHE_TTL_SECONDS = 6 * 3600

# How many days ahead sync_fixtures_to_db_espn scans -- ESPN's scoreboard
# only supports one date per call (a YYYYMMDD-YYYYMMDD range returned HTTP
# 400 in testing), so this is a real day-by-day loop; 14 matches this app's
# existing "upcoming fixtures" window convention elsewhere.
ESPN_UPCOMING_FIXTURE_WINDOW_DAYS = 14
# A past season's results never change once the season is over, but the
# *current* season's are refetched often (see refresh_live_data.py) --
# short enough to pick up yesterday's results without wasting the day's
# football-data.org quota on every dashboard page load.
RESULTS_CACHE_TTL_SECONDS = 6 * 3600


def _football_data_group(match: dict, league: League) -> str | None:
    """"Group A" for a group-stage match of a group competition (World Cup,
    Euros), from football-data.org's `stage` ("GROUP_STAGE") and `group`
    ("GROUP_A") fields -- None for knockout matches or any league without
    groups, so only genuine group matches reach the group tables.
    """
    if not league.has_groups or match.get("stage") != "GROUP_STAGE":
        return None
    raw = match.get("group")
    if not raw:
        return None
    return raw.replace("_", " ").title()


def fetch_upcoming_fixtures(league: League) -> list[dict]:
    """Raw API rows for this league's next scheduled matches."""
    data = api_client.get(
        f"/competitions/{league.api_competition_id}/matches",
        params={"status": "SCHEDULED"},
        cache_ttl_seconds=FIXTURES_CACHE_TTL_SECONDS,
    )
    return data.get("matches", [])


def sync_fixtures_to_db(session: Session, league: League) -> tuple[int, int]:
    """Resolves team names and upserts fixtures. Returns (synced, skipped)."""
    synced = 0
    skipped = 0
    for match in fetch_upcoming_fixtures(league):
        home_name = match["homeTeam"]["name"]
        away_name = match["awayTeam"]["name"]
        try:
            home_team_id = resolve(session, home_name, source="api", league_code=league.code)
            away_team_id = resolve(session, away_name, source="api", league_code=league.code)
        except UnresolvedTeamName:
            skipped += 1
            continue

        update_team_crest(session, home_team_id, match["homeTeam"].get("crest"))
        update_team_crest(session, away_team_id, match["awayTeam"].get("crest"))

        match_datetime_utc = dt.datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))
        upsert_fixture(
            session,
            league_code=league.code,
            date=match_datetime_utc.date(),
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            status=match["status"],
            kickoff_utc=match_datetime_utc.replace(tzinfo=None),
            group_name=_football_data_group(match, league),
        )
        synced += 1
    return synced, skipped


def fetch_finished_matches(league: League, season_year: int) -> list[dict]:
    """Raw API rows for one season of this league's completed matches.

    Only used for leagues with no football-data.co.uk CSV (see
    League.csv_code) -- those are the only ones missing a `matches` history
    to build standings/train from, since domestic leagues get theirs from
    the CSV pipeline instead (scripts/fetch_historical_data.py).
    """
    data = api_client.get(
        f"/competitions/{league.api_competition_id}/matches",
        params={"status": "FINISHED", "season": season_year},
        cache_ttl_seconds=RESULTS_CACHE_TTL_SECONDS,
    )
    return data.get("matches", [])


def sync_results_to_db(session: Session, league: League, season: str) -> tuple[int, int]:
    """Resolves team names and upserts finished matches for one season.
    Returns (synced, skipped). Mirrors sync_fixtures_to_db, but writes to
    the `matches` table (standings/training history) instead of `fixtures`
    (the upcoming-games list).
    """
    synced = 0
    skipped = 0
    for match in fetch_finished_matches(league, league.api_season_year(season)):
        score = match.get("score", {}).get("fullTime", {})
        if score.get("home") is None or score.get("away") is None:
            continue
        home_name = match["homeTeam"]["name"]
        away_name = match["awayTeam"]["name"]
        try:
            home_team_id = resolve(session, home_name, source="api", league_code=league.code)
            away_team_id = resolve(session, away_name, source="api", league_code=league.code)
        except UnresolvedTeamName:
            skipped += 1
            continue

        update_team_crest(session, home_team_id, match["homeTeam"].get("crest"))
        update_team_crest(session, away_team_id, match["awayTeam"].get("crest"))

        match_date = dt.datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")).date()
        upsert_match(
            session,
            league_code=league.code,
            season=season,
            date=match_date,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=score["home"],
            away_goals=score["away"],
            source="football-data.org",
            group_name=_football_data_group(match, league),
        )
        synced += 1
    return synced, skipped


def _resolve_espn_match(session: Session, league: League, match: espn_client.EspnMatch) -> tuple[int, int] | None:
    """Resolves both sides of an ESPN match to seeded Team rows, or None if
    either side isn't a known team in this league -- this is the only
    filter needed to keep out other competitions (Leagues Cup, US Open Cup,
    etc.) an MLS team's ESPN data may also include, since a genuine
    cross-competition opponent simply won't resolve. A same-competition cup
    match between two seeded MLS teams would slip through this filter, but
    none showed up in a real full-season sample checked while building this
    -- worth a second look if one turns up later.
    """
    home_team = team_by_espn_id(session, league.code, int(match.espn_home_id))
    away_team = team_by_espn_id(session, league.code, int(match.espn_away_id))
    if league.has_groups:
        # Every match on a group competition's own schedule belongs to it, so
        # an opponent missing from ESPN's *current* team list is a team that
        # has since left the competition (e.g. Russia, suspended from UEFA
        # events after 2022) rather than a foreign opponent -- add it, or
        # its old group's matches would be silently dropped and every
        # opponent's table would be short of games.
        if home_team is None:
            home_team = _create_espn_team(session, league, match.espn_home_id, match.home_name)
        if away_team is None:
            away_team = _create_espn_team(session, league, match.espn_away_id, match.away_name)
    if home_team is None or away_team is None:
        return None
    return home_team.id, away_team.id


def _create_espn_team(session: Session, league: League, espn_id: str, name: str) -> Team:
    team = get_or_create_team(session, name, league.code)
    update_espn_team_id(session, team.id, int(espn_id))
    session.flush()
    return team


def _group_for_match(match: espn_client.EspnMatch, membership: dict[str, str]) -> str | None:
    """The group a match belongs to: ESPN's own per-match label when the
    scoreboard supplied one, else the shared group of both teams for that
    edition (see espn_client.fetch_group_membership). Knockout and playoff
    matches pair teams from different groups, so they correctly get None --
    which is also what keeps them out of the group tables.
    """
    if match.group_name:
        return match.group_name
    home_group = membership.get(match.espn_home_id)
    if home_group is not None and home_group == membership.get(match.espn_away_id):
        return home_group
    return None


def _is_group_stage_name(stage: str | None) -> bool:
    # "Group Stage" (2020-2024 editions), "League Phase" (2026), or a bare
    # division name like "League A" (how ESPN labels the 2018-19 edition).
    return stage is not None and stage.casefold().startswith(("group", "league"))


def derive_groups_from_matches(
    matches: Iterable[espn_client.EspnMatch], membership: dict[str, str]
) -> dict[tuple[str, str, dt.date], str]:
    """Fallback for matches ESPN's standings can't place in a group (its
    2018-19 data lists only League D): a group is a round-robin, so the
    group-stage matches of one stage label split into connected components
    of "who played whom", and each component is one group. Named
    "{stage} - Group {n}" (n by first team alphabetically, so it's stable
    across runs). Matches already covered by `membership`, or not group
    stage at all, are left out.
    """
    by_stage: dict[str, list[espn_client.EspnMatch]] = {}
    for match in matches:
        if _group_for_match(match, membership) is not None or not _is_group_stage_name(match.stage):
            continue
        by_stage.setdefault(match.stage, []).append(match)

    derived: dict[tuple[str, str, dt.date], str] = {}
    for stage, stage_matches in by_stage.items():
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        names: dict[str, str] = {}
        for m in stage_matches:
            names[m.espn_home_id] = m.home_name
            names[m.espn_away_id] = m.away_name
            parent[find(m.espn_home_id)] = find(m.espn_away_id)

        components: dict[str, list[str]] = {}
        for team_id in names:
            components.setdefault(find(team_id), []).append(team_id)
        ordered_roots = sorted(components, key=lambda root: min(names[t] for t in components[root]))
        label_by_root = {root: f"{stage} - Group {i}" for i, root in enumerate(ordered_roots, start=1)}

        for m in stage_matches:
            derived[(m.espn_home_id, m.espn_away_id, m.date)] = label_by_root[find(m.espn_home_id)]
    return derived


def sync_fixtures_to_db_espn(session: Session, league: League) -> tuple[int, int]:
    """ESPN-backed equivalent of sync_fixtures_to_db, for leagues with
    data_source == "espn" (e.g. MLS). Scans the next
    ESPN_UPCOMING_FIXTURE_WINDOW_DAYS days one at a time -- ESPN's
    scoreboard endpoint only accepts a single date per call.
    """
    synced = 0
    skipped = 0
    today = dt.date.today()
    # Fixtures are always for the league's latest edition.
    membership = (
        espn_client.fetch_group_membership(league.espn_league_slug, league.api_season_year(league.seasons[-1]))
        if league.has_groups
        else {}
    )
    for offset in range(ESPN_UPCOMING_FIXTURE_WINDOW_DAYS):
        day = today + dt.timedelta(days=offset)
        for match in espn_client.fetch_day_fixtures(league.espn_league_slug, day):
            resolved = _resolve_espn_match(session, league, match)
            if resolved is None:
                skipped += 1
                continue
            home_team_id, away_team_id = resolved
            upsert_fixture(
                session,
                league_code=league.code,
                date=match.date,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                status="SCHEDULED",
                kickoff_utc=match.kickoff_utc,
                group_name=_group_for_match(match, membership) if league.has_groups else None,
            )
            synced += 1
    return synced, skipped


def sync_results_to_db_espn(session: Session, league: League, season: str) -> tuple[int, int]:
    """ESPN-backed equivalent of sync_results_to_db, for leagues with
    data_source == "espn" (e.g. MLS). Iterates every already-seeded team's
    schedule (see team_mapper.seed_teams_from_espn) rather than a
    league-wide endpoint, since ESPN has none for completed-results history.
    """
    synced = 0
    skipped = 0
    season_year = league.api_season_year(season)
    # A match shows up in both teams' schedules -- collect each once so the
    # group-derivation fallback below sees every match of a stage exactly once.
    resolved_matches: dict[tuple[str, str, dt.date], tuple[espn_client.EspnMatch, int, int]] = {}
    for team_id in teams_for_league(session, league.code):
        team = session.get(Team, team_id)
        if team is None or team.espn_team_id is None:
            continue
        for match in espn_client.fetch_team_results(league.espn_league_slug, str(team.espn_team_id), season_year):
            resolved = _resolve_espn_match(session, league, match)
            if resolved is None:
                skipped += 1
                continue
            resolved_matches[(match.espn_home_id, match.espn_away_id, match.date)] = (match, *resolved)
            # Counted per occurrence (a match in both teams' schedules counts
            # twice), same as before the collect-then-write split.
            synced += 1

    membership: dict[str, str] = {}
    derived: dict[tuple[str, str, dt.date], str] = {}
    if league.has_groups:
        membership = espn_client.fetch_group_membership(league.espn_league_slug, season_year)
        derived = derive_groups_from_matches((m for m, _, _ in resolved_matches.values()), membership)

    for key, (match, home_team_id, away_team_id) in resolved_matches.items():
        group_name = _group_for_match(match, membership) if league.has_groups else None
        group_is_fallback = False
        if league.has_groups and group_name is None:
            group_name = derived.get(key)
            group_is_fallback = group_name is not None
        upsert_match(
            session,
            league_code=league.code,
            season=season,
            date=match.date,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=match.home_score,
            away_goals=match.away_score,
            source="espn",
            group_name=group_name,
            group_is_fallback=group_is_fallback,
        )
    return synced, skipped
