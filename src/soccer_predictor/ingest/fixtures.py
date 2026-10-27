"""Fetches upcoming fixtures from football-data.org and syncs them to SQLite."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable

from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import api_client, espn_client
from soccer_predictor.ingest.team_mapper import UnresolvedTeamName, resolve
from soccer_predictor.storage.models import Team
from soccer_predictor.storage.repository import (
    get_or_create_team,
    match_count_for_season,
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


def ninety_minute_score(match: dict) -> tuple[int, int] | None:
    """The score after 90 minutes of a football-data.org match, or None if it
    has none yet.

    `score.fullTime` also counts extra-time goals and, for a tie settled on
    penalties, the shoot-out as well (an Arsenal 1-0 Porto that went to
    penalties comes back as 5-2), while `score.regularTime` -- present only for
    a match that ran past 90 minutes -- is the score the ratings must be fit on
    and the predicted scoreline is graded against.
    """
    score = match.get("score") or {}
    for key in ("regularTime", "fullTime"):
        side = score.get(key) or {}
        if side.get("home") is not None and side.get("away") is not None:
            return side["home"], side["away"]
    return None


def sync_results_to_db(session: Session, league: League, season: str) -> tuple[int, int]:
    """Resolves team names and upserts finished matches for one season.
    Returns (synced, skipped). Mirrors sync_fixtures_to_db, but writes to
    the `matches` table (standings/training history) instead of `fixtures`
    (the upcoming-games list).
    """
    synced = 0
    skipped = 0
    for match in fetch_finished_matches(league, league.api_season_year(season)):
        score = ninety_minute_score(match)
        if score is None:
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
            home_goals=score[0],
            away_goals=score[1],
            source="football-data.org",
            group_name=_football_data_group(match, league),
        )
        synced += 1
    return synced, skipped


# ESPN lists an undecided slot as if it were a team: "TBD" for a qualifier still to
# be played, and for knockout fixtures published before the groups finish "Group A
# Winner", "Group C 2nd Place", "3rd Place Group A/C/D", "Round of 16 1 Winner",
# "Semifinal 2 Loser" (verified on the 2027 Asian Cup scoreboard: 30 of the 54
# "teams" it names are such slots, all flagged isActive=false). None of them is a
# team: they must never be stored, rated or predicted, or the Track Record would
# lock in forecasts for matches between nobody.
_PLACEHOLDER_TEAM_RE = re.compile(
    r"\b(winner|loser|runners?-up)\b"
    r"|\b\d(st|nd|rd|th) place\b"
    r"|\bplace group\b"
    r"|^(tbd|tba|to be (determined|announced))$",
    re.IGNORECASE,
)


def _is_placeholder_team(name: str | None) -> bool:
    return bool(_PLACEHOLDER_TEAM_RE.search((name or "").strip()))


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
    if _is_placeholder_team(match.home_name) or _is_placeholder_team(match.away_name):
        return None
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


def _create_espn_team(
    session: Session, league: League, espn_id: str, name: str, logo_url: str | None = None
) -> Team:
    team = get_or_create_team(session, name, league.code)
    update_espn_team_id(session, team.id, int(espn_id))
    if logo_url:
        update_team_crest(session, team.id, logo_url)
    session.flush()
    return team


def seed_edition_participants(session: Session, league: League, participants) -> int:
    """Makes sure every team that played an edition exists as a Team of this
    league. seed_teams_from_espn only knows the CURRENT field, so a team that
    has since dropped out (or only qualified in earlier years) would be
    invisible to the schedule walk below -- and a match between two such teams
    would never be found at all. Returns how many teams were added."""
    added = 0
    for participant in participants:
        if _is_placeholder_team(participant.name):
            continue
        if team_by_espn_id(session, league.code, int(participant.espn_id)) is None:
            _create_espn_team(session, league, participant.espn_id, participant.name, participant.logo_url)
            added += 1
    return added


def _group_for_match(
    match: espn_client.EspnMatch, membership: dict[str, str], require_label: bool = False
) -> str | None:
    """The group a match belongs to: ESPN's own per-match label when the
    scoreboard supplied one, else the shared group of both teams for that
    edition (see espn_client.fetch_group_membership).

    Knockout and playoff matches get None, which is also what keeps them out
    of the group tables. Usually they pair teams from different groups so the
    shared-group rule already excludes them, but not always: two teams from
    one group can meet again in a knockout round (the 2015 Asian Cup final was
    Australia v South Korea, both Group A). So when the match's stage is known
    and isn't a group stage, it is never a group match.

    `require_label=True` skips the shared-group rule entirely: the scoreboard
    (used by the frequent quick refresh) names no stage, so a rematch there
    can't be told from a group game -- only ESPN's own group label counts.
    """
    if match.group_name:
        return match.group_name
    if require_label:
        return None
    if match.stage and not _is_group_stage_name(match.stage):
        return None
    home_group = membership.get(match.espn_home_id)
    if home_group is not None and home_group == membership.get(match.espn_away_id):
        return home_group
    return None


def _is_group_stage_name(stage: str | None) -> bool:
    # "Group Stage" (2020-2024 editions), "2015 Group Stage" (older Asian Cups,
    # where the year leads), "League Phase" (2026), or a bare division name like
    # "League A" (how ESPN labels the 2018-19 edition).
    if stage is None:
        return False
    name = stage.casefold()
    return "group" in name or name.startswith("league")


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


def sync_fixtures_to_db_espn(
    session: Session, league: League, window_days: int = ESPN_UPCOMING_FIXTURE_WINDOW_DAYS
) -> tuple[int, int]:
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
    def matches_to_sync():
        for offset in range(window_days):
            yield from espn_client.fetch_day_fixtures(league.espn_league_slug, today + dt.timedelta(days=offset))
        if league.has_groups:
            # A group competition's whole schedule is published long before it
            # starts (the Asian Cup's draw is months ahead): take this year's and
            # next year's in one request each, so its groups and fixtures show
            # up now rather than two weeks before kickoff.
            for year in (today.year, today.year + 1):
                yield from espn_client.fetch_year_fixtures(league.espn_league_slug, year)

    for match in matches_to_sync():
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
            espn_event_id=match.event_id,
        )
        synced += 1
    return synced, skipped


def espn_seasons_to_sync(session: Session, league: League, all_seasons: bool = False) -> list[str]:
    """Which of a league's seasons a full historical sync needs to fetch.

    Syncing one ESPN season costs a request per team (55 for the Nations
    League), throttled to ~2s each, and a concluded edition's results never
    change -- so, like the CSV leagues' past seasons, a past season that
    already has matches stored is skipped rather than re-walked on every run.
    The latest season is always synced (it's the one still changing), and so
    is any past season with nothing stored yet (first run, or an earlier
    attempt that failed). `all_seasons` forces the full re-walk.
    """
    if all_seasons:
        return list(league.seasons)
    *past, latest = league.seasons
    return [s for s in past if match_count_for_season(session, league.code, s) == 0] + [latest]


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
    participants = espn_client.fetch_group_participants(league.espn_league_slug, season_year) if league.has_groups else []
    seed_edition_participants(session, league, participants)
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
        membership = {p.espn_id: p.group for p in participants}
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
            espn_event_id=match.event_id,
        )
    return synced, skipped


# Cached just long enough that two runs in a row don't repeat a request, short
# enough that a result that has just landed is picked up.
RECENT_RESULTS_CACHE_TTL_SECONDS = 300


def sync_recent_results_to_db_espn(
    session: Session, league: League, season: str, days_back: int = 2
) -> tuple[int, int]:
    """Cheap alternative to sync_results_to_db_espn for frequent refreshes:
    reads the league-wide scoreboard for the last `days_back` days (plus
    today) -- a handful of requests -- instead of every team's whole-season
    schedule (one request per team: 55 for the Nations League). Only
    completed matches with a score are stored, upserted on the same key as
    the full sync so the two never create duplicates. Returns (synced, skipped).
    """
    synced = 0
    skipped = 0
    season_year = league.api_season_year(season)
    membership = (
        espn_client.fetch_group_membership(league.espn_league_slug, season_year) if league.has_groups else {}
    )
    today = dt.datetime.now(dt.UTC).date()
    for offset in range(days_back, -1, -1):
        day = today - dt.timedelta(days=offset)
        for match in espn_client.fetch_day_matches(
            league.espn_league_slug, day, cache_ttl_seconds=RECENT_RESULTS_CACHE_TTL_SECONDS
        ):
            if not match.completed or match.home_score is None or match.away_score is None:
                continue
            resolved = _resolve_espn_match(session, league, match)
            if resolved is None:
                skipped += 1
                continue
            home_team_id, away_team_id = resolved
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
                # The scoreboard names no stage, so a knockout rematch of two
                # group-mates can't be told from a group game: trust only ESPN's
                # own group label here.
                group_name=_group_for_match(match, membership, require_label=True) if league.has_groups else None,
                espn_event_id=match.event_id,
            )
            synced += 1
    return synced, skipped
