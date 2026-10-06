"""Query helpers shared by ingestion, the model layer, and the dashboard."""

from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from soccer_predictor.storage.models import (
    Fixture,
    HistoricalTournamentGoal,
    Injury,
    Match,
    PredictionRecord,
    Team,
    TeamAlias,
    TeamFormNote,
)


def team_by_id(session: Session, team_id: int) -> Team | None:
    return session.get(Team, team_id)


def update_team_crest(session: Session, team_id: int, crest_url: str | None) -> None:
    """Sets Team.crest_url. Never clobbers a known crest with a falsy one."""
    if not crest_url:
        return
    team = session.get(Team, team_id)
    if team is not None:
        team.crest_url = crest_url


def update_api_football_team_id(session: Session, team_id: int, api_football_team_id: int | None) -> None:
    """Sets Team.api_football_team_id. Never clobbers a known id with a
    falsy one -- same "learn once, keep forever" pattern as update_team_crest.
    """
    if not api_football_team_id:
        return
    team = session.get(Team, team_id)
    if team is not None:
        team.api_football_team_id = api_football_team_id


def update_espn_team_id(session: Session, team_id: int, espn_team_id: int | None) -> None:
    """Sets Team.espn_team_id. Never clobbers a known id with a falsy one --
    same "learn once, keep forever" pattern as update_api_football_team_id.
    """
    if not espn_team_id:
        return
    team = session.get(Team, team_id)
    if team is not None:
        team.espn_team_id = espn_team_id


def team_by_espn_id(session: Session, league_code: str, espn_team_id: int) -> Team | None:
    return session.scalar(
        select(Team).where(Team.league_code == league_code, Team.espn_team_id == espn_team_id)
    )


def update_team_conference(session: Session, team_id: int, conference: str | None) -> None:
    """Sets Team.conference. Never clobbers a known value with a falsy one --
    same "learn once, keep forever" pattern as update_team_crest.
    """
    if not conference:
        return
    team = session.get(Team, team_id)
    if team is not None:
        team.conference = conference


def team_conferences_for_league(session: Session, league_code: str) -> dict[int, str]:
    """Only includes teams with a known conference -- callers should
    .get(id) and treat a missing entry as "no conference split for this
    league" rather than an error.
    """
    rows = session.execute(
        select(Team.id, Team.conference).where(
            Team.league_code == league_code, Team.conference.is_not(None)
        )
    ).all()
    return {team_id: conference for team_id, conference in rows}


def get_or_create_team(session: Session, canonical_name: str, league_code: str) -> Team:
    """Teams are scoped per league now -- the same club name can have a
    separate row per competition (see storage.models.Team's docstring), so
    the lookup must match on league_code too, not canonical_name alone.
    """
    team = session.scalar(
        select(Team).where(Team.canonical_name == canonical_name, Team.league_code == league_code)
    )
    if team is None:
        team = Team(canonical_name=canonical_name, league_code=league_code)
        session.add(team)
        session.flush()
    return team


def upsert_team_alias(session: Session, team: Team, alias_text: str, source: str) -> None:
    existing = session.scalar(
        select(TeamAlias).where(
            TeamAlias.alias_text == alias_text,
            TeamAlias.source == source,
            TeamAlias.league_code == team.league_code,
        )
    )
    if existing is None:
        session.add(
            TeamAlias(
                alias_text=alias_text, source=source, team_id=team.id, league_code=team.league_code
            )
        )


def find_team_by_alias(session: Session, alias_text: str, source: str, league_code: str) -> Team | None:
    alias = session.scalar(
        select(TeamAlias).where(
            TeamAlias.alias_text == alias_text,
            TeamAlias.source == source,
            TeamAlias.league_code == league_code,
        )
    )
    return alias.team if alias else None


def all_alias_texts(session: Session, source: str, league_code: str) -> list[tuple[str, int]]:
    """Returns (alias_text, team_id) pairs for a source, scoped to one league.

    Scoping matters for fuzzy matching: with several leagues sharing one
    aliases table, an unscoped candidate list could fuzzy-match a
    similarly-spelled team from the wrong league instead of correctly
    falling through to unmatched.
    """
    rows = session.execute(
        select(TeamAlias.alias_text, TeamAlias.team_id).where(
            TeamAlias.source == source, TeamAlias.league_code == league_code
        )
    ).all()
    return [(text, team_id) for text, team_id in rows]


def upsert_match(
    session: Session,
    league_code: str,
    season: str,
    date: dt.date,
    home_team_id: int,
    away_team_id: int,
    home_goals: int,
    away_goals: int,
    source: str = "football-data.co.uk",
    group_name: str | None = None,
    group_is_fallback: bool = False,
) -> None:
    """`group_name` is only ever set, never cleared: a re-sync from a source
    that doesn't know the group (e.g. ESPN's team-schedule endpoint) must
    not wipe a group an earlier pass already filled in. A `group_is_fallback`
    name (one derived from the match graph, see
    ingest/fixtures.py::derive_groups_from_matches) is a guess, so it only
    fills a match that has no group yet -- it never replaces a real label.
    """
    existing = session.scalar(
        select(Match).where(
            Match.league_code == league_code,
            Match.date == date,
            Match.home_team_id == home_team_id,
            Match.away_team_id == away_team_id,
        )
    )
    if existing is not None:
        existing.home_goals = home_goals
        existing.away_goals = away_goals
        existing.season = season
        if group_name is not None and not (group_is_fallback and existing.group_name is not None):
            existing.group_name = group_name
        return
    session.add(
        Match(
            league_code=league_code,
            season=season,
            date=date,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=home_goals,
            away_goals=away_goals,
            source=source,
            group_name=group_name,
        )
    )


def set_match_group_names(
    session: Session, league_code: str, season: str, group_by_match: dict[tuple[int, int], str]
) -> int:
    """Fills Match.group_name for already-stored matches of one season, keyed
    by (home_team_id, away_team_id) -- used when the group is only learnable
    after the results are in (ESPN's team-schedule endpoint carries no
    group). Returns how many rows were updated.
    """
    updated = 0
    for match in session.scalars(
        select(Match).where(Match.league_code == league_code, Match.season == season)
    ).all():
        group = group_by_match.get((match.home_team_id, match.away_team_id))
        if group is not None and match.group_name != group:
            match.group_name = group
            updated += 1
    return updated


def replace_historical_tournament_goals(
    session: Session, league_code: str, season: str, goals: list[dict], source: str
) -> None:
    """Replaces all HistoricalTournamentGoal rows for (league_code, season)
    with `goals` -- same "delete existing, insert fresh" idiom as
    replace_injuries, appropriate here since a re-run of the archive
    ingest scripts (see scripts/fetch_historical_data.py) means the parser
    changed, not that the underlying (unchanging) tournament result did.

    Each dict in `goals` needs team_id (already resolved by the caller --
    see ingest/historical_csv.py::ingest_goals_into_db), player_name,
    minute, match_date, own_goal, penalty.
    """
    existing = session.scalars(
        select(HistoricalTournamentGoal).where(
            HistoricalTournamentGoal.league_code == league_code,
            HistoricalTournamentGoal.season == season,
        )
    ).all()
    for row in existing:
        session.delete(row)
    session.flush()
    for entry in goals:
        session.add(
            HistoricalTournamentGoal(
                league_code=league_code,
                season=season,
                team_id=entry["team_id"],
                player_name=entry["player_name"],
                minute=entry.get("minute"),
                match_date=entry.get("match_date"),
                own_goal=bool(entry.get("own_goal", False)),
                penalty=bool(entry.get("penalty", False)),
                source=source,
            )
        )


def goals_for_team_season(
    session: Session, team_id: int, league_code: str, season: str
) -> list[HistoricalTournamentGoal]:
    return list(
        session.scalars(
            select(HistoricalTournamentGoal).where(
                HistoricalTournamentGoal.team_id == team_id,
                HistoricalTournamentGoal.league_code == league_code,
                HistoricalTournamentGoal.season == season,
            )
        ).all()
    )


def has_prediction_record(
    session: Session, league_code: str, date: dt.date, home_team_id: int, away_team_id: int
) -> bool:
    return (
        session.scalar(
            select(PredictionRecord.id).where(
                PredictionRecord.league_code == league_code,
                PredictionRecord.date == date,
                PredictionRecord.home_team_id == home_team_id,
                PredictionRecord.away_team_id == away_team_id,
            )
        )
        is not None
    )


def insert_prediction_record(
    session: Session,
    league_code: str,
    date: dt.date,
    home_team_id: int,
    away_team_id: int,
    predicted_home_goals: int,
    predicted_away_goals: int,
    p_home: float | None = None,
    p_draw: float | None = None,
    p_away: float | None = None,
) -> None:
    """Locks in a prediction -- unlike upsert_match/upsert_fixture, an
    existing row means skip, never update. That's the whole point of a
    snapshot: it has to stay whatever was forecast at the time, even if the
    model is retrained before the match is actually played.
    """
    if has_prediction_record(session, league_code, date, home_team_id, away_team_id):
        return
    session.add(
        PredictionRecord(
            league_code=league_code,
            date=date,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            predicted_home_goals=predicted_home_goals,
            predicted_away_goals=predicted_away_goals,
            snapshotted_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
        )
    )


def prediction_records_missing_probabilities(
    session: Session, league_code: str, from_date: dt.date
) -> list[PredictionRecord]:
    """Locked-in records dated `from_date` or later that have no saved
    probabilities (i.e. made before those were stored)."""
    return list(
        session.scalars(
            select(PredictionRecord).where(
                PredictionRecord.league_code == league_code,
                PredictionRecord.date >= from_date,
                PredictionRecord.p_home.is_(None),
            )
        ).all()
    )


def all_prediction_records(session: Session) -> list[PredictionRecord]:
    return list(session.scalars(select(PredictionRecord)).all())


def matches_for_league(session: Session, league_code: str) -> pd.DataFrame:
    rows = session.execute(
        select(
            Match.date,
            Match.season,
            Match.home_team_id,
            Match.away_team_id,
            Match.home_goals,
            Match.away_goals,
            Match.group_name,
        ).where(Match.league_code == league_code)
    ).all()
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "season",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
            "group_name",
        ],
    )


def matches_for_team(session: Session, team_id: int) -> pd.DataFrame:
    """Newest-first: this feeds a "recent results" display, unlike
    matches_for_league (which feeds the trainer and doesn't care about order).
    """
    rows = session.execute(
        select(
            Match.date,
            Match.home_team_id,
            Match.away_team_id,
            Match.home_goals,
            Match.away_goals,
        )
        .where(or_(Match.home_team_id == team_id, Match.away_team_id == team_id))
        .order_by(Match.date.desc())
    ).all()
    return pd.DataFrame(
        rows,
        columns=["date", "home_team_id", "away_team_id", "home_goals", "away_goals"],
    )


def teams_for_league(session: Session, league_code: str) -> dict[int, str]:
    rows = session.execute(
        select(Team.id, Team.canonical_name).where(Team.league_code == league_code)
    ).all()
    return {team_id: name for team_id, name in rows}


def team_crests_for_league(session: Session, league_code: str) -> dict[int, str]:
    """Only includes teams with a known crest -- callers should .get(id, "")."""
    rows = session.execute(
        select(Team.id, Team.crest_url).where(
            Team.league_code == league_code, Team.crest_url.is_not(None)
        )
    ).all()
    return {team_id: crest_url for team_id, crest_url in rows}


def upsert_fixture(
    session: Session,
    league_code: str,
    date: dt.date,
    home_team_id: int,
    away_team_id: int,
    status: str,
    kickoff_utc: dt.datetime | None = None,
    group_name: str | None = None,
) -> None:
    existing = session.scalar(
        select(Fixture).where(
            Fixture.league_code == league_code,
            Fixture.date == date,
            Fixture.home_team_id == home_team_id,
            Fixture.away_team_id == away_team_id,
        )
    )
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    if existing is not None:
        existing.status = status
        existing.fetched_at = now
        if kickoff_utc is not None:
            existing.kickoff_utc = kickoff_utc
        if group_name is not None:
            existing.group_name = group_name
        return
    session.add(
        Fixture(
            league_code=league_code,
            date=date,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            status=status,
            fetched_at=now,
            kickoff_utc=kickoff_utc,
            group_name=group_name,
        )
    )


def team_group_names_for_season(session: Session, league_code: str, season: str) -> dict[int, str]:
    """{team_id: group name} for one edition of a group competition, from its
    group-tagged matches plus its fixtures (fixtures carry no season, so they
    count as the latest edition -- pass the league's latest season for those
    to be meaningful). Used to read each Nations League team's division."""
    teams: dict[int, str] = {}
    match_rows = session.execute(
        select(Match.home_team_id, Match.away_team_id, Match.group_name).where(
            Match.league_code == league_code, Match.season == season, Match.group_name.is_not(None)
        )
    ).all()
    fixture_rows = session.execute(
        select(Fixture.home_team_id, Fixture.away_team_id, Fixture.group_name).where(
            Fixture.league_code == league_code, Fixture.group_name.is_not(None)
        )
    ).all()
    for home_id, away_id, group in [*match_rows, *fixture_rows]:
        teams.setdefault(home_id, group)
        teams.setdefault(away_id, group)
    return teams


def fixture_groups_for_league(session: Session, league_code: str) -> pd.DataFrame:
    """(home_team_id, away_team_id, group_name) for every stored fixture with
    a group -- lets a group that hasn't played yet still list its teams
    (see model/standings.py::compute_group_standings's `fixture_groups`).
    """
    rows = session.execute(
        select(Fixture.home_team_id, Fixture.away_team_id, Fixture.group_name).where(
            Fixture.league_code == league_code, Fixture.group_name.is_not(None)
        )
    ).all()
    return pd.DataFrame(rows, columns=["home_team_id", "away_team_id", "group_name"])


def fixtures_for_league(
    session: Session, league_code: str, start: dt.date, end: dt.date
) -> pd.DataFrame:
    rows = session.execute(
        select(
            Fixture.date,
            Fixture.home_team_id,
            Fixture.away_team_id,
            Fixture.status,
            Fixture.kickoff_utc,
        ).where(
            Fixture.league_code == league_code,
            Fixture.date >= start,
            Fixture.date <= end,
        )
    ).all()
    return pd.DataFrame(
        rows, columns=["date", "home_team_id", "away_team_id", "status", "kickoff_utc"]
    )


def live_fixtures_across_leagues(
    session: Session, now: dt.datetime, window: dt.timedelta
) -> pd.DataFrame:
    """Every fixture (any league) whose kickoff falls within `window` before
    `now` -- i.e. plausibly still in progress (see
    dashboard.components.LIVE_MATCH_WINDOW). Used for the global "Live Now"
    strip on the Leagues page, which shows live games across every league
    regardless of which one is currently selected.
    """
    rows = session.execute(
        select(
            Fixture.league_code,
            Fixture.home_team_id,
            Fixture.away_team_id,
            Fixture.kickoff_utc,
        ).where(
            Fixture.kickoff_utc.is_not(None),
            Fixture.kickoff_utc <= now,
            Fixture.kickoff_utc >= now - window,
        )
    ).all()
    return pd.DataFrame(
        rows, columns=["league_code", "home_team_id", "away_team_id", "kickoff_utc"]
    )


def fixtures_for_team(
    session: Session, team_id: int, start: dt.date, end: dt.date
) -> pd.DataFrame:
    rows = session.execute(
        select(
            Fixture.date,
            Fixture.home_team_id,
            Fixture.away_team_id,
            Fixture.status,
            Fixture.kickoff_utc,
        ).where(
            or_(Fixture.home_team_id == team_id, Fixture.away_team_id == team_id),
            Fixture.date >= start,
            Fixture.date <= end,
        )
    ).all()
    return pd.DataFrame(
        rows, columns=["date", "home_team_id", "away_team_id", "status", "kickoff_utc"]
    )


def next_fixture_per_team(
    session: Session, league_code: str, as_of: dt.date
) -> dict[int, int]:
    """Maps each team_id to its soonest opponent's team_id on/after `as_of`.

    Teams with no upcoming fixture loaded are simply absent from the result.
    """
    rows = session.execute(
        select(Fixture.date, Fixture.home_team_id, Fixture.away_team_id)
        .where(Fixture.league_code == league_code, Fixture.date >= as_of)
        .order_by(Fixture.date.asc())
    ).all()

    next_opponent: dict[int, int] = {}
    for _, home_team_id, away_team_id in rows:
        if home_team_id not in next_opponent:
            next_opponent[home_team_id] = away_team_id
        if away_team_id not in next_opponent:
            next_opponent[away_team_id] = home_team_id
    return next_opponent


def replace_injuries(
    session: Session, team_id: int, source: str, injuries: list[dict]
) -> None:
    """Replaces all injuries for (team_id, source) with the given list."""
    existing = session.scalars(
        select(Injury).where(Injury.team_id == team_id, Injury.source == source)
    ).all()
    for row in existing:
        session.delete(row)
    session.flush()
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    for entry in injuries:
        session.add(
            Injury(
                team_id=team_id,
                player_name=entry["player_name"],
                position=entry["position"],
                importance_weight=entry["importance_weight"],
                source=source,
                note=entry.get("note", ""),
                fetched_at=now,
            )
        )


def active_chat_injuries(session: Session, as_of: dt.date) -> list[tuple[Injury, Team]]:
    """Every chat-sourced injury still counting as of `as_of` (no return date,
    or one that hasn't passed), across ALL leagues, each with its team."""
    return [
        (injury, team)
        for injury, team in session.execute(
            select(Injury, Team)
            .join(Team, Team.id == Injury.team_id)
            .where(
                Injury.source == "chat",
                or_(Injury.expected_return_date.is_(None), Injury.expected_return_date >= as_of),
            )
        ).all()
    ]


def active_form_notes(session: Session, as_of: dt.date) -> list[tuple[TeamFormNote, Team]]:
    """Every form note not yet expired as of `as_of`, across ALL leagues,
    each with its team."""
    return [
        (note, team)
        for note, team in session.execute(
            select(TeamFormNote, Team)
            .join(Team, Team.id == TeamFormNote.team_id)
            .where(TeamFormNote.expires_on >= as_of)
        ).all()
    ]


def injuries_for_team(session: Session, team_id: int) -> list[Injury]:
    return list(session.scalars(select(Injury).where(Injury.team_id == team_id)).all())


def add_or_update_chat_injury(
    session: Session,
    team_id: int,
    player_name: str,
    position: str,
    importance_weight: float,
    expected_return_date: dt.date | None,
    note: str = "",
) -> Injury:
    """Upserts a source="chat" injury, keyed on (team_id, player_name).

    Plain insert would duplicate every time the user corrects an estimate
    for the same player ("actually he's back sooner") -- this keeps one row.
    """
    existing = session.scalar(
        select(Injury).where(
            Injury.team_id == team_id,
            Injury.source == "chat",
            Injury.player_name.ilike(player_name),
        )
    )
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    if existing is not None:
        existing.position = position
        existing.importance_weight = importance_weight
        existing.expected_return_date = expected_return_date
        existing.note = note
        existing.fetched_at = now
        return existing

    injury = Injury(
        team_id=team_id,
        player_name=player_name,
        position=position,
        importance_weight=importance_weight,
        source="chat",
        note=note,
        fetched_at=now,
        expected_return_date=expected_return_date,
    )
    session.add(injury)
    session.flush()
    return injury


def delete_injury(session: Session, injury_id: int) -> None:
    injury = session.get(Injury, injury_id)
    if injury is not None:
        session.delete(injury)


def add_form_note(
    session: Session,
    team_id: int,
    raw_text: str,
    summary: str,
    magnitude: float,
    affects: str,
    expires_on: dt.date,
) -> TeamFormNote:
    note = TeamFormNote(
        team_id=team_id,
        raw_text=raw_text,
        summary=summary,
        magnitude=magnitude,
        affects=affects,
        expires_on=expires_on,
        fetched_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
    )
    session.add(note)
    session.flush()
    return note


def delete_form_note(session: Session, note_id: int) -> None:
    note = session.get(TeamFormNote, note_id)
    if note is not None:
        session.delete(note)


def form_notes_for_team(session: Session, team_id: int) -> list[TeamFormNote]:
    return list(
        session.scalars(select(TeamFormNote).where(TeamFormNote.team_id == team_id)).all()
    )
