"""SQLAlchemy ORM models — the canonical schema every ingest module writes into."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    """A team belongs to exactly one `league_code` scope. A club that plays in
    both a domestic league and UEFA Champions League gets one Team row per
    competition (e.g. "Manchester City" under EPL, a separate row under UCL)
    -- canonical_name is only unique *within* a league, not globally, so
    they don't collide (see migrate_multi_league_teams.py)."""

    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("canonical_name", "league_code"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String, index=True)
    league_code: Mapped[str] = mapped_column(String, index=True)
    # Only ever populated once a live fixture sync resolves this team and
    # the football-data.org response includes a crest URL -- historical-only
    # teams (no live sync yet) stay None; the dashboard falls back to a
    # placeholder icon rather than erroring.
    crest_url: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    # API-Football's own numeric team id, once resolved via a name search
    # (ingest/injuries.py) -- cached here permanently so every later injury
    # sync skips straight to the injuries call instead of re-searching by
    # name every single time, which was the single biggest cost in
    # refresh_live_data.py's runtime (2 API-Football calls/team -> 1).
    api_football_team_id: Mapped[int | None] = mapped_column(nullable=True, default=None)
    # ESPN's own numeric team id (site.api.espn.com), for leagues with
    # League.data_source == "espn" (e.g. MLS) -- same permanent-cache idiom
    # as api_football_team_id above. ESPN keys everything by this stable id,
    # so leagues using it never need name-based alias resolution at all.
    espn_team_id: Mapped[int | None] = mapped_column(nullable=True, default=None)
    # e.g. "Eastern Conference"/"Western Conference" for MLS -- only ever
    # populated for leagues with a real conference split (see
    # ingest/espn_client.py::fetch_team_conference); None everywhere else.
    conference: Mapped[str | None] = mapped_column(String, nullable=True, default=None)

    aliases: Mapped[list["TeamAlias"]] = relationship(back_populates="team")


class TeamAlias(Base):
    # Same alias text/source pair can now point at different teams in
    # different leagues (e.g. API name "Manchester City FC" resolves to a
    # different team_id under EPL vs under UCL) -- scoped by league_code too.
    __tablename__ = "team_aliases"
    __table_args__ = (UniqueConstraint("alias_text", "source", "league_code"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    alias_text: Mapped[str] = mapped_column(String, index=True)
    source: Mapped[str] = mapped_column(String)  # "csv" | "api"
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    league_code: Mapped[str] = mapped_column(String, index=True)

    team: Mapped[Team] = relationship(back_populates="aliases")


class Match(Base):
    """A completed historical match (goals known), from the CSV source."""

    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("league_code", "date", "home_team_id", "away_team_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    league_code: Mapped[str] = mapped_column(String, index=True)
    season: Mapped[str] = mapped_column(String)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    home_goals: Mapped[int]
    away_goals: Mapped[int]
    source: Mapped[str] = mapped_column(String, default="football-data.co.uk")
    # e.g. "Group A1" for a group-stage match of a group competition (Nations
    # League, World Cup, Euros) -- None for everything else (domestic
    # leagues, knockout/playoff matches), which is also how knockout results
    # are kept out of the per-group tables (see model/standings.py).
    group_name: Mapped[str | None] = mapped_column(String, nullable=True, default=None)


class Fixture(Base):
    """An upcoming (not yet played) match, from the live API."""

    __tablename__ = "fixtures"
    __table_args__ = (
        UniqueConstraint("league_code", "date", "home_team_id", "away_team_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    league_code: Mapped[str] = mapped_column(String, index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    status: Mapped[str] = mapped_column(String, default="SCHEDULED")
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime)
    # Naive but semantically UTC (matches every other stored datetime in
    # this app). Nullable: rows synced before this column existed, or ever,
    # simply have no kickoff time -- callers fall back to date-only display.
    kickoff_utc: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    # Same meaning as Match.group_name -- lets a group with no results yet
    # still list its teams.
    group_name: Mapped[str | None] = mapped_column(String, nullable=True, default=None)


class Injury(Base):
    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    player_name: Mapped[str] = mapped_column(String)
    position: Mapped[str] = mapped_column(String)  # "attack" | "defense"
    importance_weight: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String)  # "manual" | "api" | "chat"
    note: Mapped[str] = mapped_column(String, default="")
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime)
    # Only ever populated for source="chat" -- manual/api rows stay
    # unconditionally active (None), preserving prior behavior exactly.
    expected_return_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True, default=None)


class TeamFormNote(Base):
    """A free-text "team's been flat lately" style note from the chat UI.

    Unlike Injury, this has no real mechanism behind it -- see
    model/form_adjustment.py's docstring for why it's kept small and capped.
    """

    __tablename__ = "team_form_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    raw_text: Mapped[str] = mapped_column(String)
    summary: Mapped[str] = mapped_column(String, default="")
    magnitude: Mapped[float] = mapped_column(Float)  # signed, -1..1; negative = bad form
    affects: Mapped[str] = mapped_column(String)  # "attack" | "defense" | "both"
    source: Mapped[str] = mapped_column(String, default="chat")
    expires_on: Mapped[dt.date] = mapped_column(Date)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime)


class FittedParams(Base):
    """Persisted Dixon-Coles fit output for one league, one point in time."""

    __tablename__ = "fitted_params"

    id: Mapped[int] = mapped_column(primary_key=True)
    league_code: Mapped[str] = mapped_column(String, index=True)
    fitted_at: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    params_json: Mapped[str] = mapped_column(String)
    time_window_desc: Mapped[str] = mapped_column(String, default="")


class HistoricalTournamentGoal(Base):
    """One goal from a static tournament archive (World Cup via
    ingest/worldcup_archive.py, Euro via ingest/euro_archive.py) -- goals
    only, not full appearances/lineups, since goals are what the dashboard's
    star-badge/top-scorer display already surfaces first (see
    dashboard/components.py::compute_top_scorer_and_assister) and neither
    archive source has reliable assist data anyway.

    Unlike Match/Fixture, re-ingesting a (league_code, season) fully
    replaces its rows (see storage/repository.py::replace_historical_tournament_goals)
    rather than upserting row-by-row -- these archives never change once
    downloaded, so a re-run only ever means "the parser logic changed."
    """

    __tablename__ = "historical_tournament_goals"

    id: Mapped[int] = mapped_column(primary_key=True)
    league_code: Mapped[str] = mapped_column(String, index=True)
    season: Mapped[str] = mapped_column(String, index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    player_name: Mapped[str] = mapped_column(String)
    # Regulation minute only (stoppage-time added-on ignored) -- None when
    # the source didn't record one for this goal.
    minute: Mapped[int | None] = mapped_column(nullable=True, default=None)
    match_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True, default=None)
    own_goal: Mapped[bool] = mapped_column(default=False)
    penalty: Mapped[bool] = mapped_column(default=False)
    source: Mapped[str] = mapped_column(String)  # "worldcup_archive" | "euro_archive"


class PredictionRecord(Base):
    """A prediction locked in the first time a fixture was seen as upcoming
    with a trained model available -- never overwritten, even if the model
    is later retrained, so this stays a genuine advance forecast to grade
    against once the real result is known (see prediction/tracking.py).
    """

    __tablename__ = "prediction_records"
    __table_args__ = (
        UniqueConstraint("league_code", "date", "home_team_id", "away_team_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    league_code: Mapped[str] = mapped_column(String, index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    predicted_home_goals: Mapped[int]
    predicted_away_goals: Mapped[int]
    snapshotted_at: Mapped[dt.datetime] = mapped_column(DateTime)
    # The model's win/draw/loss probabilities for this match, saved with the
    # scoreline so the Track Record can grade the most likely OUTCOME (not
    # just the single most likely scoreline, which is usually a 0-0/1-1 draw
    # even when a win is likelier) and score the probabilities themselves.
    # None for records locked in before these columns existed -- those are
    # graded from the scoreline alone.
    p_home: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    p_draw: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    p_away: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
