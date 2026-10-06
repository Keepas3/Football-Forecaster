"""Engine/session setup -- the local soccer.db SQLite file by default, or a
Turso (libSQL) database instead when TURSO_DATABASE_URL/TURSO_AUTH_TOKEN are
set (see config.py). Turso exists for deployments with no persistent local
disk (e.g. Streamlit Community Cloud); local development is unaffected
unless those env vars are explicitly set.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from soccer_predictor.config import DATA_DIR, DB_PATH, turso_auth_token, turso_database_url
from soccer_predictor.storage.models import Base

_engine = None
_engine_file_signature: tuple | None = None
_SessionLocal: sessionmaker | None = None


def _db_file_signature() -> tuple | None:
    """Identity of the SQLite file as it is on disk right now. A `git pull`
    (or the refresh workflows' commits arriving) replaces data/soccer.db with
    a NEW file, which changes at least the inode/mtime."""
    try:
        stat = DB_PATH.stat()
    except OSError:
        return None
    return (stat.st_ino, stat.st_mtime_ns, stat.st_size)


def get_engine():
    """The shared engine. For the local SQLite file it is rebuilt whenever the
    file on disk has been replaced: a long-running dashboard (Streamlit Cloud)
    otherwise keeps pooled connections to the OLD file for as long as it
    runs -- the scheduled refreshes only commit a new data/soccer.db, which
    touches no Python file, so Streamlit never restarts the app, and the site
    kept showing data from whenever it last started."""
    global _engine, _engine_file_signature
    if turso_database_url():
        if _engine is None:
            _engine = create_engine(
                f"sqlite+libsql://{turso_database_url()}?secure=true",
                connect_args={"auth_token": turso_auth_token()},
                future=True,
            )
        return _engine

    signature = _db_file_signature()
    if _engine is not None and signature != _engine_file_signature:
        _engine.dispose()  # connections already checked out finish normally
        _engine = None
    if _engine is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
        _engine_file_signature = _db_file_signature()
    return _engine


def ensure_group_name_columns(engine=None) -> list[str]:
    """Adds matches.group_name / fixtures.group_name to a database created
    before those columns existed, returning the tables that were altered.
    Unlike this app's earlier one-off migration scripts, init_db() runs this
    automatically: the deployed dashboard opens a committed data/soccer.db
    that may predate the columns, and every query selecting them would
    otherwise fail until the next scheduled refresh rewrote the file.
    """
    engine = engine or get_engine()
    altered = []
    for table in ("matches", "fixtures"):
        columns = {col["name"] for col in inspect(engine).get_columns(table)}
        if "group_name" in columns:
            continue
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN group_name VARCHAR"))
        altered.append(table)
    return altered


def ensure_prediction_probability_columns(engine=None) -> list[str]:
    """Adds prediction_records.p_home/p_draw/p_away to a database created before
    they existed, returning the columns that were added -- automatic in
    init_db() for the same reason as ensure_group_name_columns."""
    engine = engine or get_engine()
    existing = {col["name"] for col in inspect(engine).get_columns("prediction_records")}
    added = []
    for column in ("p_home", "p_draw", "p_away"):
        if column not in existing:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE prediction_records ADD COLUMN {column} FLOAT"))
            added.append(column)
    return added


def init_db() -> None:
    Base.metadata.create_all(get_engine())
    ensure_group_name_columns()
    ensure_prediction_probability_columns()


@contextmanager
def session_scope() -> Iterator[Session]:
    global _SessionLocal
    engine = get_engine()
    if _SessionLocal is None or _SessionLocal.kw.get("bind") is not engine:
        _SessionLocal = sessionmaker(bind=engine, future=True)
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
