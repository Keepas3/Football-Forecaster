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
_SessionLocal: sessionmaker | None = None


def get_engine():
    global _engine
    if _engine is None:
        turso_url = turso_database_url()
        if turso_url:
            _engine = create_engine(
                f"sqlite+libsql://{turso_url}?secure=true",
                connect_args={"auth_token": turso_auth_token()},
                future=True,
            )
        else:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
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


def init_db() -> None:
    Base.metadata.create_all(get_engine())
    ensure_group_name_columns()


@contextmanager
def session_scope() -> Iterator[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), future=True)
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
