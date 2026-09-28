"""Engine/session setup -- the local soccer.db SQLite file by default, or a
Turso (libSQL) database instead when TURSO_DATABASE_URL/TURSO_AUTH_TOKEN are
set (see config.py). Turso exists for deployments with no persistent local
disk (e.g. Streamlit Community Cloud); local development is unaffected
unless those env vars are explicitly set.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
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


def init_db() -> None:
    Base.metadata.create_all(get_engine())


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
