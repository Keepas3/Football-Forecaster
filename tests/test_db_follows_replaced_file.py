"""The dashboard must notice when data/soccer.db is replaced underneath it.

The scheduled refreshes only commit a new database file (no Python file
changes), so Streamlit Cloud doesn't restart the app -- and a cached engine
kept reading the old file, which froze the Track Record on stale numbers."""

from __future__ import annotations

import os
import sqlite3

import pytest
from sqlalchemy import text

import soccer_predictor.storage.db as db


def _make_db(path, value):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (x INTEGER)")
    con.execute("INSERT INTO t VALUES (?)", (value,))
    con.commit()
    con.close()


def _read(session_scope):
    with session_scope() as session:
        return session.execute(text("SELECT x FROM t")).scalar_one()


def _use_local_file(monkeypatch, tmp_path):
    path = tmp_path / "soccer.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "turso_database_url", lambda: None)
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_engine_file_signature", None)
    monkeypatch.setattr(db, "_SessionLocal", None)
    return path


def test_sessions_see_a_different_database_file_once_it_is_in_place(monkeypatch, tmp_path):
    """Portable stand-in for a file swap: the engine was built for one file,
    and the file now at DB_PATH is a different one (different inode/mtime)."""
    path = _use_local_file(monkeypatch, tmp_path)
    _make_db(path, 1)
    assert _read(db.session_scope) == 1  # opens (and pools a connection to) the file

    replacement = tmp_path / "other" / "soccer.db"
    replacement.parent.mkdir()
    _make_db(replacement, 2)
    monkeypatch.setattr(db, "DB_PATH", replacement)

    assert _read(db.session_scope) == 2


def test_sessions_see_a_database_file_that_was_swapped_in_place(monkeypatch, tmp_path):
    """The real thing a `git pull` does on Linux (Streamlit Cloud): write a
    new file and rename it over the old one while connections are open."""
    path = _use_local_file(monkeypatch, tmp_path)
    _make_db(path, 1)
    assert _read(db.session_scope) == 1

    incoming = tmp_path / "incoming.db"
    _make_db(incoming, 2)
    try:
        os.replace(incoming, path)
    except PermissionError:
        pytest.skip("Windows can't replace a file that is open; Linux can (covered by the test above)")

    assert _read(db.session_scope) == 2


def test_an_unchanged_file_keeps_the_same_engine(monkeypatch, tmp_path):
    path = _use_local_file(monkeypatch, tmp_path)
    _make_db(path, 1)

    first = db.get_engine()
    for _ in range(3):
        assert db.get_engine() is first


def test_the_apps_own_writes_do_not_break_later_reads(monkeypatch, tmp_path):
    path = _use_local_file(monkeypatch, tmp_path)
    _make_db(path, 1)

    with db.session_scope() as session:
        session.execute(text("UPDATE t SET x = 5"))
    assert _read(db.session_scope) == 5


def test_a_missing_file_does_not_crash_the_signature_check(monkeypatch, tmp_path):
    _use_local_file(monkeypatch, tmp_path)
    assert db._db_file_signature() is None
