"""scripts/lock_due_check.py: the near-free question the lock-lineups workflow asks first."""

from __future__ import annotations

import datetime as dt
import importlib.util
import sqlite3
from pathlib import Path

import pytest

from soccer_predictor.prediction.tracking import LINEUP_WINDOW_MINUTES

NOW = dt.datetime(2026, 10, 31, 14, 0)


def _check_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "lock_due_check.py"
    spec = importlib.util.spec_from_file_location("lock_due_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "soccer.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE fixtures (id INTEGER PRIMARY KEY, league_code TEXT, date DATE, home_team_id INT, "
        "away_team_id INT, status TEXT, kickoff_utc DATETIME)"
    )
    con.execute(
        "CREATE TABLE prediction_records (id INTEGER PRIMARY KEY, league_code TEXT, date DATE, "
        "home_team_id INT, away_team_id INT, basis TEXT)"
    )
    con.commit()
    con.close()
    return path


def _fixture(db, kickoff, home=1, away=2):
    con = sqlite3.connect(db)
    stored = kickoff.strftime("%Y-%m-%d %H:%M:%S.%f") if kickoff else None  # as SQLAlchemy writes it
    con.execute(
        "INSERT INTO fixtures (league_code, date, home_team_id, away_team_id, status, kickoff_utc) "
        "VALUES ('EPL', ?, ?, ?, 'TIMED', ?)",
        (NOW.date().isoformat(), home, away, stored),
    )
    con.commit()
    con.close()


def _lock(db, basis, home=1, away=2):
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO prediction_records (league_code, date, home_team_id, away_team_id, basis) "
        "VALUES ('EPL', ?, ?, ?, ?)",
        (NOW.date().isoformat(), home, away, basis),
    )
    con.commit()
    con.close()


def test_the_check_window_covers_the_lock_window():
    assert _check_module().CHECK_WINDOW_MINUTES >= LINEUP_WINDOW_MINUTES


def test_a_match_about_to_start_with_no_lock_is_due(db):
    _fixture(db, NOW + dt.timedelta(minutes=60))
    assert _check_module().is_due(db, NOW)


def test_nothing_is_due_without_a_match_in_the_window(db):
    check = _check_module().is_due
    assert not check(db, NOW)  # no fixtures at all
    _fixture(db, NOW + dt.timedelta(hours=5))
    _fixture(db, NOW - dt.timedelta(minutes=5), home=3, away=4)  # already started
    _fixture(db, None, home=5, away=6)  # no kickoff time
    assert not check(db, NOW)


def test_a_match_with_a_lineup_or_early_lock_is_not_due(db):
    _fixture(db, NOW + dt.timedelta(minutes=60))
    _lock(db, "lineup")
    assert not _check_module().is_due(db, NOW)


def test_a_match_locked_only_by_the_older_rule_is_still_due(db):
    _fixture(db, NOW + dt.timedelta(minutes=60))
    _lock(db, None)
    assert _check_module().is_due(db, NOW)


def test_a_database_from_before_the_basis_column_has_nothing_locked(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE fixtures (league_code TEXT, date DATE, home_team_id INT, away_team_id INT, kickoff_utc DATETIME)")
    con.execute("CREATE TABLE prediction_records (league_code TEXT, date DATE, home_team_id INT, away_team_id INT)")
    con.execute("INSERT INTO fixtures VALUES ('EPL', '2026-10-31', 1, 2, '2026-10-31 15:00:00.000000')")
    con.execute("INSERT INTO prediction_records VALUES ('EPL', '2026-10-31', 1, 2)")
    con.commit()
    con.close()
    assert _check_module().is_due(path, NOW)


def test_main_prints_and_writes_the_github_output_line(db, tmp_path, monkeypatch, capsys):
    _fixture(db, NOW + dt.timedelta(minutes=30))
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    module = _check_module()
    monkeypatch.setattr(module.sys, "argv", ["lock_due_check.py", str(db)])
    monkeypatch.setattr(module, "is_due", lambda path: True)

    module.main()

    assert capsys.readouterr().out.strip() == "due=true"
    assert output.read_text() == "due=true\n"
