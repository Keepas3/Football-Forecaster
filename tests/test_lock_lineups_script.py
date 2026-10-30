"""scripts/lock_lineups.py: runs the lock for every league with a trained model and reports the total."""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path

from soccer_predictor.config import League


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "lock_lineups.py"
    spec = importlib.util.spec_from_file_location("lock_lineups_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_locks_each_league_that_has_a_model_and_reports_the_total(monkeypatch, capsys):
    script = _script()
    leagues = {
        "EPL": League(code="EPL", name="Premier League", seasons=["2627"]),
        "WC": League(code="WC", name="World Cup", seasons=["2026"]),  # no trained model
    }
    locked_for = []

    @contextmanager
    def fake_session():
        yield object()

    monkeypatch.setattr(script, "init_db", lambda: None)
    monkeypatch.setattr(script, "load_leagues", lambda: leagues)
    monkeypatch.setattr(script, "session_scope", fake_session)
    monkeypatch.setattr(script, "load_latest_params", lambda session, code: "params" if code == "EPL" else None)
    monkeypatch.setattr(
        script, "lock_due_predictions", lambda session, league, params: locked_for.append(league.code) or 2
    )
    monkeypatch.setattr(script.sys, "argv", ["lock_lineups.py"])

    script.main()

    assert locked_for == ["EPL"]
    out = capsys.readouterr().out
    assert "Premier League (EPL): 2 forecast(s) locked in" in out
    assert out.strip().endswith("2 forecast(s) locked in")


def test_leagues_can_be_named_on_the_command_line(monkeypatch, capsys):
    script = _script()
    leagues = {code: League(code=code, name=code, seasons=["2627"]) for code in ("EPL", "LALIGA")}
    seen = []

    @contextmanager
    def fake_session():
        yield object()

    monkeypatch.setattr(script, "init_db", lambda: None)
    monkeypatch.setattr(script, "load_leagues", lambda: leagues)
    monkeypatch.setattr(script, "session_scope", fake_session)
    monkeypatch.setattr(script, "load_latest_params", lambda session, code: "params")
    monkeypatch.setattr(script, "lock_due_predictions", lambda session, league, params: seen.append(league.code) or 0)
    monkeypatch.setattr(script.sys, "argv", ["lock_lineups.py", "LALIGA"])

    script.main()

    assert seen == ["LALIGA"]
    assert capsys.readouterr().out.strip() == "0 forecast(s) locked in"
