"""scripts/fetch_historical_data.py: a slow football-data.co.uk file is retried, and
one that still fails doesn't stop the other leagues."""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from soccer_predictor.config import League


@pytest.fixture
def script(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_historical_data.py"
    spec = importlib.util.spec_from_file_location("fetch_historical_data_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RAW_DIR", tmp_path)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module


def _response(status=200, body=b"Date,HomeTeam\n"):
    def raise_for_status():
        if status >= 400:
            raise requests.HTTPError(f"{status}", response=SimpleNamespace(status_code=status))

    return SimpleNamespace(status_code=status, content=body, raise_for_status=raise_for_status)


def _getter(*outcomes):
    """requests.get stand-in that plays `outcomes` in order (an exception is raised)."""
    calls = []

    def get(url, timeout=None):
        calls.append((url, timeout))
        outcome = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    get.calls = calls
    return get


def test_a_timeout_is_retried_and_the_file_still_arrives(script, monkeypatch):
    get = _getter(requests.ReadTimeout("slow"), requests.ConnectionError("reset"), _response())
    monkeypatch.setattr(script.requests, "get", get)

    path = script.download_season_csv("D1", "2627", force=True)

    assert path.read_bytes() == b"Date,HomeTeam\n" and len(get.calls) == 3
    assert all(timeout == script.CSV_TIMEOUT_SECONDS for _, timeout in get.calls)


def test_it_gives_up_after_the_last_attempt(script, monkeypatch):
    get = _getter(requests.ReadTimeout("slow"))
    monkeypatch.setattr(script.requests, "get", get)
    with pytest.raises(script.CsvDownloadFailed, match="ReadTimeout"):
        script.download_season_csv("D1", "2627", force=True)
    assert len(get.calls) == script.CSV_DOWNLOAD_ATTEMPTS


def test_server_errors_are_retried_but_a_missing_file_is_not(script, monkeypatch):
    get = _getter(_response(503), _response())
    monkeypatch.setattr(script.requests, "get", get)
    assert script.download_season_csv("D1", "2627", force=True).exists() and len(get.calls) == 2

    missing = _getter(_response(404))
    monkeypatch.setattr(script.requests, "get", missing)
    with pytest.raises(script.CsvDownloadFailed):
        script.download_season_csv("XX", "2627", force=True)
    assert len(missing.calls) == 1  # a 404 will not get better by asking again


def test_an_existing_file_is_not_downloaded_again_unless_forced(script, monkeypatch, tmp_path):
    (tmp_path / "D1_2526.csv").write_text("x")
    monkeypatch.setattr(script.requests, "get", lambda *a, **k: pytest.fail("should not download"))
    assert script.download_season_csv("D1", "2526").name == "D1_2526.csv"


# --- the refresh as a whole ------------------------------------------------------------------


def _run_main(script, monkeypatch, failing_codes):
    leagues = {
        code: League(code=code, name=code, csv_code=csv, seasons=["2627"])
        for code, csv in (("EPL", "E0"), ("BUNDESLIGA", "D1"), ("LALIGA", "SP1"))
    }
    ingested = []

    def download(csv_code, season, force=False):
        if csv_code in failing_codes:
            raise script.CsvDownloadFailed(f"{csv_code}: ReadTimeout")
        return Path(f"{csv_code}.csv")

    monkeypatch.setattr(script, "init_db", lambda: None)
    monkeypatch.setattr(script, "load_leagues", lambda: leagues)
    monkeypatch.setattr(script, "seed_teams_and_aliases", lambda *a: None)
    monkeypatch.setattr(script, "session_scope", lambda: contextlib.nullcontext())
    monkeypatch.setattr(script, "download_season_csv", download)
    monkeypatch.setattr(script, "parse_csv", lambda path: path.name)
    monkeypatch.setattr(script, "ingest_into_db", lambda session, code, season, df: ingested.append(code) or (1, 0))
    monkeypatch.setattr(script.sys, "argv", ["fetch_historical_data.py", "--current-season-only"])
    script.main()
    return ingested


def test_one_league_failing_to_download_does_not_stop_the_others(script, monkeypatch, capsys):
    ingested = _run_main(script, monkeypatch, failing_codes={"D1"})
    assert ingested == ["EPL", "LALIGA"]
    out = capsys.readouterr().out
    assert "::warning::BUNDESLIGA 2627" in out and "Could not download: BUNDESLIGA 2627" in out


def test_every_download_failing_is_an_outage_and_fails_the_job(script, monkeypatch):
    with pytest.raises(SystemExit) as stopped:
        _run_main(script, monkeypatch, failing_codes={"E0", "D1", "SP1"})
    assert stopped.value.code == 1
