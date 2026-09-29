from __future__ import annotations

import requests

from soccer_predictor.config import League
from soccer_predictor.ingest import refresh
from soccer_predictor.ingest.api_client import MissingApiKey

CSV_LEAGUE = League(code="EPL", name="English Premier League", csv_code="E0", seasons=["2526"])
NO_CSV_LEAGUE = League(code="UCL", name="UEFA Champions League", seasons=["2627"])
ESPN_LEAGUE = League(
    code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1"
)


def _patch_common(monkeypatch, params=None):
    """Stubs everything refresh_league touches so no real DB/network is hit."""
    monkeypatch.setattr(refresh, "seed_teams_from_api", lambda *a, **k: 0)
    monkeypatch.setattr(refresh, "seed_teams_from_espn", lambda *a, **k: 0)
    monkeypatch.setattr(refresh, "load_latest_params", lambda *a, **k: params)
    monkeypatch.setattr(refresh, "snapshot_upcoming_predictions", lambda *a, **k: 3 if params else 0)
    monkeypatch.setattr(refresh.injuries_module, "sync_injuries_to_db", lambda *a, **k: (5, 1))

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(refresh, "session_scope", lambda: _FakeSession())


def test_csv_league_never_calls_results_sync(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(refresh, "sync_fixtures_to_db", lambda *a, **k: (10, 0))

    def fail_if_called(*a, **k):
        raise AssertionError("sync_results_to_db should never be called for a CSV-backed league")

    monkeypatch.setattr(refresh, "sync_results_to_db", fail_if_called)

    result = refresh.refresh_league(CSV_LEAGUE)

    assert result.fixtures_synced == 10
    assert result.results_synced == 0
    assert result.results_error is None
    assert result.injuries_synced == 5
    assert result.injuries_skipped == 1


def test_no_csv_football_data_org_league_syncs_results_too(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(refresh, "sync_fixtures_to_db", lambda *a, **k: (20, 1))
    monkeypatch.setattr(refresh, "sync_results_to_db", lambda *a, **k: (15, 2))

    result = refresh.refresh_league(NO_CSV_LEAGUE)

    assert result.fixtures_synced == 20
    assert result.fixtures_skipped == 1
    assert result.results_synced == 15
    assert result.results_skipped == 2


def test_espn_league_uses_espn_functions(monkeypatch):
    _patch_common(monkeypatch)

    def fail_if_called(*a, **k):
        raise AssertionError("football-data.org sync functions should never fire for an ESPN league")

    monkeypatch.setattr(refresh, "sync_fixtures_to_db", fail_if_called)
    monkeypatch.setattr(refresh, "sync_results_to_db", fail_if_called)
    monkeypatch.setattr(refresh, "sync_fixtures_to_db_espn", lambda *a, **k: (30, 0))
    monkeypatch.setattr(refresh, "sync_results_to_db_espn", lambda *a, **k: (25, 0))

    result = refresh.refresh_league(ESPN_LEAGUE)

    assert result.fixtures_synced == 30
    assert result.results_synced == 25


def test_espn_request_exception_is_captured_not_raised(monkeypatch):
    _patch_common(monkeypatch)

    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(refresh, "sync_fixtures_to_db_espn", raise_error)

    result = refresh.refresh_league(ESPN_LEAGUE)

    assert result.fixtures_error == "network error"
    assert result.results_error == "network error"
    # Injuries (a later, independent step) should still have run.
    assert result.injuries_synced == 5


def test_fixtures_failure_does_not_block_later_steps(monkeypatch):
    _patch_common(monkeypatch, params=object())

    def raise_missing_key(*a, **k):
        raise MissingApiKey("no key")

    monkeypatch.setattr(refresh, "sync_fixtures_to_db", raise_missing_key)
    monkeypatch.setattr(refresh, "sync_results_to_db", lambda *a, **k: (0, 0))

    result = refresh.refresh_league(NO_CSV_LEAGUE)

    assert result.fixtures_error == "no key"
    # Prediction tracking and injuries are independent try blocks -- both
    # should still have run even though fixtures failed.
    assert result.snapshots_created == 3
    assert result.injuries_synced == 5


def test_no_trained_model_means_zero_snapshots_no_error(monkeypatch):
    _patch_common(monkeypatch, params=None)
    monkeypatch.setattr(refresh, "sync_fixtures_to_db", lambda *a, **k: (0, 0))

    result = refresh.refresh_league(CSV_LEAGUE)

    assert result.snapshots_created == 0


