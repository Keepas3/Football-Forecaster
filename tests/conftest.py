"""Shared test setup."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_year_fetch: run the real espn_client.fetch_year_fixtures (its HTTP call still stubbed by the test)"
    )


@pytest.fixture(autouse=True)
def _no_real_year_scoreboard_requests(request, monkeypatch):
    """sync_fixtures_to_db_espn also asks ESPN for a group competition's whole
    year of fixtures (espn_client.fetch_year_fixtures). Without this, any test
    that syncs fixtures for a has_groups league would silently hit the real
    network and store real matches. Tests of that behaviour patch it themselves,
    or opt out with @pytest.mark.real_year_fetch to exercise the real function."""
    if request.node.get_closest_marker("real_year_fetch"):
        return
    from soccer_predictor.ingest import espn_client

    monkeypatch.setattr(espn_client, "fetch_year_fixtures", lambda *a, **k: [])


@pytest.fixture(autouse=True)
def _no_committed_understat_snapshot(tmp_path_factory, monkeypatch):
    """data/understat_snapshot/ holds real player data, used when Understat is down. A test that
    makes Understat fail would otherwise be answered from it (and expect no data). Tests of the
    snapshot itself (test_understat_snapshot.py) point SNAPSHOT_DIR at their own folder."""
    from soccer_predictor.ingest import understat_client

    monkeypatch.setattr(understat_client, "SNAPSHOT_DIR", tmp_path_factory.mktemp("no_understat_snapshot"))


@pytest.fixture(autouse=True)
def _clear_in_memory_caches():
    """ttl_memo results would otherwise carry one test's patched data into the next."""
    from soccer_predictor.ingest import memo, understat_client

    memo.clear_all()
    understat_client._unreachable_until = 0.0  # the rest it takes after a failed request
    yield
    memo.clear_all()
    understat_client._unreachable_until = 0.0
