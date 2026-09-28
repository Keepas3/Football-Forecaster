from __future__ import annotations

import soccer_predictor.storage.db as db


def _reset_engine_singleton(monkeypatch):
    # get_engine() memoizes into a module-level global -- reset it so each
    # test starts fresh regardless of call order.
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_SessionLocal", None)


def test_uses_local_sqlite_when_turso_not_configured(monkeypatch):
    _reset_engine_singleton(monkeypatch)
    monkeypatch.setattr(db, "turso_database_url", lambda: None)
    monkeypatch.setattr(db, "turso_auth_token", lambda: None)

    calls = []

    def fake_create_engine(url, **kwargs):
        calls.append((url, kwargs))
        return object()

    monkeypatch.setattr(db, "create_engine", fake_create_engine)

    db.get_engine()

    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url.startswith("sqlite:///")
    assert "connect_args" not in kwargs


def test_uses_turso_when_configured(monkeypatch):
    _reset_engine_singleton(monkeypatch)
    monkeypatch.setattr(db, "turso_database_url", lambda: "my-db-myorg.turso.io")
    monkeypatch.setattr(db, "turso_auth_token", lambda: "fake-token")

    calls = []

    def fake_create_engine(url, **kwargs):
        calls.append((url, kwargs))
        return object()

    monkeypatch.setattr(db, "create_engine", fake_create_engine)

    db.get_engine()

    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "sqlite+libsql://my-db-myorg.turso.io?secure=true"
    assert kwargs["connect_args"] == {"auth_token": "fake-token"}


def test_get_engine_is_memoized(monkeypatch):
    _reset_engine_singleton(monkeypatch)
    monkeypatch.setattr(db, "turso_database_url", lambda: None)
    monkeypatch.setattr(db, "turso_auth_token", lambda: None)

    calls = []
    monkeypatch.setattr(db, "create_engine", lambda *a, **k: calls.append(1) or object())

    db.get_engine()
    db.get_engine()

    assert len(calls) == 1
