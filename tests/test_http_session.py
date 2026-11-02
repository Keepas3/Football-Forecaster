from __future__ import annotations

from soccer_predictor.ingest import http_session


class _Recorder:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.name


def test_a_post_retries_a_failed_connection_unless_it_asks_for_a_single_attempt(monkeypatch):
    shared, once = _Recorder("shared"), _Recorder("once")
    monkeypatch.setattr(http_session, "_session", shared)
    monkeypatch.setattr(http_session, "_single_attempt_session", once)

    assert http_session.post("https://example.test/a", data={"x": 1}) == "shared"
    assert http_session.post("https://example.test/b", single_attempt=True, timeout=(4, 15)) == "once"

    assert shared.calls == [("https://example.test/a", {"data": {"x": 1}})]
    assert once.calls == [("https://example.test/b", {"timeout": (4, 15)})]  # the flag is not passed on to requests


def test_the_single_attempt_session_has_no_connection_retries_and_the_shared_one_keeps_its_two():
    shared_retry = http_session._session.get_adapter("https://example.test").max_retries
    once_retry = http_session._single_attempt_session.get_adapter("https://example.test").max_retries

    assert shared_retry.connect == 2 and shared_retry.read == 0
    assert once_retry.total == 0 and once_retry.connect is None


def test_neither_session_accepts_cookies():
    """Both are shared by every dashboard viewer, so one visitor's cookie must never ride on another's request."""
    for session in (http_session._session, http_session._single_attempt_session):
        assert session.cookies._policy.allowed_domains() == ()  # the policy that refuses every domain
