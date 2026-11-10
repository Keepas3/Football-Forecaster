"""football-data.org allows 10 requests a minute. The daily refresh failed for three days running
(2026-11-08 to 2026-11-10) on "429 Too Many Requests"; these pin down how that is now avoided."""

from __future__ import annotations

import pytest
import requests

from soccer_predictor.ingest import api_client


class FakeResponse:
    def __init__(self, status_code=200, headers=None, payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload if payload is not None else {"matches": []}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)

    def json(self):
        return self._payload


@pytest.fixture
def sleeps(monkeypatch, tmp_path):
    """No real waiting, no real cache, no key lookup; returns the list of requested sleeps."""
    slept: list[float] = []
    monkeypatch.setattr(api_client.time, "sleep", slept.append)
    monkeypatch.setattr(api_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(api_client, "football_data_org_api_key", lambda: "test-key")
    monkeypatch.setattr(api_client, "_rate_limiter", api_client.RateLimiter(max_per_minute=100))
    monkeypatch.delenv("FOOTBALL_DATA_WAIT_OUT_RATE_LIMIT", raising=False)
    return slept


def _answers(monkeypatch, *responses):
    calls = []
    queue = list(responses)
    monkeypatch.setattr(api_client.http_session, "get", lambda *a, **k: calls.append(a) or queue.pop(0))
    return calls


def test_a_429_is_waited_out_and_retried_when_the_job_asks_for_it(monkeypatch, sleeps):
    monkeypatch.setenv("FOOTBALL_DATA_WAIT_OUT_RATE_LIMIT", "1")
    calls = _answers(
        monkeypatch,
        FakeResponse(429, {"X-RequestCounter-Reset": "30"}),
        FakeResponse(200, payload={"matches": [1]}),
    )

    assert api_client.get("/competitions/2000/matches", {"status": "SCHEDULED"}) == {"matches": [1]}
    assert len(calls) == 2
    assert sleeps == [31.0]  # the server's reset time, plus a second


def test_retry_after_is_preferred_over_the_counter_reset(monkeypatch, sleeps):
    monkeypatch.setenv("FOOTBALL_DATA_WAIT_OUT_RATE_LIMIT", "1")
    _answers(monkeypatch, FakeResponse(429, {"Retry-After": "5", "X-RequestCounter-Reset": "50"}), FakeResponse())

    api_client.get("/x")
    assert sleeps == [6.0]


def test_a_429_with_no_timing_waits_a_full_minute(monkeypatch, sleeps):
    monkeypatch.setenv("FOOTBALL_DATA_WAIT_OUT_RATE_LIMIT", "1")
    _answers(monkeypatch, FakeResponse(429), FakeResponse())

    api_client.get("/x")
    assert sleeps == [61.0]


def test_the_wait_is_never_longer_than_70_seconds(monkeypatch, sleeps):
    monkeypatch.setenv("FOOTBALL_DATA_WAIT_OUT_RATE_LIMIT", "1")
    _answers(monkeypatch, FakeResponse(429, {"Retry-After": "3600"}), FakeResponse())

    api_client.get("/x")
    assert sleeps == [70.0]


def test_it_gives_up_after_the_last_retry(monkeypatch, sleeps):
    monkeypatch.setenv("FOOTBALL_DATA_WAIT_OUT_RATE_LIMIT", "1")
    calls = _answers(monkeypatch, *[FakeResponse(429) for _ in range(api_client.MAX_RATE_LIMIT_RETRIES + 1)])

    with pytest.raises(requests.HTTPError, match="429"):
        api_client.get("/x")
    assert len(calls) == api_client.MAX_RATE_LIMIT_RETRIES + 1
    assert len(sleeps) == api_client.MAX_RATE_LIMIT_RETRIES


def test_the_dashboard_never_sleeps_on_a_429(monkeypatch, sleeps):
    """Without the job's setting a 429 fails at once: a page must not sit through a minute of waiting."""
    calls = _answers(monkeypatch, FakeResponse(429))

    with pytest.raises(requests.HTTPError, match="429"):
        api_client.get("/x")
    assert len(calls) == 1 and sleeps == []


def test_other_errors_are_not_retried(monkeypatch, sleeps):
    monkeypatch.setenv("FOOTBALL_DATA_WAIT_OUT_RATE_LIMIT", "1")
    calls = _answers(monkeypatch, FakeResponse(404))

    with pytest.raises(requests.HTTPError, match="404"):
        api_client.get("/x")
    assert len(calls) == 1 and sleeps == []


def test_a_successful_answer_is_cached(monkeypatch, sleeps):
    calls = _answers(monkeypatch, FakeResponse(200, payload={"matches": [7]}))

    assert api_client.get("/x") == {"matches": [7]}
    assert api_client.get("/x") == {"matches": [7]}  # second call comes from the disk cache
    assert len(calls) == 1


# --- the limiter itself ----------------------------------------------------------------------


def test_the_limiter_keeps_a_margin_past_the_minute_before_reusing_a_slot(monkeypatch):
    """The server stamps a request when it arrives, and the first one arrives late (the connection
    has to open), so the 11th request must not follow the 1st by just over 60 seconds."""
    now = [1000.0]
    slept: list[float] = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(api_client.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(api_client.time, "sleep", sleep)
    limiter = api_client.RateLimiter(max_per_minute=10)

    for _ in range(10):
        limiter.wait()
    assert slept == []  # ten in a burst are fine

    limiter.wait()  # the 11th
    assert slept and slept[0] >= 60 + api_client.WINDOW_MARGIN_SECONDS - 0.001


def test_the_limiter_does_not_sleep_when_the_window_has_passed(monkeypatch):
    now = [1000.0]
    slept: list[float] = []
    monkeypatch.setattr(api_client.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(api_client.time, "sleep", slept.append)
    limiter = api_client.RateLimiter(max_per_minute=2)

    limiter.wait()
    limiter.wait()
    now[0] += 60 + api_client.WINDOW_MARGIN_SECONDS + 1
    limiter.wait()

    assert slept == []
