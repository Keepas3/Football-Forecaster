"""Thin, rate-limited, caching HTTP client for football-data.org.

Free tier is 10 requests/minute -- the dashboard must never trigger a live
call per page render, so every GET here is cached to disk with a TTL the
caller chooses (fixtures barely change within a few hours; injuries are
refreshed daily at most).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from pathlib import Path

import requests

from soccer_predictor.config import DATA_DIR, football_data_org_api_key

BASE_URL = "https://api.football-data.org/v4"
CACHE_DIR = DATA_DIR / "cache" / "football_data_org"
MAX_REQUESTS_PER_MINUTE = 10


class MissingApiKey(Exception):
    pass


class RateLimiter:
    """Blocks just long enough to keep requests under N per rolling minute."""

    def __init__(self, max_per_minute: int = MAX_REQUESTS_PER_MINUTE):
        self.max_per_minute = max_per_minute
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] > 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_per_minute:
            sleep_for = 60 - (now - self._timestamps[0]) + 0.1
            time.sleep(max(sleep_for, 0))
        self._timestamps.append(time.monotonic())


_rate_limiter = RateLimiter()


def _cache_path(path: str, params: dict) -> Path:
    key = hashlib.sha256(f"{path}?{sorted(params.items())}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def get(path: str, params: dict | None = None, cache_ttl_seconds: int = 6 * 3600) -> dict:
    """GETs `{BASE_URL}{path}`, serving from an on-disk cache within TTL."""
    api_key = football_data_org_api_key()
    if not api_key:
        raise MissingApiKey(
            "FOOTBALL_DATA_ORG_API_KEY not set -- copy .env.example to .env and fill it in "
            "(free registration at https://www.football-data.org/client/register)"
        )

    params = params or {}
    cache_file = _cache_path(path, params)
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < cache_ttl_seconds:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    _rate_limiter.wait()
    response = requests.get(
        f"{BASE_URL}{path}",
        params=params,
        headers={"X-Auth-Token": api_key},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data
