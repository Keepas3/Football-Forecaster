"""Thin, rate-limited, caching HTTP client for api-sports.io (API-Football).

Free tier is a hard 100 requests/DAY (not just a per-minute throttle like
football-data.org) -- disk caching here matters even more, since a wasted
call can't be made up later that day. Mirrors ingest/api_client.py's shape.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from pathlib import Path

import requests

from soccer_predictor.config import DATA_DIR, api_football_key

BASE_URL = "https://v3.football.api-sports.io"
CACHE_DIR = DATA_DIR / "cache" / "api_football"
MAX_REQUESTS_PER_MINUTE = 10  # free tier's per-minute cap, separate from its 100/day cap


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


def get(path: str, params: dict | None = None, cache_ttl_seconds: int = 24 * 3600) -> dict:
    """GETs `{BASE_URL}{path}`, serving from an on-disk cache within TTL."""
    api_key = api_football_key()
    if not api_key:
        raise MissingApiKey(
            "API_FOOTBALL_KEY not set -- copy .env.example to .env and fill it in "
            "(free registration at https://dashboard.api-football.com/register)"
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
        headers={"x-apisports-key": api_key},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    # API-Football signals a quota/plan problem (e.g. the 100/day cap) inside
    # a 200 OK body's "errors" field, not an HTTP error status -- so
    # raise_for_status() above never catches it. Caching that response would
    # otherwise poison every future call for this exact (path, params) with
    # a false "no results" for the rest of cache_ttl_seconds (up to 24h),
    # long after the quota itself has reset.
    if data.get("errors"):
        return data

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data
