"""Fetches league (competition) metadata -- currently just the emblem --
from football-data.org, for the Leagues landing page.

Always degrades to None rather than raising: a missing emblem should show a
placeholder icon, never break the page. Cheap to call on every page render
since ingest/api_client.py::get() already disk-caches by path+params.
"""

from __future__ import annotations

import requests

from soccer_predictor.config import League
from soccer_predictor.ingest import api_client

LEAGUE_META_CACHE_TTL_SECONDS = 7 * 24 * 3600  # emblems essentially never change


def fetch_competition_emblem(league: League) -> str | None:
    try:
        data = api_client.get(
            f"/competitions/{league.api_competition_id}",
            cache_ttl_seconds=LEAGUE_META_CACHE_TTL_SECONDS,
        )
    except (api_client.MissingApiKey, requests.RequestException):
        return None
    return data.get("emblem")
