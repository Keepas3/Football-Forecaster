"""Recent news headlines for a league or team, for the dashboard's "Latest
news" sections -- a display-time-only fetch (never persisted), same pattern
as ingest/live_scores.py: short-TTL disk cache, degrades to an empty list on
any failure, never raises.

Source: Google News' public RSS search feed. Keyless, and confirmed live
(2026-10) to return ~100 items per query for a league ("Premier League"), a
club ("Manchester City") and a topic ("Manchester City 115 charges"), each
with a headline, publisher name, publish date and a link that resolves to the
article. Its feed terms say it is for personal, non-commercial use -- fine for
this hobby app. If Google ever serves a consent/captcha page instead of RSS
(possible for some hosted IPs), parsing fails and callers just get [].

ESPN's news endpoint also works but is league-level only (most leagues here
have no ESPN team ids), so it can't serve team pages.
"""

from __future__ import annotations

import hashlib
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from soccer_predictor.config import DATA_DIR, League

FEED_URL = "https://news.google.com/rss/search"
CACHE_DIR = DATA_DIR / "cache" / "news"
# News moves slowly relative to page reruns and concurrent viewers; 30 min
# keeps this to a couple of requests an hour per query.
CACHE_TTL_SECONDS = 30 * 60
RECENCY_WINDOW = "14d"
DEFAULT_LIMIT = 8
# A user-typed search looks further back than the front-page strip (someone
# searching "115 charges" wants the whole story, not just the last two weeks)
# and shows more results. Google's feed caps any single query at 100 items.
SEARCH_RECENCY_WINDOW = "365d"
SEARCH_LIMIT = 25
MAX_SEARCH_TEXT_LENGTH = 100


@dataclass
class NewsArticle:
    title: str
    url: str
    source: str
    published_at: datetime  # timezone-aware UTC


def team_news_query(team_name: str) -> str:
    # "football" disambiguates clubs whose names are also places/words
    # (Newcastle, Brighton, Fulham) and works for national teams too.
    return f'"{team_name}" football'


def league_news_query(league: League) -> str:
    return league.news_query or league.name


def _cache_path(full_query: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha256(full_query.encode()).hexdigest()}.xml"


def _fetch_feed_text(query: str, window: str) -> str:
    # Keyed on the full query including the window, so the same words with a
    # different time window never share a cache entry.
    full_query = f"{query} when:{window}"
    cache_file = _cache_path(full_query)
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_TTL_SECONDS:
        return cache_file.read_text(encoding="utf-8")

    response = requests.get(
        FEED_URL,
        params={"q": full_query, "hl": "en-GB", "gl": "GB", "ceid": "GB:en"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    response.raise_for_status()
    text = response.text
    # Parse before caching so a consent page / bad payload never gets cached
    # for the full TTL.
    ET.fromstring(text)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text, encoding="utf-8")
    return text


def _parse_item(item: ET.Element) -> NewsArticle | None:
    title = (item.findtext("title") or "").strip()
    url = (item.findtext("link") or "").strip()
    source = (item.findtext("source") or "").strip()
    if not title or not url.startswith("https://"):
        return None
    # Google appends " - {Publisher}" to every title; the publisher is shown
    # separately, so strip the duplicate.
    suffix = f" - {source}"
    if source and title.endswith(suffix):
        title = title[: -len(suffix)].rstrip()
    try:
        published_at = parsedate_to_datetime(item.findtext("pubDate") or "")
    except (TypeError, ValueError):
        return None
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    return NewsArticle(title=title, url=url, source=source or "Unknown", published_at=published_at)


def fetch_news(query: str, limit: int = DEFAULT_LIMIT, window: str = RECENCY_WINDOW) -> list[NewsArticle]:
    """The `limit` most recent articles matching `query` within `window`
    (Google's "when:" syntax, e.g. "14d"), newest first, de-duplicated by
    headline. Empty list on any failure.
    """
    try:
        root = ET.fromstring(_fetch_feed_text(query, window))
    except (requests.RequestException, ET.ParseError, OSError):
        return []

    articles: list[NewsArticle] = []
    seen: set[str] = set()
    for item in root.findall("./channel/item"):
        article = _parse_item(item)
        if article is None:
            continue
        key = " ".join(article.title.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        articles.append(article)

    articles.sort(key=lambda a: a.published_at, reverse=True)
    return articles[:limit]


def search_news(base_query: str, text: str, limit: int = SEARCH_LIMIT) -> list[NewsArticle]:
    """News matching what a user typed into a search box, scoped to the page
    they're on: `base_query` is the league/team query (so typing "115
    charges" on Man City's page searches Man City news), and the typed words
    are added to it. Sent to Google as a fresh query -- it searches the whole
    feed, not just the handful of headlines already on screen -- over a
    wider time window than the front-page strip. Blank text returns [].
    """
    text = " ".join(text.split())[:MAX_SEARCH_TEXT_LENGTH]
    if not text:
        return []
    return fetch_news(f"{base_query} {text}", limit=limit, window=SEARCH_RECENCY_WINDOW)
