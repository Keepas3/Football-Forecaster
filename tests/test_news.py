from __future__ import annotations

from datetime import datetime, timezone

import pytest
import requests

from soccer_predictor.config import League
from soccer_predictor.ingest import news


def _item(title="Man City appeal - BBC", link="https://news.google.com/rss/articles/abc", source="BBC",
          pub="Fri, 02 Oct 2026 08:50:24 GMT"):
    parts = [f"<title>{title}</title>", f"<link>{link}</link>", f"<source url='https://x'>{source}</source>"]
    if pub is not None:
        parts.append(f"<pubDate>{pub}</pubDate>")
    return "<item>" + "".join(parts) + "</item>"


def _rss(*items):
    return '<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>' + "".join(items) + "</channel></rss>"


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "CACHE_DIR", tmp_path)


def _serve(monkeypatch, text, calls=None):
    def fake_get(*a, **k):
        if calls is not None:
            calls.append(k.get("params"))
        return _Resp(text)

    monkeypatch.setattr(news.requests, "get", fake_get)


def test_parses_fields_and_strips_source_suffix(monkeypatch):
    _serve(monkeypatch, _rss(_item()))
    [a] = news.fetch_news("Manchester City")
    assert a.title == "Man City appeal"
    assert a.source == "BBC"
    assert a.url == "https://news.google.com/rss/articles/abc"
    assert a.published_at == datetime(2026, 10, 2, 8, 50, 24, tzinfo=timezone.utc)


def test_sorted_newest_first_deduped_and_limited(monkeypatch):
    _serve(
        monkeypatch,
        _rss(
            _item("Old story - BBC", pub="Mon, 28 Sep 2026 10:00:00 GMT"),
            _item("New story - Sky", source="Sky", pub="Fri, 02 Oct 2026 10:00:00 GMT"),
            _item("NEW  story - Sky", source="Sky", pub="Fri, 02 Oct 2026 09:00:00 GMT"),  # dup headline
            _item("Mid story - BBC", pub="Wed, 30 Sep 2026 10:00:00 GMT"),
        ),
    )
    articles = news.fetch_news("x", limit=2)
    assert [a.title for a in articles] == ["New story", "Mid story"]


def test_drops_items_missing_title_or_with_non_https_link_or_bad_date(monkeypatch):
    _serve(
        monkeypatch,
        _rss(
            _item(title=""),
            _item("Insecure - BBC", link="http://example.com/a"),
            _item("Bad date - BBC", pub="not a date"),
            _item("No date - BBC", pub=None),
            _item("Good - BBC"),
        ),
    )
    assert [a.title for a in news.fetch_news("x")] == ["Good"]


def test_second_call_served_from_disk_cache(monkeypatch):
    calls = []
    _serve(monkeypatch, _rss(_item()), calls)
    news.fetch_news("Manchester City")
    news.fetch_news("Manchester City")
    assert len(calls) == 1


def test_query_includes_recency_window(monkeypatch):
    calls = []
    _serve(monkeypatch, _rss(_item()), calls)
    news.fetch_news("Manchester City")
    assert calls[0]["q"] == "Manchester City when:14d"


def test_network_error_returns_empty_list(monkeypatch):
    def boom(*a, **k):
        raise requests.RequestException("down")

    monkeypatch.setattr(news.requests, "get", boom)
    assert news.fetch_news("x") == []


def test_non_rss_payload_returns_empty_list_and_is_not_cached(monkeypatch):
    calls = []
    _serve(monkeypatch, "<html><body>Before you continue</body>", calls)  # malformed XML / consent page
    assert news.fetch_news("x") == []
    assert news.fetch_news("x") == []
    assert len(calls) == 2  # not cached, so the next call retries


def test_query_builders():
    assert news.team_news_query("Newcastle United") == '"Newcastle United" football'
    plain = League(code="LALIGA", name="La Liga", seasons=["2425"])
    renamed = League(code="EPL", name="English Premier League", seasons=["2425"], news_query="Premier League")
    assert news.league_news_query(plain) == "La Liga"
    assert news.league_news_query(renamed) == "Premier League"


def test_search_news_adds_typed_words_to_the_page_scope_with_a_wider_window(monkeypatch):
    calls = []
    _serve(monkeypatch, _rss(_item()), calls)
    articles = news.search_news('"Man City" football', "  115   charges ")
    assert calls[0]["q"] == '"Man City" football 115 charges when:365d'
    assert len(articles) == 1


def test_search_news_blank_text_makes_no_request(monkeypatch):
    calls = []
    _serve(monkeypatch, _rss(_item()), calls)
    assert news.search_news("Premier League", "   ") == []
    assert calls == []


def test_search_news_caps_the_typed_text_length(monkeypatch):
    calls = []
    _serve(monkeypatch, _rss(_item()), calls)
    news.search_news("Premier League", "x" * 500)
    assert calls[0]["q"] == "Premier League " + "x" * news.MAX_SEARCH_TEXT_LENGTH + " when:365d"


def test_search_shows_more_results_than_the_front_page_strip(monkeypatch):
    items = [_item(f"Story {i} - BBC", pub=f"Fri, 02 Oct 2026 {i % 24:02d}:00:00 GMT") for i in range(40)]
    _serve(monkeypatch, _rss(*items))
    assert len(news.fetch_news("x")) == news.DEFAULT_LIMIT
    assert len(news.search_news("x", "y")) == news.SEARCH_LIMIT


def test_same_words_with_different_windows_do_not_share_a_cache_entry(monkeypatch):
    calls = []
    _serve(monkeypatch, _rss(_item()), calls)
    news.fetch_news("Manchester City", window="14d")
    news.fetch_news("Manchester City", window="365d")
    news.fetch_news("Manchester City", window="14d")  # cached
    assert [c["q"] for c in calls] == ["Manchester City when:14d", "Manchester City when:365d"]


def test_match_search_url_is_a_google_search_for_both_teams_and_the_league():
    from soccer_predictor.ingest.news import match_search_url

    assert (
        match_search_url("Greece", "Germany", "UEFA Nations League")
        == "https://www.google.com/search?q=Greece+vs+Germany+UEFA+Nations+League"
    )


def test_match_search_url_encodes_accents_and_reserved_characters():
    from soccer_predictor.ingest.news import match_search_url

    url = match_search_url("Türkiye", "Bosnia & Herzegovina", "League A/B")
    assert url.startswith("https://www.google.com/search?q=")
    assert "T%C3%BCrkiye" in url
    assert "%26" in url and "%2F" in url
    assert " " not in url and "&H" not in url
