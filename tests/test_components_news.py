from __future__ import annotations

import datetime as dt

from soccer_predictor.dashboard.components import _relative_time, match_team_crest, news_ticker_html
from soccer_predictor.ingest.news import NewsArticle

NOW = dt.datetime(2026, 10, 2, 12, 0, tzinfo=dt.UTC)


def _article(title="Man City appeal", url="https://news.google.com/rss/articles/abc", source="BBC", hours_ago=2):
    return NewsArticle(title=title, url=url, source=source, published_at=NOW - dt.timedelta(hours=hours_ago))


def test_relative_time_buckets():
    assert _relative_time(NOW - dt.timedelta(seconds=20), NOW) == "just now"
    assert _relative_time(NOW - dt.timedelta(minutes=5), NOW) == "5m ago"
    assert _relative_time(NOW - dt.timedelta(hours=3, minutes=10), NOW) == "3h ago"
    assert _relative_time(NOW - dt.timedelta(days=2, hours=5), NOW) == "2d ago"


def test_relative_time_future_timestamp_is_just_now():
    assert _relative_time(NOW + dt.timedelta(minutes=10), NOW) == "just now"


def test_match_team_crest_finds_named_team_case_insensitively():
    crests = {"Man City": "https://c/city.png", "Arsenal": "https://c/ars.png"}
    assert match_team_crest("MAN CITY charges: Etihad considering legal action", crests) == "https://c/city.png"


def test_match_team_crest_longest_name_wins():
    crests = {"Man": "https://c/short.png", "Man City": "https://c/city.png"}
    assert match_team_crest("Man City lodge appeal", crests) == "https://c/city.png"


def test_match_team_crest_none_when_no_team_named():
    assert match_team_crest("Deadline day wrap", {"Arsenal": "https://c/ars.png"}) is None


def test_ticker_empty_for_no_articles():
    assert news_ticker_html([]) == ""


def test_ticker_links_open_in_new_tab_with_noopener_and_show_headline_and_arrow():
    html_out = news_ticker_html([_article()], now=NOW)
    assert 'href="https://news.google.com/rss/articles/abc"' in html_out
    assert 'target="_blank"' in html_out and 'rel="noopener noreferrer"' in html_out
    assert ">Man City appeal<" in html_out
    assert "↗" in html_out
    assert "BBC, 2h ago" in html_out  # tooltip
    assert "via Google News" in html_out


def test_ticker_escapes_hostile_headline_source_and_url():
    hostile = _article(
        title='<script>alert(1)</script> "quoted" $5m',
        url='https://example.com/a?x="onmouseover="alert(1)',
        source='"><img src=x onerror=alert(1)>',
    )
    html_out = news_ticker_html([hostile], now=NOW)
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert '"onmouseover="' not in html_out  # attribute break-out neutralized
    assert "<img src=x" not in html_out


def test_ticker_uses_matched_crest_then_default_icon_then_none():
    crests = {"Man City": "https://c/city.png"}
    matched = news_ticker_html([_article("Man City appeal")], crests, default_icon="https://c/default.png", now=NOW)
    assert 'src="https://c/city.png"' in matched and "default.png" not in matched

    fallback = news_ticker_html([_article("Deadline day")], crests, default_icon="https://c/default.png", now=NOW)
    assert 'src="https://c/default.png"' in fallback

    plain = news_ticker_html([_article("Deadline day")], crests, now=NOW)
    assert "<img" not in plain


def test_results_list_empty_for_no_articles():
    from soccer_predictor.dashboard.components import news_results_html

    assert news_results_html([]) == ""


def test_results_list_shows_full_headline_source_age_and_safe_link():
    from soccer_predictor.dashboard.components import news_results_html

    long_title = "A very long headline about a club " * 6
    html_out = news_results_html([_article(title=long_title.strip())], now=NOW)
    assert long_title.strip() in html_out  # not truncated, unlike the strip
    assert "BBC · 2h ago" in html_out
    assert 'target="_blank"' in html_out and 'rel="noopener noreferrer"' in html_out


def test_results_list_escapes_hostile_content_and_matches_crests():
    from soccer_predictor.dashboard.components import news_results_html

    hostile = _article(title='Man City <img src=x onerror=alert(1)> "q"', source="<b>x</b>")
    html_out = news_results_html([hostile], {"Man City": "https://c/city.png"}, now=NOW)
    assert "<img src=x" not in html_out and "<b>x</b>" not in html_out
    assert 'src="https://c/city.png"' in html_out
