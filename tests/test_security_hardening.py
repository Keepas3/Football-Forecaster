"""Password-gate lockouts, the news-search budget, note-parser output validation
and the Streamlit error-details setting."""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from soccer_predictor.ai import note_parser
from soccer_predictor.dashboard import notes_access as na
from soccer_predictor.ingest import news
from soccer_predictor.limits import SlidingWindow


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# --- sliding window --------------------------------------------------------------------------


def test_sliding_window_blocks_when_full_and_frees_up_as_events_age_out():
    clock = FakeClock()
    window = SlidingWindow(3, 60, clock)
    assert [window.allow() for _ in range(4)] == [True, True, True, False]
    assert window.blocked()
    clock.now = 61
    assert not window.blocked() and window.allow()


def test_record_counts_without_checking():
    window = SlidingWindow(2, 60, FakeClock())
    window.record()
    window.record()
    assert window.blocked()


# --- password gates --------------------------------------------------------------------------


def test_failures_in_different_sessions_add_up_and_lock_every_session():
    limiter = SlidingWindow(4, 900, FakeClock())
    sessions = [{} for _ in range(5)]
    # Two wrong guesses from each of two sessions: each session still has attempts left...
    for state in sessions[:2]:
        assert na.try_unlock(state, "no", "secret", limiter) == na.WRONG or limiter.blocked()
        na.try_unlock(state, "no", "secret", limiter)
    assert limiter.blocked()
    # ...but now a brand-new session, even with the right password, is refused.
    assert na.try_unlock(sessions[4], "secret", "secret", limiter) == na.LOCKED_OUT
    assert not na.is_unlocked(sessions[4])


def test_the_lockout_clears_when_the_window_passes():
    clock = FakeClock()
    limiter = SlidingWindow(2, 900, clock)
    state: dict = {}
    na.try_unlock(state, "x", "secret", limiter)
    na.try_unlock({}, "x", "secret", limiter)
    assert na.try_unlock({}, "secret", "secret", limiter) == na.LOCKED_OUT
    clock.now = 901
    assert na.try_unlock({}, "secret", "secret", limiter) == na.OK


def test_two_gates_keep_separate_session_counters():
    state: dict = {}
    for _ in range(na.MAX_FAILED_ATTEMPTS):
        na.try_unlock(state, "no", "pw", unlocked_key="admin_ok", failed_key="admin_failed")
    assert na.attempts_left(state, "admin_failed") == 0
    assert na.attempts_left(state) == na.MAX_FAILED_ATTEMPTS  # the notes gate is untouched
    assert na.try_unlock(state, "pw", "pw") == na.OK


def test_a_non_ascii_password_is_just_wrong_not_an_error():
    assert na.try_unlock({}, "pässwörd", "password") == na.WRONG
    assert na.try_unlock({}, "password", "pässwörd") == na.WRONG
    assert na.try_unlock({}, "pässwörd", "pässwörd") == na.OK


def _admin_app():
    from soccer_predictor.dashboard.views import admin

    admin.render()


@pytest.fixture
def admin_gate(monkeypatch):
    from soccer_predictor.dashboard.views import admin

    monkeypatch.setattr(admin, "admin_password", lambda: "correct horse")
    monkeypatch.setattr(admin, "time", SimpleNamespace(sleep=lambda seconds: None))
    monkeypatch.setattr(na, "ADMIN_LIMITER", SlidingWindow(10, 900))


def test_admin_page_survives_odd_input_and_locks_a_session_after_five_wrong_passwords(admin_gate):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_admin_app, default_timeout=30).run()
    for guess in ("é", "ünïcödé", "x", "y", "z"):
        at.text_input[0].set_value(guess)
        at.button[0].click().run()
        assert not at.exception
    assert any("Too many wrong passwords" in e.value for e in at.error)
    # Locked out: on the next render the password box is gone, so even the right password gets nowhere.
    at.run()
    assert len(at.text_input) == 0 and any("Too many wrong passwords" in e.value for e in at.error)


def test_admin_page_unlocks_with_the_right_password(admin_gate):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_admin_app, default_timeout=30).run()
    at.text_input[0].set_value("correct horse")
    at.button[0].click().run()
    assert not at.exception and any(s.value == "Refresh live data" for s in at.subheader)


# --- news search budget ----------------------------------------------------------------------


def _rss(title="Headline"):
    return (
        '<rss><channel><item><title>%s - Src</title><link>https://example.com/a</link>'
        "<source>Src</source><pubDate>Mon, 12 Oct 2026 10:00:00 GMT</pubDate></item></channel></rss>" % title
    )


def test_uncached_searches_stop_at_the_budget_but_cached_ones_still_work(monkeypatch, tmp_path):
    monkeypatch.setattr(news, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(news, "_search_budget", SlidingWindow(2, 60, FakeClock()))
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(kwargs["params"]["q"])
        return SimpleNamespace(raise_for_status=lambda: None, text=_rss())

    monkeypatch.setattr(news.http_session, "get", fake_get)

    assert news.search_news("Premier League", "one") and news.search_news("Premier League", "two")
    assert news.search_news("Premier League", "three") == []  # over budget: no request made
    assert len(calls) == 2
    assert news.search_news("Premier League", "one")  # cached: free
    assert len(calls) == 2
    # The headline strip is not budgeted -- only user-typed searches are.
    assert news.fetch_news("Premier League")
    assert len(calls) == 3


def test_the_news_cache_is_capped(monkeypatch, tmp_path):
    monkeypatch.setattr(news, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(news, "MAX_CACHE_FILES", 5)
    monkeypatch.setattr(news.http_session, "get", lambda *a, **k: SimpleNamespace(raise_for_status=lambda: None, text=_rss()))
    for i in range(12):
        news.fetch_news(f"query {i}")
    assert len(list(tmp_path.glob("*.xml"))) == 5


def test_a_session_can_only_try_so_many_different_searches():
    from streamlit.testing.v1 import AppTest

    def app():
        from soccer_predictor.dashboard import components
        from soccer_predictor.ingest import news as n

        n.fetch_news = lambda *a, **k: []
        components.fetch_news = n.fetch_news
        components.search_news = lambda *a, **k: []
        components.render_news_section("Premier League", "t")

    at = AppTest.from_function(app, default_timeout=30).run()
    for i in range(35):
        at.text_input[0].set_value(f"query {i}").run()
    assert any("Search limit reached" in c.value for c in at.caption)


# --- note parser output ----------------------------------------------------------------------


def _parse_with(monkeypatch, tool_input):
    block = SimpleNamespace(type="tool_use", input=tool_input)
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(content=[block])))
    monkeypatch.setattr(note_parser, "anthropic_api_key", lambda: "key")
    monkeypatch.setattr(note_parser.anthropic, "Anthropic", lambda api_key: client)
    return note_parser.parse_note("anything", {1: "Arsenal"}, dt.date(2026, 10, 1))


def test_out_of_range_and_hostile_model_output_is_clamped_or_dropped(monkeypatch):
    result = _parse_with(
        monkeypatch,
        {
            "injury_entries": [
                {"player_name": "A" * 500, "team_name": "Arsenal", "position": "goalkeeper-of-doom", "importance_weight": 50,
                 "estimated_weeks_until_return": 1e9, "rationale": "r" * 1000},
                {"player_name": "NaN man", "team_name": "Arsenal", "position": "attack", "importance_weight": float("nan")},
                {"team_name": "no player name", "position": "attack", "importance_weight": 0.5},
                "not even a dict",
            ],
            "form_entries": [
                {"team_name": "Arsenal", "magnitude": -9, "affects": "everything", "fade_out_weeks": 1e12, "summary": "s" * 900},
                {"team_name": "Arsenal", "magnitude": "lots", "affects": "both", "fade_out_weeks": 2},
            ],
        },
    )

    [injury] = result.injuries
    assert len(injury.player_name) == note_parser.MAX_NAME_LENGTH and len(injury.rationale) == note_parser.MAX_TEXT_LENGTH
    assert injury.position == "attack" and injury.importance_weight == 1.0
    assert injury.expected_return_date == dt.date(2026, 10, 1) + dt.timedelta(weeks=note_parser.MAX_WEEKS)
    assert injury.team_id == 1

    [form] = result.form_notes
    assert form.magnitude == -1.0 and form.affects == "both" and len(form.summary) == note_parser.MAX_TEXT_LENGTH
    assert form.expires_on == dt.date(2026, 10, 1) + dt.timedelta(weeks=note_parser.MAX_WEEKS)


def test_sane_model_output_passes_through_unchanged(monkeypatch):
    result = _parse_with(
        monkeypatch,
        {
            "injury_entries": [
                {"player_name": "Saka", "team_name": "Arsenal", "position": "attack", "importance_weight": 0.8,
                 "estimated_weeks_until_return": 3, "rationale": "stated"}
            ],
            "form_entries": [],
        },
    )
    [injury] = result.injuries
    assert (injury.player_name, injury.importance_weight, injury.expected_return_date) == (
        "Saka", 0.8, dt.date(2026, 10, 1) + dt.timedelta(weeks=3)
    )


# --- streamlit config ------------------------------------------------------------------------


def test_visitors_are_not_shown_tracebacks():
    config = tomllib.loads((Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["client"]["showErrorDetails"] == "none"
