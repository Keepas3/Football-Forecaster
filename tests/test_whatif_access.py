"""Limits on the public AI what-if tab (dashboard/whatif_access.py)."""

from __future__ import annotations

from soccer_predictor.dashboard import whatif_access as wa
from soccer_predictor.limits import SlidingWindow


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _windows(per_minute=100, per_day=1000, clock=None):
    clock = clock or Clock()
    return {
        "minute": SlidingWindow(per_minute, 60, clock=clock),
        "day": SlidingWindow(per_day, 86400, clock=clock),
        "clock": clock,
    }


def test_a_normal_note_is_allowed_and_counted():
    state, w = {}, _windows()
    assert wa.try_ask(state, "Saka is out", **w) == wa.OK
    assert wa.asks_left(state) == wa.MAX_ASKS_PER_SESSION - 1


def test_blank_and_overlong_notes_are_refused_without_counting():
    state, w = {}, _windows()
    assert wa.try_ask(state, "   ", **w) == wa.EMPTY
    assert wa.try_ask(state, "x" * (wa.MAX_NOTE_LENGTH + 1), **w) == wa.TOO_LONG
    assert wa.try_ask(state, "x" * wa.MAX_NOTE_LENGTH, **w) == wa.OK
    assert wa.asks_left(state) == wa.MAX_ASKS_PER_SESSION - 1


def test_a_session_has_a_cap_and_a_pause_between_asks():
    state, w = {}, _windows()
    assert wa.try_ask(state, "one", **w) == wa.OK
    assert wa.try_ask(state, "two", **w) == wa.COOLDOWN
    for _ in range(wa.MAX_ASKS_PER_SESSION - 1):
        w["clock"].now += wa.COOLDOWN_SECONDS + 1
        assert wa.try_ask(state, "again", **w) == wa.OK
    w["clock"].now += wa.COOLDOWN_SECONDS + 1
    assert wa.try_ask(state, "one too many", **w) == wa.SESSION_LIMIT


def test_the_site_wide_minute_window_counts_every_visitor_and_recovers():
    w = _windows(per_minute=2)
    assert wa.try_ask({}, "a", **w) == wa.OK
    assert wa.try_ask({}, "b", **w) == wa.OK
    state = {}
    assert wa.try_ask(state, "c", **w) == wa.BUSY
    assert wa.asks_left(state) == wa.MAX_ASKS_PER_SESSION  # a refused ask costs nothing
    w["clock"].now += 61
    assert wa.try_ask(state, "c", **w) == wa.OK


def test_the_daily_window_stops_everyone_until_it_clears():
    w = _windows(per_day=1)
    assert wa.try_ask({}, "a", **w) == wa.OK
    assert wa.try_ask({}, "b", **w) == wa.DAY_LIMIT
    w["clock"].now += 86401
    assert wa.try_ask({}, "b", **w) == wa.OK


def test_every_refusal_has_a_message():
    for reason in (wa.TOO_LONG, wa.EMPTY, wa.SESSION_LIMIT, wa.COOLDOWN, wa.BUSY, wa.DAY_LIMIT):
        assert wa.MESSAGES[reason]
