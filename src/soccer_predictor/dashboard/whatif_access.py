"""How much the public "AI what-if" tab may be used. Unlike the password-gated
Notes chat, anyone can use it, and every note goes to Google's Gemini API on
the app owner's free-tier key -- so the free quota is the thing to protect:
a cap per visitor session, a short pause between asks, and two server-wide
windows (per minute, per day) that count every visitor together, set well
under what the free tier allows. Google's own 429 is the real backstop (the
counters live in this process and reset when it restarts); see
ai/whatif_parser.py.

All functions take the session state as a plain mapping (Streamlit's
st.session_state, or a dict in tests) -- no Streamlit imports here.
"""

from __future__ import annotations

import time
from collections.abc import Callable, MutableMapping

from soccer_predictor.limits import SlidingWindow

# One note is a sentence or two; this also keeps each request's tokens small.
MAX_NOTE_LENGTH = 300
# Asks one visitor session may make (a fresh session starts over, which is
# why the server-wide windows below exist).
MAX_ASKS_PER_SESSION = 6
# Seconds a session must wait between asks.
COOLDOWN_SECONDS = 4.0

# Server-wide, all visitors together. Tune these to the limits shown for the
# key's project in Google AI Studio (aistudio.google.com/rate-limit).
MAX_ASKS_PER_MINUTE = 8
MAX_ASKS_PER_DAY = 250
MINUTE_LIMITER = SlidingWindow(MAX_ASKS_PER_MINUTE, 60)
DAY_LIMITER = SlidingWindow(MAX_ASKS_PER_DAY, 24 * 60 * 60)

ASK_COUNT_KEY = "whatif_ask_count"
LAST_ASK_KEY = "whatif_last_ask"

OK = "ok"
TOO_LONG = "too_long"
EMPTY = "empty"
SESSION_LIMIT = "session_limit"
COOLDOWN = "cooldown"
BUSY = "busy"  # the per-minute window is full
DAY_LIMIT = "day_limit"

MESSAGES = {
    TOO_LONG: f"Keep the note under {MAX_NOTE_LENGTH} characters - a sentence or two is plenty.",
    EMPTY: "Type a note first.",
    SESSION_LIMIT: f"That's the limit of {MAX_ASKS_PER_SESSION} AI asks per visit. Reload the page to start a new visit.",
    COOLDOWN: "One moment - wait a few seconds between asks.",
    BUSY: "Lots of people are trying this right now. Try again in a minute.",
    DAY_LIMIT: "The free AI allowance for today has been used up. Try again tomorrow.",
}


def asks_left(state: MutableMapping) -> int:
    return max(0, MAX_ASKS_PER_SESSION - state.get(ASK_COUNT_KEY, 0))


def try_ask(
    state: MutableMapping,
    note: str,
    *,
    minute: SlidingWindow | None = None,
    day: SlidingWindow | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """OK -- and the ask is counted against the session and both server-wide
    windows (the module's own unless given) -- or the reason it was refused
    (nothing is counted then)."""
    minute = MINUTE_LIMITER if minute is None else minute
    day = DAY_LIMITER if day is None else day
    if not note.strip():
        return EMPTY
    if len(note) > MAX_NOTE_LENGTH:
        return TOO_LONG
    if asks_left(state) == 0:
        return SESSION_LIMIT
    now = clock()
    last = state.get(LAST_ASK_KEY)
    if last is not None and now - last < COOLDOWN_SECONDS:
        return COOLDOWN
    if day.blocked():
        return DAY_LIMIT
    if minute.blocked():
        return BUSY
    day.record()
    minute.record()
    state[ASK_COUNT_KEY] = state.get(ASK_COUNT_KEY, 0) + 1
    state[LAST_ASK_KEY] = now
    return OK
