"""Who may use the Notes chat, and how much. Every note typed into the chat
is sent to the Anthropic API on the app owner's key, so on a public site the
chat must not be open to everyone: it's locked behind NOTES_PASSWORD (see
config.notes_password), with caps on what one unlocked session can spend.
Visitors without the password can still read the active notes.

All functions take the session state as a plain mapping (Streamlit's
st.session_state, or a dict in tests) -- no Streamlit imports here.
"""

from __future__ import annotations

import secrets
from collections.abc import MutableMapping

UNLOCKED_KEY = "notes_unlocked"
FAILED_ATTEMPTS_KEY = "notes_failed_attempts"
PARSE_COUNT_KEY = "notes_parse_count"

# A session is locked out after this many wrong passwords. It can't stop
# someone opening fresh sessions, so it only slows guessing -- the real
# protection is a long password.
MAX_FAILED_ATTEMPTS = 5
# One note is a sentence or two; this keeps a single request's token cost small.
MAX_NOTE_LENGTH = 1000
# Parses (new notes + edits re-parsed) one unlocked session may run -- a
# ceiling on spend if the password is shared or a session is left open.
MAX_PARSES_PER_SESSION = 40

OK = "ok"
WRONG = "wrong"
LOCKED_OUT = "locked_out"


def password_matches(entered: str, expected: str) -> bool:
    # Constant-time, and on bytes so a non-ASCII password can't raise.
    return secrets.compare_digest(entered.encode("utf-8"), expected.encode("utf-8"))


def is_unlocked(state: MutableMapping) -> bool:
    return bool(state.get(UNLOCKED_KEY))


def attempts_left(state: MutableMapping) -> int:
    return max(0, MAX_FAILED_ATTEMPTS - state.get(FAILED_ATTEMPTS_KEY, 0))


def try_unlock(state: MutableMapping, entered: str, expected: str) -> str:
    """OK (and the session is unlocked), WRONG, or LOCKED_OUT once the
    session has used up its attempts -- after which even the right password
    is refused."""
    if attempts_left(state) == 0:
        return LOCKED_OUT
    if password_matches(entered, expected):
        state[UNLOCKED_KEY] = True
        state[FAILED_ATTEMPTS_KEY] = 0
        return OK
    state[FAILED_ATTEMPTS_KEY] = state.get(FAILED_ATTEMPTS_KEY, 0) + 1
    return LOCKED_OUT if attempts_left(state) == 0 else WRONG


def lock(state: MutableMapping) -> None:
    state[UNLOCKED_KEY] = False


def note_length_ok(text: str) -> bool:
    return len(text) <= MAX_NOTE_LENGTH


def parses_left(state: MutableMapping) -> int:
    return max(0, MAX_PARSES_PER_SESSION - state.get(PARSE_COUNT_KEY, 0))


def record_parse(state: MutableMapping) -> None:
    state[PARSE_COUNT_KEY] = state.get(PARSE_COUNT_KEY, 0) + 1
