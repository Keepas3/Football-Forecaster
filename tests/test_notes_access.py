from __future__ import annotations

from soccer_predictor import config
from soccer_predictor.dashboard import notes_access as na


def test_correct_password_unlocks_and_resets_the_failure_count():
    state = {na.FAILED_ATTEMPTS_KEY: 3}
    assert na.try_unlock(state, "hunter2", "hunter2") == na.OK
    assert na.is_unlocked(state)
    assert state[na.FAILED_ATTEMPTS_KEY] == 0


def test_wrong_password_stays_locked_and_counts_the_attempt():
    state = {}
    assert na.try_unlock(state, "nope", "hunter2") == na.WRONG
    assert not na.is_unlocked(state)
    assert na.attempts_left(state) == na.MAX_FAILED_ATTEMPTS - 1


def test_a_session_is_locked_out_after_too_many_wrong_passwords_even_for_the_right_one():
    state = {}
    results = [na.try_unlock(state, "nope", "hunter2") for _ in range(na.MAX_FAILED_ATTEMPTS)]
    assert results[:-1] == [na.WRONG] * (na.MAX_FAILED_ATTEMPTS - 1)
    assert results[-1] == na.LOCKED_OUT

    assert na.try_unlock(state, "hunter2", "hunter2") == na.LOCKED_OUT
    assert not na.is_unlocked(state)


def test_non_ascii_passwords_work_and_do_not_raise():
    assert na.password_matches("pässwörd✓", "pässwörd✓")
    assert not na.password_matches("pässwörd", "pässwörd✓")


def test_password_comparison_is_exact():
    assert not na.password_matches("hunter2 ", "hunter2")
    assert not na.password_matches("HUNTER2", "hunter2")
    assert not na.password_matches("", "hunter2")


def test_lock_relocks_the_session():
    state = {}
    na.try_unlock(state, "pw", "pw")
    na.lock(state)
    assert not na.is_unlocked(state)


def test_note_length_cap():
    assert na.note_length_ok("x" * na.MAX_NOTE_LENGTH)
    assert not na.note_length_ok("x" * (na.MAX_NOTE_LENGTH + 1))


def test_parse_budget_runs_out():
    state = {}
    assert na.parses_left(state) == na.MAX_PARSES_PER_SESSION
    for _ in range(na.MAX_PARSES_PER_SESSION):
        assert na.parses_left(state) > 0
        na.record_parse(state)
    assert na.parses_left(state) == 0


def test_notes_password_is_unset_by_default_and_read_from_the_environment(monkeypatch):
    monkeypatch.delenv("NOTES_PASSWORD", raising=False)
    assert config.notes_password() is None
    monkeypatch.setenv("NOTES_PASSWORD", "")
    assert config.notes_password() is None  # blank means disabled, not "empty password"
    monkeypatch.setenv("NOTES_PASSWORD", "hunter2")
    assert config.notes_password() == "hunter2"
