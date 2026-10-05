from __future__ import annotations

from soccer_predictor.dashboard.chat_notes import add_reply, delete_exchange, new_message, replace_note_text


def _exchange(history, cards, text, n_cards=1):
    note = new_message("user", text)
    history.append(note)
    add_reply(history, note["id"], f"reply to {text}")
    for _ in range(n_cards):
        cards.append({"uid": f"{note['id']}-{len(cards)}", "kind": "injury", "source_id": note["id"]})
    return note


def test_delete_removes_the_note_its_reply_and_its_unsaved_cards_only():
    history, cards = [], []
    first = _exchange(history, cards, "Saka is injured", n_cards=2)
    second = _exchange(history, cards, "Arsenal are flat")

    delete_exchange(history, cards, first["id"])

    assert [m["text"] for m in history] == ["Arsenal are flat", "reply to Arsenal are flat"]
    assert all(c["source_id"] == second["id"] for c in cards) and len(cards) == 1


def test_delete_of_an_unknown_id_changes_nothing():
    history, cards = [], []
    _exchange(history, cards, "note")
    delete_exchange(history, cards, "nope")
    assert len(history) == 2 and len(cards) == 1


def test_edit_rewrites_the_note_and_clears_the_stale_reply_and_cards():
    history, cards = [], []
    note = _exchange(history, cards, "Saka is injured", n_cards=2)

    assert replace_note_text(history, cards, note["id"], "Saka is injured for 6 weeks") is True

    assert [(m["role"], m["text"]) for m in history] == [("user", "Saka is injured for 6 weeks")]
    assert history[0]["id"] == note["id"]  # same note, so its menu keeps working
    assert cards == []


def test_edit_leaves_other_notes_alone():
    history, cards = [], []
    _exchange(history, cards, "first")
    second = _exchange(history, cards, "second")

    replace_note_text(history, cards, second["id"], "second, edited")

    assert [m["text"] for m in history] == ["first", "reply to first", "second, edited"]
    assert len(cards) == 1  # the first note's card survives


def test_edit_only_applies_to_user_notes_that_exist():
    history, cards = [], []
    note = _exchange(history, cards, "note")
    reply = history[1]
    assert replace_note_text(history, cards, reply["id"], "tampered") is False
    assert replace_note_text(history, cards, "nope", "x") is False
    assert [m["text"] for m in history] == ["note", "reply to note"]
    assert len(cards) == 1 and note["text"] == "note"


def test_a_reply_goes_right_after_its_note_not_at_the_end():
    history, cards = [], []
    first = _exchange(history, cards, "first")
    _exchange(history, cards, "second")

    replace_note_text(history, cards, first["id"], "first, edited")
    add_reply(history, first["id"], "new reply")

    assert [m["text"] for m in history] == ["first, edited", "new reply", "second", "reply to second"]
