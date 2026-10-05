"""The Notes tab's chat history, as plain list operations (no Streamlit) so
editing and deleting a message are unit-testable.

A history is a list of message dicts: {"id", "role", "text", "reply_to"}. A
user message is a note the person typed; the assistant message answering it
carries `reply_to` = that note's id. The unsaved review cards the parser
produced from a note (`pending_cards`) carry `source_id` = the same id.
Already-saved notes live in the database, not here, and have their own Delete
buttons under "Active chat-sourced notes".
"""

from __future__ import annotations

from uuid import uuid4


def new_message(role: str, text: str, reply_to: str | None = None) -> dict:
    return {"id": uuid4().hex, "role": role, "text": text, "reply_to": reply_to}


def _drop_replies_and_cards(history: list[dict], cards: list[dict], message_id: str) -> None:
    history[:] = [m for m in history if m.get("reply_to") != message_id]
    cards[:] = [c for c in cards if c.get("source_id") != message_id]


def delete_exchange(history: list[dict], cards: list[dict], message_id: str) -> None:
    """Removes a note, the assistant's reply to it, and any review cards it
    produced that haven't been saved or discarded yet (in place)."""
    _drop_replies_and_cards(history, cards, message_id)
    history[:] = [m for m in history if m.get("id") != message_id]


def replace_note_text(history: list[dict], cards: list[dict], message_id: str, new_text: str) -> bool:
    """Rewrites a user note in place, discarding the now-stale reply and
    unsaved review cards so the edited text can be parsed afresh. False (and
    no change) if there is no such user message."""
    message = next((m for m in history if m.get("id") == message_id and m.get("role") == "user"), None)
    if message is None:
        return False
    _drop_replies_and_cards(history, cards, message_id)
    message["text"] = new_text
    return True


def add_reply(history: list[dict], message_id: str, text: str) -> None:
    """Puts the assistant's reply directly after the note it answers -- not at
    the end, which matters when an older note is edited and re-parsed."""
    index = next((i for i, m in enumerate(history) if m.get("id") == message_id), len(history) - 1)
    history.insert(index + 1, new_message("assistant", text, reply_to=message_id))
