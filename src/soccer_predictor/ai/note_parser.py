"""Parses freeform chat notes ("Saka is injured, ~3 weeks") into structured
data via the Anthropic API, using forced tool-use so the response is
guaranteed to match a fixed schema rather than freeform text we'd have to
hope is valid JSON.

Nothing here writes to the database or affects predictions directly --
prediction/service.py and storage/repository.py only ever see data the
dashboard's review-card flow has saved, never a raw parse result.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import anthropic
from rapidfuzz import fuzz, process

from soccer_predictor.config import anthropic_api_key, claude_model

TEAM_NAME_FUZZY_THRESHOLD = 85  # looser than ingest/team_mapper.py's 90 -- chat text is noisier

TOOL_NAME = "extract_player_and_team_notes"

_TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Records structured player-availability and team-form notes extracted "
        "from a freeform message about a soccer team or player."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "injury_entries": {
                "type": "array",
                "description": "One entry per player mentioned as injured, suspended, or otherwise unavailable.",
                "items": {
                    "type": "object",
                    "properties": {
                        "player_name": {"type": "string"},
                        "team_name": {
                            "type": "string",
                            "description": "The player's team. Prefer an exact match from the candidate team list provided; otherwise your best literal guess.",
                        },
                        "position": {
                            "type": "string",
                            "enum": ["attack", "defense"],
                            "description": "'defense' for defenders/goalkeepers, 'attack' for forwards/midfielders.",
                        },
                        "importance_weight": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "How much the team relies on this player, 0-1.",
                        },
                        "estimated_weeks_until_return": {
                            "type": ["number", "null"],
                            "description": "Weeks until expected return, if stated or reasonably inferable; null if unknown.",
                        },
                        "rationale": {"type": "string", "description": "One short sentence explaining the extraction."},
                    },
                    "required": ["player_name", "team_name", "position", "importance_weight", "rationale"],
                },
            },
            "form_entries": {
                "type": "array",
                "description": "One entry per team mentioned in a general team-form/morale/performance context (not a specific player availability note).",
                "items": {
                    "type": "object",
                    "properties": {
                        "team_name": {"type": "string"},
                        "magnitude": {
                            "type": "number",
                            "minimum": -1,
                            "maximum": 1,
                            "description": "Signed strength of the note: negative = bad form, positive = good form.",
                        },
                        "affects": {
                            "type": "string",
                            "enum": ["attack", "defense", "both"],
                            "description": "Which side of the team's play this note is about.",
                        },
                        "fade_out_weeks": {
                            "type": "number",
                            "description": "Always provide your best estimate (even if the user didn't say) of how many weeks this note should keep mattering before it's stale.",
                        },
                        "summary": {"type": "string", "description": "Short human-readable gloss of the note."},
                        "rationale": {"type": "string", "description": "One short sentence explaining the extraction."},
                    },
                    "required": ["team_name", "magnitude", "affects", "fade_out_weeks", "summary", "rationale"],
                },
            },
        },
        "required": ["injury_entries", "form_entries"],
    },
}


class MissingApiKey(Exception):
    pass


class NoteParsingError(Exception):
    """Wraps Anthropic SDK errors so callers only need to catch one type."""


@dataclass
class ParsedInjuryNote:
    player_name: str
    team_name_raw: str
    team_id: int | None  # None if unresolved/ambiguous -- never guessed
    position: str
    importance_weight: float
    expected_return_date: dt.date | None
    rationale: str


@dataclass
class ParsedFormNote:
    team_name_raw: str
    team_id: int | None
    magnitude: float
    affects: str
    expires_on: dt.date
    summary: str
    rationale: str


@dataclass
class ParsedNotes:
    injuries: list[ParsedInjuryNote]
    form_notes: list[ParsedFormNote]


MAX_NAME_LENGTH = 100
MAX_TEXT_LENGTH = 300
# No injury or form note is worth predicting two years out.
MAX_WEEKS = 104.0


def _clamp(value: float, low: float, high: float) -> float:
    if value != value:  # NaN
        raise ValueError("not a number")
    return max(low, min(high, value))


def _text(value, max_length: int) -> str:
    return str(value or "").strip()[:max_length]


def _resolve_team_name(raw_name: str, team_names: dict[int, str]) -> int | None:
    """Fuzzy-matches `raw_name` against known team names for this league.

    Never guesses below threshold -- same "log/refuse rather than guess"
    philosophy as ingest/team_mapper.py::resolve(), just returning None
    instead of raising, since here the caller shows a review-card picker
    instead of failing outright.
    """
    if not team_names:
        return None
    ids = list(team_names.keys())
    names = [team_names[i] for i in ids]
    match = process.extractOne(raw_name, names, scorer=fuzz.WRatio)
    if match is None:
        return None
    _, score, index = match
    if score < TEAM_NAME_FUZZY_THRESHOLD:
        return None
    return ids[index]


def parse_note(
    raw_text: str, team_names: dict[int, str], reference_date: dt.date
) -> ParsedNotes:
    api_key = anthropic_api_key()
    if not api_key:
        raise MissingApiKey(
            "ANTHROPIC_API_KEY not set -- copy .env.example to .env and add a key from "
            "https://console.anthropic.com/settings/keys"
        )

    candidate_names = ", ".join(sorted(team_names.values()))
    prompt = (
        "Extract structured player-availability and team-form notes from this message "
        f"about a soccer/football team or player:\n\n\"{raw_text}\"\n\n"
        f"Candidate team names in this league: {candidate_names}\n"
        f"Today's date is {reference_date.isoformat()}. "
        "Use the extract_player_and_team_notes tool to record what you find. "
        "If the message has nothing extractable, call the tool with empty arrays."
    )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=claude_model(),
            max_tokens=1024,
            tools=[_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise NoteParsingError(f"Anthropic API error: {exc}") from exc

    tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
    if not tool_use_blocks:
        raise NoteParsingError("Model response contained no tool_use block")

    return build_parsed_notes(tool_use_blocks[0].input, team_names, reference_date)


def build_parsed_notes(data: dict, team_names: dict[int, str], reference_date: dt.date) -> ParsedNotes:
    """Turns a model's extracted JSON (either provider's, same schema) into
    ParsedNotes.

    The model's output is only as trustworthy as the note it was given (a note
    can try to instruct it), so nothing from it is taken as-is: numbers are
    clamped to the ranges the schema promises, choices to their allowed
    values, text to a sane length, and an entry that can't be read is dropped.
    """
    injuries = []
    for entry in data.get("injury_entries") or []:
        try:
            weeks = entry.get("estimated_weeks_until_return")
            injuries.append(
                ParsedInjuryNote(
                    player_name=_text(entry["player_name"], MAX_NAME_LENGTH),
                    team_name_raw=_text(entry["team_name"], MAX_NAME_LENGTH),
                    team_id=_resolve_team_name(_text(entry["team_name"], MAX_NAME_LENGTH), team_names),
                    position=entry["position"] if entry.get("position") in ("attack", "defense") else "attack",
                    importance_weight=_clamp(float(entry["importance_weight"]), 0.0, 1.0),
                    expected_return_date=(
                        reference_date + dt.timedelta(weeks=_clamp(float(weeks), 0.0, MAX_WEEKS))
                        if weeks is not None
                        else None
                    ),
                    rationale=_text(entry.get("rationale"), MAX_TEXT_LENGTH),
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue

    form_notes = []
    for entry in data.get("form_entries") or []:
        try:
            form_notes.append(
                ParsedFormNote(
                    team_name_raw=_text(entry["team_name"], MAX_NAME_LENGTH),
                    team_id=_resolve_team_name(_text(entry["team_name"], MAX_NAME_LENGTH), team_names),
                    magnitude=_clamp(float(entry["magnitude"]), -1.0, 1.0),
                    affects=entry["affects"] if entry.get("affects") in ("attack", "defense", "both") else "both",
                    expires_on=reference_date
                    + dt.timedelta(weeks=_clamp(float(entry["fade_out_weeks"]), 0.0, MAX_WEEKS)),
                    summary=_text(entry.get("summary"), MAX_TEXT_LENGTH),
                    rationale=_text(entry.get("rationale"), MAX_TEXT_LENGTH),
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue

    return ParsedNotes(injuries=injuries, form_notes=form_notes)
