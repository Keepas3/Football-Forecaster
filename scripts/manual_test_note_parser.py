"""Manual smoke test for ai/note_parser.py against the real Anthropic API.

Not a pytest test (needs a live ANTHROPIC_API_KEY and costs a few cents) --
run by hand to eyeball extraction quality against realistic notes.

Usage:
    uv run python scripts/manual_test_note_parser.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.ai.note_parser import parse_note  # noqa: E402
from soccer_predictor.storage.db import init_db, session_scope  # noqa: E402
from soccer_predictor.storage.repository import teams_for_league  # noqa: E402

EXAMPLE_NOTES = [
    "Saka is injured, hamstring, estimate ~3 weeks out",
    "Saka and Havertz are both injured for Arsenal",
    "Arsenal have been flat since the manager change",
    "Man Utd's keeper is suspended for the next 2 games",
]


def main() -> None:
    init_db()
    with session_scope() as session:
        team_names = teams_for_league(session, "EPL")

    if not team_names:
        print("No EPL teams in the DB -- run scripts/fetch_historical_data.py EPL first.")
        return

    today = dt.date.today()
    for note in EXAMPLE_NOTES:
        print(f"\n--- {note!r} ---")
        result = parse_note(note, team_names, today)
        for injury in result.injuries:
            print(
                f"  INJURY: {injury.player_name} ({injury.team_name_raw} -> "
                f"team_id={injury.team_id}) position={injury.position} "
                f"weight={injury.importance_weight:.2f} "
                f"return={injury.expected_return_date} -- {injury.rationale}"
            )
        for form in result.form_notes:
            print(
                f"  FORM: {form.team_name_raw} -> team_id={form.team_id} "
                f"magnitude={form.magnitude:+.2f} affects={form.affects} "
                f"expires={form.expires_on} -- {form.summary}"
            )


if __name__ == "__main__":
    main()
