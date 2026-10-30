"""Locks in the final forecast of every match that is about to kick off.

Usage:
    uv run python scripts/lock_lineups.py [LEAGUE_CODE ...]

For each league with a trained model, matches kicking off within the next hour and
a half are looked up on ESPN; once a match's starting XIs are published the forecast
is made from them and locked in, and a match with none by twenty minutes before
kickoff is locked on its early forecast instead (see
prediction.tracking.lock_due_predictions). Safe to run as often as you like:
matches already locked are left alone, and with nothing due it does nothing.

The lock-lineups GitHub workflow runs this every time it is pinged (about every ten
minutes, see the README); refresh_live_data.py also calls the same function, as a
backstop.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.config import load_leagues  # noqa: E402
from soccer_predictor.prediction.service import load_latest_params  # noqa: E402
from soccer_predictor.prediction.tracking import lock_due_predictions  # noqa: E402
from soccer_predictor.storage.db import init_db, session_scope  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    init_db()
    leagues = load_leagues()
    requested = sys.argv[1:] or list(leagues)

    total = 0
    for code in requested:
        league = leagues[code]
        with session_scope() as session:
            params = load_latest_params(session, league.code)
            if params is None:
                continue
            locked = lock_due_predictions(session, league, params)
        if locked:
            print(f"{league.name} ({league.code}): {locked} forecast(s) locked in")
        total += locked
    print(f"{total} forecast(s) locked in")


if __name__ == "__main__":
    main()
