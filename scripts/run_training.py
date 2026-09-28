"""Fits Dixon-Coles for one or more leagues and persists the result.

Usage:
    uv run python scripts/run_training.py [LEAGUE_CODE ...]

With no arguments, trains every league in config/leagues.yaml. Prints a
team strength leaderboard so you can sanity-check the fit (does the
strongest team have the best net attack-minus-defense?).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.config import load_leagues  # noqa: E402
from soccer_predictor.prediction.training import train_league  # noqa: E402
from soccer_predictor.storage.db import init_db, session_scope  # noqa: E402
from soccer_predictor.storage.repository import teams_for_league  # noqa: E402


def main() -> None:
    init_db()
    leagues = load_leagues()
    # Leagues with no historical match source (see League.supports_predictions)
    # are silently skipped when no args are given, but still explained if
    # explicitly requested by code -- never trained either way.
    requested = sys.argv[1:] or [c for c, l in leagues.items() if l.supports_predictions]

    for code in requested:
        league = leagues[code]
        if not league.supports_predictions:
            print(
                f"\nSkipping {league.name} ({league.code}): no historical match data source "
                "configured (see League.supports_predictions) -- this competition never gets a "
                "trained model."
            )
            continue
        print(f"\nTraining {league.name} ({league.code})...")
        with session_scope() as session:
            params = train_league(session, league.code)
            team_names = teams_for_league(session, league.code)

        print(f"  home_advantage={params.home_advantage:.3f}  rho={params.rho:.3f}  n_matches={params.n_matches}")
        print("  Leaderboard (net strength = attack - defense, higher is better):")
        ranked = sorted(
            params.attack.keys(),
            key=lambda tid: params.attack[tid] - params.defense[tid],
            reverse=True,
        )
        for tid in ranked:
            name = team_names.get(tid, f"team#{tid}")
            print(
                f"    {name:<20} attack={params.attack[tid]:.2f}  defense={params.defense[tid]:.2f}"
            )


if __name__ == "__main__":
    main()
