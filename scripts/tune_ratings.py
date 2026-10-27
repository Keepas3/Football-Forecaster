"""Walk-forward test of a league's rating settings (memory length and smoothing).

Usage:
    uv run python scripts/tune_ratings.py LEAGUE_CODE [--min-season CODE] [--eval-from YYYY-MM-DD]
                                                      [--holdout-from YYYY-MM-DD]

Every candidate (rating_decay_xi, rating_ridge) pair predicts each month of the
tuning window using only the matches played before it; the pair with the lowest
ranked probability score (RPS) wins, and is then checked on a holdout window it
never saw. Read the numbers off into config/leagues.yaml next to the league's
rating_decay_xi / rating_ridge. See prediction/tuning.py.

--min-season drops matches from seasons before that code, to check whether the
oldest history helps. --eval-from defaults to the start of the fourth stored
season (earlier months have too little history to forecast from), --holdout-from
to the start of the league's browsable seasons (League.seasons[0]).
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.config import load_leagues  # noqa: E402
from soccer_predictor.model.time_weighting import DEFAULT_XI  # noqa: E402
from soccer_predictor.prediction.tuning import grid_search, paired_bootstrap, walk_forward_rps  # noqa: E402
from soccer_predictor.storage.db import init_db, session_scope  # noqa: E402
from soccer_predictor.storage.repository import matches_for_league  # noqa: E402


def _option(name: str) -> str | None:
    args = sys.argv[1:]
    return args[args.index(name) + 1] if name in args else None


def _season_start(matches, season: str) -> dt.date:
    return matches.loc[matches["season"] == season, "date"].min()


def main() -> None:
    init_db()
    leagues = load_leagues()
    codes = [a for a in sys.argv[1:] if a in leagues]
    if not codes:
        sys.exit("Usage: tune_ratings.py LEAGUE_CODE [--min-season CODE] [--eval-from DATE] [--holdout-from DATE]")
    league = leagues[codes[0]]

    with session_scope() as session:
        matches = matches_for_league(session, league.code)
    min_season = _option("--min-season")
    if min_season:
        matches = matches[matches["season"] >= min_season]
    seasons = sorted(matches["season"].unique())
    if len(seasons) < 5:
        sys.exit(f"Only {len(seasons)} season(s) stored for {league.code}; need at least 5 to tune and hold out.")

    eval_from = dt.date.fromisoformat(_option("--eval-from")) if _option("--eval-from") else _season_start(matches, seasons[3])
    holdout_from = (
        dt.date.fromisoformat(_option("--holdout-from"))
        if _option("--holdout-from")
        else _season_start(matches, league.seasons[0])
    )
    holdout_to = None
    tune_to = holdout_from - dt.timedelta(days=1)
    print(
        f"{league.code}: {len(matches)} matches, seasons {seasons[0]}..{seasons[-1]}; tuning {eval_from}..{tune_to}, "
        f"holdout from {holdout_from}"
    )

    results = grid_search(matches, league.code, eval_from, tune_to)
    base = results[0]["base_rps"]
    print(f"\nTuning window ({results[0]['n']} matches), no-skill guess RPS {base:.4f}. Best 8 of {len(results)}:")
    for entry in results[:8]:
        print(f"  xi={entry['xi']:<7} ridge={entry['ridge']:<5} RPS {entry['rps']:.4f}")
    default = next((e for e in results if e["xi"] == DEFAULT_XI and e["ridge"] == 0.0), None)
    if default:
        print(f"  default fit (xi={DEFAULT_XI}, ridge=0): RPS {default['rps']:.4f}")

    best = results[0]
    held_best = walk_forward_rps(matches, league.code, best["xi"], best["ridge"], holdout_from, holdout_to)
    held_default = walk_forward_rps(matches, league.code, DEFAULT_XI, 0.0, holdout_from, holdout_to)
    if held_best.empty:
        print("\nNo holdout matches to score.")
        return
    print(f"\nHoldout from {holdout_from} ({len(held_best)} matches, never used to choose):")
    print(f"  no-skill guess:                    RPS {held_best['base_rps'].mean():.4f}")
    print(f"  default fit (xi={DEFAULT_XI}, ridge=0):  RPS {held_default['rps'].mean():.4f}")
    print(f"  chosen (xi={best['xi']}, ridge={best['ridge']}):       RPS {held_best['rps'].mean():.4f}")
    for label, worse in (("default fit", held_default["rps"]), ("no-skill guess", held_best["base_rps"])):
        change, low, high = paired_bootstrap(held_best["rps"].to_numpy(), worse.to_numpy())
        print(f"  chosen vs {label}: RPS change {change:+.4f} (95% interval {low:+.4f} .. {high:+.4f})")


if __name__ == "__main__":
    main()
