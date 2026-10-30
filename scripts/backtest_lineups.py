"""Does knowing the starting XI make the forecast better?  Backtests the lineup
adjustment on finished matches of the current season.

Usage:
    uv run python scripts/backtest_lineups.py [--cutoff YYYY-MM-DD] [--out FILE.json] [LEAGUE ...]

For each league, the base Dixon-Coles ratings are fitted ONLY on matches before
the cutoff (so the test matches are genuinely out of sample), then every later
match of the current season is forecast twice: with no lineup information, and
with each side's missing regulars (from ESPN's starters for that match against
the team's usual XI from this season's player stats) priced in at a grid of
strengths. Reports RPS and log loss, and the gain of the factors chosen on one
half of the matches when scored on the other half, with a bootstrap interval.

Caveats printed with the result: the usual XI comes from the season's stats up
to today, so it is known slightly with hindsight; and historical injuries are not
stored, so this compares lineup-vs-none, not lineup-vs-injury-feed.

Network: one ESPN summary per match (and scoreboard days), cached on disk, at the
client's courtesy rate; the first run for a league takes several minutes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_predictor.config import load_leagues  # noqa: E402
from soccer_predictor.ingest import espn_client, lineups  # noqa: E402
from soccer_predictor.model.dixon_coles import fit_league  # noqa: E402
from soccer_predictor.model.lineup_adjustment import lineup_absences, same_player  # noqa: E402
from soccer_predictor.model.time_weighting import DEFAULT_XI  # noqa: E402
from soccer_predictor.prediction import lineup_backtest as lb  # noqa: E402
from soccer_predictor.prediction.tracking import _outcome_sign  # noqa: E402
from soccer_predictor.storage.db import init_db, session_scope  # noqa: E402
from soccer_predictor.storage.repository import (  # noqa: E402
    espn_event_id_for_match,
    matches_for_league,
    teams_for_league,
)

DEFAULT_CUTOFF = dt.date(2026, 9, 1)
LEAGUES = ["EPL", "LALIGA", "BUNDESLIGA", "SERIEA", "LIGUE1", "MLS"]
FINISHED_CACHE_SECONDS = 90 * 24 * 3600  # a played match never changes


def collect(league_code: str, cutoff: dt.date) -> tuple[list[lb.Observation], dict]:
    league = load_leagues()[league_code]
    diagnostics = {"matches": 0, "no_event": 0, "no_lineup": 0, "no_squad": 0, "xi_players": 0, "xi_matched": 0, "absences": [], "unresolved": []}
    observations: list[lb.Observation] = []
    with session_scope() as session:
        matches = matches_for_league(session, league_code)
        names = teams_for_league(session, league_code)
        train = matches[matches.date < cutoff]
        test = matches[(matches.season == league.seasons[-1]) & (matches.date >= cutoff)].sort_values("date")
        params = fit_league(
            train, league_code, xi=league.rating_decay_xi or DEFAULT_XI, ridge=league.rating_ridge
        )
        squads: dict[int, list] = {}
        print(f"{league_code}: fitted on {len(train)} matches before {cutoff}; {len(test)} to test", flush=True)
        for row in test.itertuples(index=False):
            diagnostics["matches"] += 1
            home_name, away_name = names[row.home_team_id], names[row.away_team_id]
            event_id = espn_event_id_for_match(session, league_code, row.date, row.home_team_id, row.away_team_id)
            if not event_id:
                found = espn_client.find_event_by_names(
                    league.espn_league_slug, row.date, home_name, away_name, cache_ttl_seconds=FINISHED_CACHE_SECONDS
                )
                event_id = found[0] if found else None
            if not event_id:
                diagnostics["no_event"] += 1
                diagnostics["unresolved"].append(f"{row.date} {home_name} v {away_name}")
                continue
            detail = espn_client.fetch_match_detail(
                league.espn_league_slug, event_id, state="post", cache_ttl_seconds=FINISHED_CACHE_SECONDS
            )
            xi = lineups.confirmed_xi(detail, home_name, away_name)
            if xi is None:
                diagnostics["no_lineup"] += 1
                continue
            for team_id, name in ((row.home_team_id, home_name), (row.away_team_id, away_name)):
                if team_id not in squads:
                    squads[team_id] = lineups.expected_squad(name, league)
            home_squad, away_squad = squads[row.home_team_id], squads[row.away_team_id]
            if not home_squad or not away_squad:
                diagnostics["no_squad"] += 1
                continue
            for squad, xi_names in ((home_squad, xi.home), (away_squad, xi.away)):
                diagnostics["xi_players"] += len(xi_names)
                diagnostics["xi_matched"] += sum(any(same_player(p.name, n) for p in squad) for n in xi_names)
            home_absences, away_absences = lineup_absences(home_squad, xi.home), lineup_absences(away_squad, xi.away)
            diagnostics["absences"] += [len(home_absences), len(away_absences)]
            observations.append(
                lb.Observation(
                    league=league_code,
                    date=row.date,
                    base_home_attack=params.attack[row.home_team_id],
                    base_home_defense=params.defense[row.home_team_id],
                    base_away_attack=params.attack[row.away_team_id],
                    base_away_defense=params.defense[row.away_team_id],
                    home_advantage=params.home_advantage,
                    rho=params.rho,
                    home_absences=home_absences,
                    away_absences=away_absences,
                    actual=_outcome_sign(row.home_goals, row.away_goals),
                )
            )
            if len(observations) % 25 == 0:
                print(f"  {len(observations)} matches with lineups so far", flush=True)
    return observations, diagnostics


def report(observations: list[lb.Observation], diagnostics: dict[str, dict]) -> dict:
    scores = lb.score_grid(observations)
    base_rps, base_log = scores[lb.NO_LINEUP]
    best = lb.best_factors(scores)
    cv_rps, cv_log, chosen = lb.cross_validated_scores(scores)
    rps_gain = lb.paired_bootstrap(base_rps, cv_rps)
    log_gain = lb.paired_bootstrap(base_log, cv_log)

    print(f"\n=== {len(observations)} matches with a lineup on both sides ===")
    for code, d in diagnostics.items():
        coverage = d["xi_matched"] / d["xi_players"] if d["xi_players"] else float("nan")
        mean_absent = float(np.mean(d["absences"])) if d["absences"] else float("nan")
        print(
            f"{code:11} tested {d['matches']:3}  no ESPN event {d['no_event']:3}  no lineup {d['no_lineup']:3}  "
            f"no squad {d['no_squad']:3}  XI names found in squad {coverage:.0%}  absences per side {mean_absent:.2f}"
        )
    for code, d in diagnostics.items():
        for line in d["unresolved"]:
            print(f"  no ESPN event found: {code} {line}")
    print(f"\nno lineup information : RPS {base_rps.mean():.4f}  log loss {base_log.mean():.4f}")
    print(f"best pair on all data : attack {best[0]}, defense {best[1]} -> RPS {scores[best][0].mean():.4f}  log loss {scores[best][1].mean():.4f}")
    print(f"pairs chosen per fold : {chosen}")
    print(f"out-of-fold change    : RPS {rps_gain[0]:+.4f} (95% CI {rps_gain[1]:+.4f}..{rps_gain[2]:+.4f})   log loss {log_gain[0]:+.4f} (95% CI {log_gain[1]:+.4f}..{log_gain[2]:+.4f})   (negative = better)")
    print("\nRPS by (attack factor, defense factor); lower is better:")
    print("        " + "".join(f"d={d:<7}" for d in lb.DEFENSE_FACTORS))
    for a in lb.ATTACK_FACTORS:
        print(f"a={a:<5} " + "".join(f"{scores[(a, d)][0].mean():<9.4f}" for d in lb.DEFENSE_FACTORS))
    by_league = {}
    for code in diagnostics:
        rows = [i for i, o in enumerate(observations) if o.league == code]
        if rows:
            by_league[code] = {
                "n": len(rows),
                "base_rps": float(base_rps[rows].mean()),
                "best_rps": float(scores[best][0][rows].mean()),
            }
    print("\nper league at the best pair:", json.dumps(by_league, indent=None))
    return {
        "n": len(observations),
        "base_rps": float(base_rps.mean()),
        "base_log_loss": float(base_log.mean()),
        "best": list(best),
        "best_rps": float(scores[best][0].mean()),
        "folds": [list(c) for c in chosen],
        "out_of_fold_rps_change": list(rps_gain),
        "out_of_fold_log_loss_change": list(log_gain),
        "by_league": by_league,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("leagues", nargs="*", default=LEAGUES)
    parser.add_argument("--cutoff", type=dt.date.fromisoformat, default=DEFAULT_CUTOFF)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    init_db()
    observations, diagnostics = [], {}
    for code in args.leagues:
        found, diagnostics[code] = collect(code, args.cutoff)
        observations += found
    if len(observations) < 40:
        sys.exit(f"Only {len(observations)} usable matches -- too few to say anything.")
    result = report(observations, diagnostics)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
