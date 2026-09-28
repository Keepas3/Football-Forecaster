"""Pure math for turning a player's own stats into an injury importance_weight
(model/injury_adjustment.py's per-player [0, 1] multiplier input) -- so
losing a team's top scorer hurts a prediction more than losing a rarely-used
squad player, instead of every API-sourced injury counting identically.

Both functions return None (never 0.0) when there isn't enough data to say
anything -- a real rostered player is never truly zero impact, so "we don't
know" and "we know they barely matter" must stay distinguishable. Callers
fall back to a flat default in the None case (see ingest/player_importance.py).

Not empirically validated -- same "tunable heuristic" caveat as
model/injury_adjustment.py's own ATTACK_IMPACT_FACTOR/DEFENSE_IMPACT_FACTOR;
free-tier data is too sparse to backtest this properly.
"""

from __future__ import annotations

GOAL_WEIGHT = 1.0
ASSIST_WEIGHT = 0.7  # goals valued above assists, standard "goal involvement" convention

# Calibration constant: a focal-point attacker responsible for ~35-40% of a
# team's combined weighted goal contributions maps to ~0.9-1.0 before
# clipping (i.e. roughly "irreplaceable").
ATTACK_SHARE_SCALE = 2.5

MIN_IMPORTANCE_WEIGHT = 0.05  # a real rostered player is never truly zero impact
MAX_IMPORTANCE_WEIGHT = 0.95  # leaves headroom above any computed value for a manual override

RATING_NUDGE_CAP = 0.3  # max +/-30% swing on the minutes-share defense estimate


def goal_contribution_value(goals: float, assists: float) -> float:
    return goals * GOAL_WEIGHT + assists * ASSIST_WEIGHT


def compute_attack_importance(
    goals: float | None, assists: float | None, team_goal_contribution_total: float
) -> float | None:
    """Share of the team's total (goals + weighted assists) this one player
    accounts for, scaled and clipped into the same [0, 1]-ish range manual/
    chat-entered weights already use.
    """
    if team_goal_contribution_total <= 0:
        return None
    contribution = goal_contribution_value(goals or 0, assists or 0)
    share = contribution / team_goal_contribution_total
    return max(MIN_IMPORTANCE_WEIGHT, min(share * ATTACK_SHARE_SCALE, MAX_IMPORTANCE_WEIGHT))


def compute_defense_importance(
    minutes: float | None, max_minutes_on_roster: float, rating: float | None
) -> float | None:
    """No defensive +/- or goals-prevented data exists on the free tier, so
    this proxies "how key is this defender/keeper" with minutes-share of the
    roster's most-used player (an undisputed starter vs. squad depth), with
    a small optional nudge from their average match rating when known.
    """
    if max_minutes_on_roster <= 0:
        return None
    minutes_share = min((minutes or 0) / max_minutes_on_roster, 1.0)
    rating_nudge = 0.0
    if rating is not None:
        rating_nudge = max(-1.0, min((rating - 6.0) / 2.0, 1.0)) * RATING_NUDGE_CAP
    return max(MIN_IMPORTANCE_WEIGHT, min(minutes_share * (1 + rating_nudge), MAX_IMPORTANCE_WEIGHT))
