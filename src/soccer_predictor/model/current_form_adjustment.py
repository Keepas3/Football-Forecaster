"""Applies a small, capped nudge to attack from each team's real current-
season attacking output (goals + xG-weighted assists, via Understat/
American Soccer Analysis -- see
ingest/player_importance.py::resolve_current_attack_strength), on top of
the base Dixon-Coles rating (which is fit purely from historical final
scores, averaged with time-decay over a team's ENTIRE match history -- see
model/dixon_coles.py, model/time_weighting.py).

Attack-only, deliberately: neither Understat nor ASA's player data carries
any defensive signal (no goals-conceded-while-on-pitch, tackles, saves) --
see ingest/player_importance.py's own module docstring. There is no
defensive counterpart to this function.

Unlike model/injury_adjustment.py (malus-only: an injury can only ever
weaken a team), this can push a rating either way -- a team currently
creating more/better chances than their long-run historical rating implies
gets a small boost; a team underperforming their own history gets a small
cut. Grounded in a real stat (unlike model/form_adjustment.py's hand-typed
notes), but still a deliberately small, capped, documented heuristic, not
a rigorously fit signal -- treat it the same way in the UI.
"""

from __future__ import annotations

CURRENT_FORM_IMPACT_CAP = 0.15  # max +/-15% swing on attack


def adjust_for_current_attack_form(base_attack: float, relative_xg_strength: float | None) -> float:
    """`relative_xg_strength` is centered on 1.0 (this team's current-season
    attacking rate vs. their league's current average) -- None (no
    coverage, insufficient sample, fetch failure) leaves `base_attack`
    untouched, same degrade-gracefully contract as every adjustment here.
    """
    if relative_xg_strength is None:
        return base_attack
    shift = max(-1.0, min(1.0, relative_xg_strength - 1.0)) * CURRENT_FORM_IMPACT_CAP
    return base_attack * (1.0 + shift)
