"""Applies a small, capped nudge to attack/defense from free-text "form" notes.

This is strictly more speculative than model/injury_adjustment.py's already-
unvalidated heuristic: injury_adjustment at least rests on a real mechanism
(a specific player is missing). Here, even the *existence* of an effect from
something like "team's been flat since the manager change" is a guess, let
alone its size. Treat this as a deliberately small, capped nudge -- not a
modeled signal -- and say so wherever it's shown in the UI, not just here.
"""

from __future__ import annotations

from dataclasses import dataclass

FORM_IMPACT_CAP = 0.15  # max +/-15% swing on attack or defense, combined across all notes


@dataclass
class FormNoteEntry:
    magnitude: float  # signed, -1.0..1.0; negative = bad form, positive = good form
    affects: str  # "attack" | "defense" | "both"


def adjust_for_form(
    base_attack: float, base_defense: float, notes: list[FormNoteEntry]
) -> tuple[float, float]:
    attack_shift = sum(n.magnitude for n in notes if n.affects in ("attack", "both"))
    defense_shift = sum(n.magnitude for n in notes if n.affects in ("defense", "both"))

    attack_shift = max(-1.0, min(1.0, attack_shift)) * FORM_IMPACT_CAP
    defense_shift = max(-1.0, min(1.0, defense_shift)) * FORM_IMPACT_CAP

    # Sign inversion is deliberate: defense is a goals-conceded multiplier
    # (lower = better), so bad form (negative magnitude) must RAISE it --
    # the same inversion injury_adjustment.py has to get right for defenders.
    attack_multiplier = 1.0 + attack_shift
    defense_multiplier = 1.0 - defense_shift

    return base_attack * attack_multiplier, base_defense * defense_multiplier
