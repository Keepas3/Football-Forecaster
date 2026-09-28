"""Adjusts a team's fitted attack/defense strength for current injuries.

This is a tunable heuristic for v1, not empirically validated -- free-tier
injury data is too sparse to backtest the adjustment's magnitude (see the
plan's open risks). Each injury has an importance_weight in [0, 1]; injured
attackers pull down attack, injured defenders/keeper pull down defense
(recall: lower `defense` value = stronger defensively, since it's a
goals-conceded multiplier -- so an injury *raises* it).
"""

from __future__ import annotations

from dataclasses import dataclass

ATTACK_IMPACT_FACTOR = 0.5  # how much importance_weight translates to a rating swing
DEFENSE_IMPACT_FACTOR = 0.5
MIN_MULTIPLIER = 0.5  # floor: no combination of injuries can swing a rating past this
MAX_MULTIPLIER = 2.0  # symmetric ceiling for the defense (goals-conceded) multiplier


@dataclass
class InjuryEntry:
    player_name: str
    position: str  # "attack" | "defense"
    importance_weight: float


def adjust_strength(
    base_attack: float, base_defense: float, injuries: list[InjuryEntry]
) -> tuple[float, float]:
    attack_hit = sum(
        i.importance_weight for i in injuries if i.position == "attack"
    ) * ATTACK_IMPACT_FACTOR
    defense_hit = sum(
        i.importance_weight for i in injuries if i.position == "defense"
    ) * DEFENSE_IMPACT_FACTOR

    attack_multiplier = max(1.0 - attack_hit, MIN_MULTIPLIER)
    defense_multiplier = min(1.0 + defense_hit, MAX_MULTIPLIER)

    return base_attack * attack_multiplier, base_defense * defense_multiplier
