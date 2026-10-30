"""Adjusts a team's fitted attack/defense strength for current injuries.

Each injury has an importance_weight in [0, 1]; injured attackers pull down
attack, injured defenders/keeper pull down defense (recall: lower `defense`
value = stronger defensively, since it's a goals-conceded multiplier -- so an
injury *raises* it).

The factors are the ones the lineup backtest chose (scripts/backtest_lineups.py,
see model/lineup_adjustment.py): on 434 real matches a missing regular costs about
0.3 of his weight in attack (0.5 and above scored worse), while the data cannot
place the defense factor -- anything from 0 to 0.75 scored within 0.001. Historical
injuries are not stored, so the backtest prices real absences from the starting XI,
not the injury feed itself; the feed is assumed to behave like it. The floor and
ceiling are not fitted: they only stop a long feed (a dozen players listed at
once) from rewriting a team's rating.

Both factors used to be 0.5, with a 0.5 floor and a 2.0 ceiling. That let a list
of six injured defenders double a side's goals conceded, which no backtest
supported.
"""

from __future__ import annotations

from dataclasses import dataclass

ATTACK_IMPACT_FACTOR = 0.3  # how much importance_weight translates to a rating swing
DEFENSE_IMPACT_FACTOR = 0.1
MIN_MULTIPLIER = 0.7  # floor: no combination of injuries can swing a rating past this
MAX_MULTIPLIER = 1.3  # symmetric ceiling for the defense (goals-conceded) multiplier


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
