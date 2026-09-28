from __future__ import annotations

from soccer_predictor.model.injury_adjustment import (
    MAX_MULTIPLIER,
    MIN_MULTIPLIER,
    InjuryEntry,
    adjust_strength,
)


def test_no_injuries_leaves_strength_unchanged():
    attack, defense = adjust_strength(1.2, 0.9, injuries=[])
    assert attack == 1.2
    assert defense == 0.9


def test_key_striker_injury_reduces_attack_only():
    injuries = [InjuryEntry(player_name="Star Striker", position="attack", importance_weight=0.8)]
    attack, defense = adjust_strength(1.2, 0.9, injuries)

    assert attack < 1.2
    assert defense == 0.9


def test_key_defender_injury_increases_defense_value_only():
    # Recall: defense is a goals-conceded multiplier, so losing a key
    # defender makes the team WORSE defensively, i.e. a HIGHER value.
    injuries = [InjuryEntry(player_name="Star Defender", position="defense", importance_weight=0.8)]
    attack, defense = adjust_strength(1.2, 0.9, injuries)

    assert attack == 1.2
    assert defense > 0.9


def test_multiple_severe_injuries_are_clipped_to_floor():
    injuries = [
        InjuryEntry(player_name=f"Player {i}", position="attack", importance_weight=1.0)
        for i in range(5)
    ]
    attack, _ = adjust_strength(1.2, 0.9, injuries)
    assert attack == 1.2 * MIN_MULTIPLIER


def test_multiple_severe_defensive_injuries_are_clipped_to_ceiling():
    injuries = [
        InjuryEntry(player_name=f"Player {i}", position="defense", importance_weight=1.0)
        for i in range(5)
    ]
    _, defense = adjust_strength(1.2, 0.9, injuries)
    assert defense == 0.9 * MAX_MULTIPLIER
