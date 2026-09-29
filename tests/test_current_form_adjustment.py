from __future__ import annotations

from soccer_predictor.model.current_form_adjustment import (
    CURRENT_FORM_IMPACT_CAP,
    adjust_for_current_attack_form,
)


def test_none_leaves_attack_unchanged():
    assert adjust_for_current_attack_form(1.2, None) == 1.2


def test_above_average_current_form_raises_attack():
    attack = adjust_for_current_attack_form(1.2, relative_xg_strength=1.5)
    assert attack > 1.2


def test_below_average_current_form_lowers_attack():
    attack = adjust_for_current_attack_form(1.2, relative_xg_strength=0.5)
    assert attack < 1.2


def test_exactly_league_average_leaves_attack_unchanged():
    assert adjust_for_current_attack_form(1.2, relative_xg_strength=1.0) == 1.2


def test_large_overperformance_clips_to_cap():
    attack = adjust_for_current_attack_form(1.2, relative_xg_strength=5.0)
    assert attack == 1.2 * (1.0 + CURRENT_FORM_IMPACT_CAP)


def test_large_underperformance_clips_to_cap():
    attack = adjust_for_current_attack_form(1.2, relative_xg_strength=-2.0)
    assert attack == 1.2 * (1.0 - CURRENT_FORM_IMPACT_CAP)
