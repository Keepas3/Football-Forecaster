from __future__ import annotations

from soccer_predictor.model.form_adjustment import (
    FORM_IMPACT_CAP,
    FormNoteEntry,
    adjust_for_form,
)


def test_no_notes_leaves_strength_unchanged():
    attack, defense = adjust_for_form(1.2, 0.9, notes=[])
    assert attack == 1.2
    assert defense == 0.9


def test_positive_form_note_raises_attack():
    notes = [FormNoteEntry(magnitude=0.5, affects="attack")]
    attack, defense = adjust_for_form(1.2, 0.9, notes)

    assert attack > 1.2
    assert defense == 0.9


def test_negative_form_note_on_defense_raises_defense_value():
    # Recall: defense is a goals-conceded multiplier, so bad form (negative
    # magnitude) makes the team WORSE defensively, i.e. a HIGHER value.
    notes = [FormNoteEntry(magnitude=-0.5, affects="defense")]
    attack, defense = adjust_for_form(1.2, 0.9, notes)

    assert attack == 1.2
    assert defense > 0.9


def test_positive_form_note_on_defense_lowers_defense_value():
    notes = [FormNoteEntry(magnitude=0.5, affects="defense")]
    _, defense = adjust_for_form(1.2, 0.9, notes)
    assert defense < 0.9


def test_both_applies_to_attack_and_defense():
    notes = [FormNoteEntry(magnitude=-0.4, affects="both")]
    attack, defense = adjust_for_form(1.2, 0.9, notes)
    assert attack < 1.2
    assert defense > 0.9


def test_multiple_notes_sum_then_clip_to_cap():
    notes = [FormNoteEntry(magnitude=1.0, affects="attack") for _ in range(5)]
    attack, _ = adjust_for_form(1.2, 0.9, notes)
    assert attack == 1.2 * (1.0 + FORM_IMPACT_CAP)


def test_opposing_notes_partially_cancel():
    notes = [
        FormNoteEntry(magnitude=0.6, affects="attack"),
        FormNoteEntry(magnitude=-0.2, affects="attack"),
    ]
    attack, _ = adjust_for_form(1.2, 0.9, notes)
    assert attack == 1.2 * (1.0 + 0.4 * FORM_IMPACT_CAP)
