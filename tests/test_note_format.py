from __future__ import annotations

import datetime as dt

from soccer_predictor.dashboard.note_format import describe_form_note, describe_injury
from soccer_predictor.model.form_adjustment import FormNoteEntry, adjust_for_form
from soccer_predictor.model.injury_adjustment import InjuryEntry, adjust_strength

TODAY = dt.date(2026, 10, 5)


def test_bad_form_note_headline_and_maths_are_on_separate_lines():
    headline, maths = describe_form_note(
        "Missing key star player Edin Dzeko for the upcoming Poland game impacts attacking strength.",
        -0.6,
        "attack",
        dt.date(2026, 10, 12),
        TODAY,
    )
    assert headline == (
        "**Form note** · 📉 Bad form — Missing key star player Edin Dzeko for the upcoming Poland game "
        "impacts attacking strength"
    )
    assert "−9.0%" in maths and "attack strength" in maths
    assert "−0.60 × 15% cap" in maths
    assert "Expires Oct 12, 2026 (in 7 days)" in maths
    assert "\n" not in headline and "Maths:" not in headline


def test_good_form_note_is_a_boost():
    headline, maths = describe_form_note("Poland won 6-0.", 0.7, "attack", dt.date(2026, 10, 19), TODAY)
    assert "📈 Good form" in headline
    assert "attack strength +10.5%" in maths


def test_defence_form_note_shows_goals_conceded_not_a_confusing_negative():
    # Bad defensive form means conceding MORE.
    _, maths = describe_form_note("Leaky", -0.5, "defense", dt.date(2026, 10, 12), TODAY)
    assert "goals conceded +7.5%" in maths


def test_form_note_that_affects_both_lists_both_effects():
    _, maths = describe_form_note("Flat", -1.0, "both", dt.date(2026, 10, 12), TODAY)
    assert "attack strength −15.0%" in maths and "goals conceded +15.0%" in maths


def test_form_note_numbers_match_what_the_model_actually_applies():
    attack, defense = adjust_for_form(1.0, 1.0, [FormNoteEntry(-0.6, "both")])
    _, maths = describe_form_note("x", -0.6, "both", dt.date(2026, 10, 12), TODAY)
    # attack multiplier 0.91 -> −9.0%; defence (goals conceded) multiplier 1.09 -> +9.0%
    assert abs((attack - 1) + 0.09) < 1e-9 and "attack strength −9.0%" in maths
    assert abs((defense - 1) - 0.09) < 1e-9 and "goals conceded +9.0%" in maths


def test_injury_with_a_return_date():
    headline, maths = describe_injury("Lewandowski", "attack", 0.7, dt.date(2026, 10, 26), TODAY)
    assert headline == "**Injury** · 🩹 Lewandowski (attack player) — expected back Oct 26, 2026 (in 21 days)"
    assert "importance 0.70 × 0.5" in maths and "attack strength −35.0%" in maths


def test_injury_without_a_return_date_says_it_counts_until_deleted():
    headline, _ = describe_injury("Saka", "attack", 0.8, None, TODAY)
    assert "no return date set" in headline and "until you delete it" in headline


def test_defender_injury_raises_goals_conceded():
    _, maths = describe_injury("Rúben Dias", "defense", 0.6, None, TODAY)
    assert "goals conceded +30.0%" in maths


def test_injury_numbers_match_the_model_and_respect_its_floor():
    attack, _ = adjust_strength(1.0, 1.0, [InjuryEntry("x", "attack", 0.7)])
    _, maths = describe_injury("x", "attack", 0.7, None, TODAY)
    assert abs(attack - 0.65) < 1e-9 and "−35.0%" in maths
    # An importance above 1 can't cut a rating past the floor.
    _, floored = describe_injury("x", "attack", 1.5, None, TODAY)
    assert "−50.0%" in floored


def test_countdown_wording():
    def when(days):
        return describe_form_note("x", 0.1, "attack", TODAY + dt.timedelta(days=days), TODAY)[1]

    assert "(today)" in when(0)
    assert "(tomorrow)" in when(1)
    assert "(in 14 days)" in when(14)
