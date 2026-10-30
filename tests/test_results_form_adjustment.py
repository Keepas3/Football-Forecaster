from __future__ import annotations

import datetime as dt

import pytest

from soccer_predictor.model.results_form_adjustment import (
    AVERAGE_POINTS_PER_MATCH,
    MAX_SPAN_DAYS,
    RECENT_MATCHES,
    RESULTS_FORM_STRENGTH,
    ResultsForm,
    adjust_for_results_form,
    results_form,
)

MATCH_DAY = dt.date(2026, 10, 31)


def _run(points: list[int], days_apart: int = 7) -> list[tuple[dt.date, int]]:
    """Results newest first, one every `days_apart` days before the match day."""
    return [(MATCH_DAY - dt.timedelta(days=days_apart * (i + 1)), p) for i, p in enumerate(points)]


def test_an_average_run_barely_moves_the_rating():
    form = results_form(_run([3, 3, 1, 1, 0, 0]), MATCH_DAY)  # 8 points: 1.33 a match, the average is 1.375
    assert form.points == 8 and form.matches == RECENT_MATCHES and abs(form.shift) < 0.0025


def test_a_winless_run_cuts_the_rating_and_a_perfect_one_lifts_it_by_the_same_rule():
    bad = results_form(_run([0, 0, 0, 0, 0, 0]), MATCH_DAY)
    good = results_form(_run([3, 3, 3, 3, 3, 3]), MATCH_DAY)
    assert bad.shift == pytest.approx(-RESULTS_FORM_STRENGTH * AVERAGE_POINTS_PER_MATCH)
    assert good.shift == pytest.approx(RESULTS_FORM_STRENGTH * (3 - AVERAGE_POINTS_PER_MATCH))
    assert -0.08 < bad.shift < 0 < good.shift < 0.09  # the arithmetic itself bounds it


def test_only_the_latest_six_results_count():
    results = _run([0, 0, 0, 0, 0, 0, 3, 3, 3])  # a long unbeaten run, but long ago
    assert results_form(results, MATCH_DAY).points == 0


def test_fewer_than_six_results_is_no_form_at_all():
    assert results_form(_run([3] * (RECENT_MATCHES - 1)), MATCH_DAY) is None
    assert results_form([], MATCH_DAY) is None


def test_six_results_stretched_over_too_long_a_time_are_not_form():
    stretched = _run([0] * RECENT_MATCHES, days_apart=MAX_SPAN_DAYS // RECENT_MATCHES + 5)
    assert results_form(stretched, MATCH_DAY) is None
    just_inside = _run([0] * RECENT_MATCHES, days_apart=MAX_SPAN_DAYS // RECENT_MATCHES)
    assert results_form(just_inside, MATCH_DAY) is not None


def test_poor_form_lowers_attack_and_raises_goals_conceded():
    form = ResultsForm(points=2, matches=6, shift=-0.05)
    attack, defense = adjust_for_results_form(1.2, 0.9, form)
    assert attack == pytest.approx(1.2 * 0.95)
    assert defense == pytest.approx(0.9 / 0.95) and defense > 0.9


def test_good_form_lifts_attack_and_lowers_goals_conceded():
    attack, defense = adjust_for_results_form(1.2, 0.9, ResultsForm(points=16, matches=6, shift=0.04))
    assert attack == pytest.approx(1.2 * 1.04)
    assert defense == pytest.approx(0.9 / 1.04) and defense < 0.9


def test_no_form_leaves_the_rating_untouched():
    assert adjust_for_results_form(1.2, 0.9, None) == (1.2, 0.9)
