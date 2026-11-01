"""The script that makes a table link open its panel without reloading the page: it has to find
the link, the hidden box and the URL parameter by the same names the page draws them with."""

from __future__ import annotations

import datetime as dt

import pytest

from soccer_predictor.dashboard.components import (
    FIXTURE_LINKS_JS,
    FIXTURE_PARAM,
    FIXTURE_PICK_KEY,
    INJURY_LINKS_JS,
    INJURY_PARAM,
    INJURY_PICK_KEY,
    PLAYER_LINKS_JS,
    PLAYER_PARAM,
    PLAYER_PICK_KEY,
    RATING_LINKS_JS,
    RATING_PARAM,
    RATING_PICK_KEY,
    SCROLL_TO_BREAKDOWN_JS,
    SCROLL_TO_INJURY_JS,
    SCROLL_TO_PLAYER_JS,
    SCROLL_TO_RATING_JS,
    FixtureRef,
    fixtures_table_html,
)


@pytest.mark.parametrize(
    ("script", "selector", "param", "box", "scroll"),
    [
        (RATING_LINKS_JS, "table.fb-table a.stat", RATING_PARAM, RATING_PICK_KEY, SCROLL_TO_RATING_JS),
        (FIXTURE_LINKS_JS, "table.fb-fixtures a.score", FIXTURE_PARAM, FIXTURE_PICK_KEY, SCROLL_TO_BREAKDOWN_JS),
        (PLAYER_LINKS_JS, "table.fb-table a.player", PLAYER_PARAM, PLAYER_PICK_KEY, SCROLL_TO_PLAYER_JS),
        (INJURY_LINKS_JS, "table.fb-table a.injured", INJURY_PARAM, INJURY_PICK_KEY, SCROLL_TO_INJURY_JS),
    ],
)
def test_each_script_uses_its_own_link_box_parameter_and_scroll(script, selector, param, box, scroll):
    assert f"closest('{selector}')" in script
    assert f".st-key-{box} input" in script
    assert f"searchParams.get('{param}')" in script
    assert f"__fbInPlace_{param}" in script  # installed once per parameter, so they never block each other
    assert scroll in script


@pytest.mark.parametrize("script", [RATING_LINKS_JS, FIXTURE_LINKS_JS, PLAYER_LINKS_JS, INJURY_LINKS_JS])
def test_no_template_placeholder_is_left_in_a_script(script):
    for placeholder in ("SELECTOR", "BOXKEY", "SCROLL", "PARAM"):
        assert placeholder not in script


def test_the_scripts_do_not_answer_each_others_links():
    assert "a.stat" not in FIXTURE_LINKS_JS and "a.score" not in RATING_LINKS_JS
    assert "a.player" not in RATING_LINKS_JS + FIXTURE_LINKS_JS
    assert "a.stat" not in PLAYER_LINKS_JS and "a.score" not in PLAYER_LINKS_JS
    assert "a.injured" not in RATING_LINKS_JS + FIXTURE_LINKS_JS + PLAYER_LINKS_JS
    assert not any(link in INJURY_LINKS_JS for link in ("a.stat", "a.score", "a.player"))


def test_the_predicted_score_link_is_the_one_the_script_looks_for():
    ref = FixtureRef(1, 2, "A", "B", dt.date(2026, 11, 1))
    row = {"Home": "A", "Away": "B", "Predicted score": "2-1", "Kickoff": "x", "_predicted_home_goals": 2, "_predicted_away_goals": 1}
    markup = fixtures_table_html([row], [ref], "EPL")
    assert 'class="fb-fixtures"' in markup and '<a class="score" href="?league=EPL' in markup
    assert f"&amp;{FIXTURE_PARAM}=" in markup  # the value the script reads from the link


@pytest.mark.parametrize("script", [RATING_LINKS_JS, FIXTURE_LINKS_JS, PLAYER_LINKS_JS, INJURY_LINKS_JS])
def test_the_click_is_committed_with_enter_not_with_focus_events(script):
    # A browser that is not the active window sends no focus or blur events, so a click that
    # relied on them would silently do nothing there; a key event is delivered regardless.
    assert "key: 'Enter'" in script and "keydown" in script
    assert ".focus(" not in script and ".blur(" not in script


@pytest.mark.parametrize("script", [RATING_LINKS_JS, FIXTURE_LINKS_JS, PLAYER_LINKS_JS, INJURY_LINKS_JS])
def test_only_a_plain_left_click_is_taken_over(script):
    for guard in ("event.button !== 0", "event.ctrlKey", "event.metaKey", "event.shiftKey", "event.altKey"):
        assert guard in script  # a new-tab click still follows the link
