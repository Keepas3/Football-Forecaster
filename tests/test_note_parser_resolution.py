from __future__ import annotations

from soccer_predictor.ai.note_parser import _resolve_team_name

TEAM_NAMES = {1: "Arsenal", 2: "Aston Villa", 3: "Man United", 4: "Man City"}


def test_exact_name_resolves():
    assert _resolve_team_name("Arsenal", TEAM_NAMES) == 1


def test_near_typo_resolves():
    assert _resolve_team_name("Arsenl", TEAM_NAMES) == 1
    assert _resolve_team_name("Aston Vila", TEAM_NAMES) == 2


def test_abbreviation_does_not_resolve_and_falls_back_to_manual_pick():
    # "Man Utd" isn't a close-enough edit-distance match to "Man United" at
    # this threshold -- by design it should fall through to the review
    # card's manual team picker rather than silently guessing.
    assert _resolve_team_name("Man Utd", TEAM_NAMES) is None


def test_unrelated_name_does_not_resolve():
    assert _resolve_team_name("Totally Unrelated FC", TEAM_NAMES) is None


def test_empty_team_names_never_resolves():
    assert _resolve_team_name("Arsenal", {}) is None
