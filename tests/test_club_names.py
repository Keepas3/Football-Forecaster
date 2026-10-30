"""Matching a feed's club name to ours (ingest/club_names.py)."""

from __future__ import annotations

from soccer_predictor.ingest import club_names

OURS = ["Arsenal FC", "Manchester City FC", "Manchester United FC", "Club Atlético de Madrid", "SK Slavia Praha"]


def test_accents_and_punctuation_are_folded_away():
    assert club_names.fold_name("Atlético de Madrid") == "atletico de madrid"
    assert club_names.fold_name("D.C. United") == "d c united"
    assert club_names.strip_accents("Pongračić") == "Pongracic"
    assert club_names.strip_accents("Ngankam") == "Ngankam"  # nothing to strip: unchanged, capitals kept


def test_words_matching_clubs_are_found_and_exact_names_win():
    assert club_names.resolve_club_name("Arsenal", OURS) == "Arsenal FC"
    assert club_names.resolve_club_name("Atlético de Madrid", OURS) == "Club Atlético de Madrid"
    assert club_names.resolve_club_name("Manchester City FC", OURS) == "Manchester City FC"


def test_ties_and_strangers_are_left_unmatched():
    assert club_names.resolve_club_name("Manchester", OURS) is None  # City or United
    assert club_names.resolve_club_name("Atlantis", OURS) is None


def test_an_override_names_the_club_words_alone_cannot():
    assert club_names.resolve_club_name("Slavia Prag", OURS) is None
    assert club_names.resolve_club_name("Slavia Prag", OURS, {"slavia prag": "SK Slavia Praha"}) == "SK Slavia Praha"
