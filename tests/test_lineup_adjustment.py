"""model/lineup_adjustment.py: who is missing from a team's usual XI, and what that does to its rating."""

from __future__ import annotations

import pytest

from soccer_predictor.model.injury_adjustment import MAX_MULTIPLIER, MIN_MULTIPLIER
from soccer_predictor.model.lineup_adjustment import (
    ExpectedPlayer,
    LineupAbsence,
    adjust_for_lineup,
    expected_xi,
    fold_player_name,
    lineup_absences,
    same_player,
)


def _player(name, minutes=900, position="attack", weight=0.3, keeper=False):
    return ExpectedPlayer(name, position, weight, minutes, keeper)


def _squad():
    """A keeper, ten usual outfield players (minutes 1000 down to 910) and two rotation players."""
    squad = [_player("Keeper One", 1080, "defense", 0.95, keeper=True), _player("Spare Keeper", 90, "defense", 0.05, keeper=True)]
    squad += [_player(f"Regular {i}", 1000 - i * 10) for i in range(10)]
    squad += [_player("Rotation A", 300), _player("Rotation B", 120)]
    return squad


# --- names ---------------------------------------------------------------------------------------


def test_names_are_folded_to_plain_lower_case():
    assert fold_player_name("N'Golo Kanté") == "ngolo kante"
    assert fold_player_name("Trent Alexander-Arnold") == "trent alexander arnold"
    assert fold_player_name("") == ""


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Bukayo Saka", "Bukayo Saka"),
        ("Bukayo Saka", "B. Saka"),
        ("Saka", "Bukayo Saka"),
        ("N'Golo Kanté", "Ngolo Kante"),
        ("Aleksey Miranchuk", "Alexey Miranchuk"),
        ("Vinicius Junior", "Vinicius Jose Paixao de Oliveira Junior"),
        ("Ezri Konsa Ngoyo", "Ezri Konsa"),
    ],
)
def test_spellings_of_one_player_match(a, b):
    assert same_player(a, b) and same_player(b, a)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Saka", "Sakai"),
        ("Regular 3", "Regular 4"),
        ("Bernardo Silva", "David Silva"),
        ("Marcus Rashford", "Marcus Thuram"),
        ("", "Bukayo Saka"),
    ],
)
def test_different_players_do_not_match(a, b):
    assert not same_player(a, b)


# --- the usual XI and the absences ----------------------------------------------------------------


def test_the_usual_xi_is_the_main_keeper_and_the_ten_outfielders_with_most_minutes():
    xi = expected_xi(_squad())
    names = [p.name for p in xi]
    assert len(xi) == 11 and names[0] == "Keeper One"
    assert "Spare Keeper" not in names and "Rotation A" not in names and "Rotation B" not in names
    assert "Regular 9" in names  # the tenth outfielder


def test_players_who_never_played_are_not_in_the_usual_xi():
    squad = [_player("Played", 500), _player("Never Played", 0)]
    assert [p.name for p in expected_xi(squad)] == ["Played"]


def test_without_a_keeper_in_the_data_the_eleven_busiest_are_taken():
    squad = [_player(f"P{i}", 1000 - i) for i in range(14)]
    assert len(expected_xi(squad)) == 11


def test_the_usual_xi_playing_means_no_absences():
    squad = _squad()
    assert lineup_absences(squad, [p.name for p in expected_xi(squad)]) == []


def test_a_missing_regular_is_an_absence_with_his_weight():
    squad = _squad()
    xi = [p.name for p in expected_xi(squad) if p.name != "Regular 3"] + ["Rotation A"]
    assert lineup_absences(squad, xi) == [LineupAbsence("Regular 3", "attack", 0.3)]


def test_a_rotation_player_not_starting_is_never_an_absence():
    squad = _squad()
    xi = [p.name for p in expected_xi(squad)]
    assert "Rotation A" not in xi and lineup_absences(squad, xi) == []


def test_a_different_spelling_in_the_named_xi_still_counts_as_present():
    squad = [_player("Bukayo Saka", 900)]
    assert lineup_absences(squad, ["B. Saka"]) == []


# --- the adjustment -------------------------------------------------------------------------------


def test_no_absences_or_zero_factors_change_nothing():
    assert adjust_for_lineup(1.3, 0.9, []) == (1.3, 0.9)
    assert adjust_for_lineup(1.3, 0.9, [LineupAbsence("A", "attack", 0.9)], 0.0, 0.0) == (1.3, 0.9)


def test_an_absent_attacker_lowers_attack_and_an_absent_defender_raises_goals_conceded():
    attack, defense = adjust_for_lineup(1.0, 1.0, [LineupAbsence("A", "attack", 0.4)], 0.5, 0.5)
    assert (attack, defense) == (pytest.approx(0.8), 1.0)
    attack, defense = adjust_for_lineup(1.0, 1.0, [LineupAbsence("D", "defense", 0.5)], 0.5, 0.5)
    assert (attack, defense) == (1.0, pytest.approx(1.25))


def test_the_two_factors_are_independent():
    absences = [LineupAbsence("A", "attack", 0.4), LineupAbsence("D", "defense", 0.4)]
    attack, defense = adjust_for_lineup(1.0, 1.0, absences, attack_factor=0.25, defense_factor=0.1)
    assert attack == pytest.approx(0.9) and defense == pytest.approx(1.04)


def test_many_absences_stop_at_the_same_floor_and_ceiling_as_injuries():
    attackers = [LineupAbsence(f"A{i}", "attack", 0.9) for i in range(8)]
    defenders = [LineupAbsence(f"D{i}", "defense", 0.9) for i in range(8)]
    attack, defense = adjust_for_lineup(1.0, 1.0, attackers + defenders, 0.5, 0.5)
    assert attack == MIN_MULTIPLIER and defense == MAX_MULTIPLIER
