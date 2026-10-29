"""The Injuries tab's table (dashboard/injury_table.py): the feeds' note strings
split into columns, and the filters and team totals built on them."""

from __future__ import annotations

import datetime as dt

import pytest

from soccer_predictor.dashboard import injury_table as it
from soccer_predictor.model.injury_adjustment import InjuryEntry
from soccer_predictor.prediction.service import TeamInjuries

TODAY = dt.date(2026, 10, 29)


@pytest.mark.parametrize(
    ("note", "status", "problem", "chance", "back"),
    [
        ("FPL: Back injury - Unknown return date", it.OUT, "Back injury", 0, None),
        ("FPL: Hip injury - 75% chance of playing", it.DOUBTFUL, "Hip injury", 75, None),
        ("FPL: Hamstring injury - Expected back 1 Nov", it.OUT, "Hamstring injury", 0, dt.date(2026, 11, 1)),
        ("FPL: Suspended until 29 Nov", it.SUSPENDED, "Suspension", 0, dt.date(2026, 11, 29)),
        ("FPL: Unspecified injury - 25% chance of playing", it.DOUBTFUL, "Unspecified injury", 25, None),
        ("FPL: Doubtful (50% chance of playing)", it.DOUBTFUL, "Not specified", 50, None),
        ("FPL: Out", it.OUT, "Out", 0, None),
        ("Day-To-Day", it.DOUBTFUL, "Day-To-Day", None, None),
        ("", it.OUT, "Not specified", 0, None),
        ("Biwenger: Ankle injury - Expected back 25 Nov 2026", it.OUT, "Ankle injury", 0, dt.date(2026, 11, 25)),
        ("Biwenger: Meniscus injury - Expected back mid-November", it.OUT, "Meniscus injury", 0, dt.date(2026, 11, 15)),
        ("Biwenger: Tibia fracture - Expected back early January", it.OUT, "Tibia fracture", 0, dt.date(2027, 1, 5)),
        ("Biwenger: Knee - Expected back end of November", it.OUT, "Knee", 0, dt.date(2026, 11, 28)),
        ("Biwenger: Knee injury - Unknown return date", it.OUT, "Knee injury", 0, None),
        ("Biwenger: Muscle injury - 50% chance of playing", it.DOUBTFUL, "Muscle injury", 50, None),
        ("Biwenger: 50% chance of playing", it.DOUBTFUL, "Not specified", 50, None),
        ("Biwenger: Direct red card - Suspended through Matchday 4", it.SUSPENDED, "Direct red card", 0, None),
        ("Biwenger: Back injury - Expected back mid-June", it.OUT, "Back injury", 0, dt.date(2027, 6, 15)),
    ],
)
def test_a_feed_note_is_split_into_columns(note, status, problem, chance, back):
    parsed = it.parse_note(note, TODAY)
    assert (parsed.status, parsed.problem, parsed.chance, parsed.back) == (status, problem, chance, back)


def test_a_return_date_in_early_january_is_next_year_and_a_stored_date_wins():
    assert it.parse_note("FPL: Knee injury - Expected back 2 Jan", TODAY).back == dt.date(2027, 1, 2)
    stored = dt.date(2026, 12, 12)
    assert it.parse_note("FPL: Knee injury - Expected back 2 Jan", TODAY, stored).back == stored


def _team(entries, sources, notes, returns=None):
    return TeamInjuries(
        entries=entries, sources=sources, expected_return_dates=returns or [None] * len(entries), notes=notes
    )


def _frame():
    arsenal = _team(
        [InjuryEntry("Saka", "attack", 0.54), InjuryEntry("Saliba", "defense", 0.10), InjuryEntry("Rice", "attack", 0.03)],
        ["api", "api", "api"],
        ["FPL: Thigh injury - Unknown return date", "FPL: Back injury - Unknown return date", "FPL: Suspended until 29 Nov"],
    )
    villa = _team([InjuryEntry("Onana", "attack", 0.10)], ["manual"], ["Knee"], [dt.date(2027, 1, 5)])
    return it.players_frame(it.player_rows("Arsenal", arsenal, TODAY) + it.player_rows("Aston Villa", villa, TODAY))


def test_rows_have_readable_values_and_biggest_impact_first_within_a_team():
    frame = _frame()
    assert list(frame.columns) == it.PLAYER_COLUMNS
    assert list(frame[frame["Team"] == "Arsenal"]["Player"]) == ["Saka", "Saliba", "Rice"]
    saliba = frame[frame["Player"] == "Saliba"].iloc[0]
    assert (saliba["Position"], saliba["Source"], saliba["Status"]) == ("Defence", "Automatic feed", it.OUT)
    onana = frame[frame["Player"] == "Onana"].iloc[0]
    assert onana["Source"] == "Manual list" and onana["Expected back"] == dt.date(2027, 1, 5)


def test_filters_combine_and_empty_ones_do_nothing():
    frame = _frame()
    assert len(it.filter_players(frame)) == 4
    assert list(it.filter_players(frame, teams=["Arsenal"], positions=["Attack"])["Player"]) == ["Saka", "Rice"]
    assert list(it.filter_players(frame, statuses=[it.SUSPENDED])["Player"]) == ["Rice"]
    assert list(it.filter_players(frame, query="  sAk ")["Player"]) == ["Saka"]
    assert it.filter_players(frame, teams=["Arsenal"], query="Onana").empty
    assert list(it.filter_players(frame, query="a.")["Player"]) == []  # taken literally, not as a pattern


def test_counts_and_an_empty_frame():
    counts = it.summary_counts(_frame())
    assert counts == {"players": 4, "teams": 2, it.OUT: 3, it.DOUBTFUL: 0, it.SUSPENDED: 1}
    empty = it.players_frame([])
    assert empty.empty and list(empty.columns) == it.PLAYER_COLUMNS
    assert it.summary_counts(empty)["players"] == 0


def test_team_impact_uses_the_models_own_adjustment_and_lists_the_worst_hit_first():
    impact = it.team_impact_frame(
        {
            "Arsenal": (1.5, 0.64, [InjuryEntry("Saka", "attack", 0.54), InjuryEntry("Saliba", "defense", 0.10)]),
            "Villa": (1.2, 1.0, [InjuryEntry("Onana", "attack", 0.10)]),
            "Healthy": (1.0, 1.0, []),
        }
    )
    assert list(impact["Team"]) == ["Arsenal", "Villa"]  # a team with nobody listed is left out
    arsenal = impact.iloc[0]
    assert arsenal["Players listed"] == 2
    assert arsenal["Attack with injuries"] == pytest.approx(1.5 * (1 - 0.54 * 0.5), abs=0.01)
    assert arsenal["Attack change"] == pytest.approx(-27.0, abs=0.1)
    assert arsenal["Defence change"] == pytest.approx(5.0, abs=0.1)
    assert it.team_impact_frame({}).empty
