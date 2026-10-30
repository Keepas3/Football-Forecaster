"""Adjusts a team's strength for the starting XI actually named for a match.

The fitted attack/defense rating is an average over the whole season, in which
the team's usual players played most of the minutes, so a team fielding that
usual XI needs no adjustment. Each member of the usual XI who is NOT in the
named XI (injured, suspended, rested, dropped -- the reason does not matter) is
an absence, and costs the team that player's importance weight, the same
weight an injury to him would cost (see model/player_importance.py).

Injuries are only an early guess at who will play; once the XI is named it
replaces them (a "doubtful" player who starts is no longer penalised, and an
unannounced absence now counts), so the two are never applied together.

The impact factors scale a weight into a rating swing. They are tuned by
scripts/backtest_lineups.py against real past line-ups, not guessed.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz

from soccer_predictor.model.injury_adjustment import MAX_MULTIPLIER, MIN_MULTIPLIER

EXPECTED_XI_SIZE = 11
# How much one absent player's importance weight moves the rating. Chosen by
# scripts/backtest_lineups.py on 433 matches from six leagues of the 2026-27
# season, with base ratings fitted before 1 Sep 2026. At these values the lineup
# improves the ranked probability score slightly (0.2088 -> 0.2072), but on
# held-out halves the gain is only -0.0012 with a 95% interval of -0.0040..+0.0013,
# so it is small and not statistically certain (better in four leagues, slightly
# worse in La Liga and Ligue 1). Strengths as large as an injury's (0.5 for both)
# make it clearly worse (0.2120), so the injury factors are NOT reused. The
# surface is flat between about 0.2 and 0.3 for attack and 0.05 and 0.1 for
# defense. Re-run the backtest as the season's sample grows. The what-if injuries
# a visitor adds keep the injury factors (0.5, model/injury_adjustment.py).
LINEUP_ATTACK_FACTOR = 0.3
LINEUP_DEFENSE_FACTOR = 0.05
# Two full names that are spelt differently count as the same player at or above
# this similarity, with the surnames alone at least SURNAME_MATCH_THRESHOLD alike.
NAME_MATCH_THRESHOLD = 92
SURNAME_MATCH_THRESHOLD = 80


@dataclass(frozen=True)
class ExpectedPlayer:
    """One member of a team's squad as the season's stats describe him."""

    name: str
    position: str  # "attack" | "defense"
    weight: float  # importance weight, same scale as an injury's
    minutes: float
    is_keeper: bool = False


@dataclass(frozen=True)
class LineupAbsence:
    name: str
    position: str  # "attack" | "defense"
    weight: float


def fold_player_name(name: str) -> str:
    """Lower-case, accent-free, punctuation-free: "N'Golo Kanté" -> "ngolo kante"."""
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^a-z0-9 ]+", "", text.replace("-", " ")).strip()


def same_player(a: str, b: str, threshold: int = NAME_MATCH_THRESHOLD) -> bool:
    """Whether two feeds' spellings name the same player: "Bukayo Saka" and
    "B. Saka" and "Saka" do, "Saka" and "Sakai" do not."""
    folded_a, folded_b = fold_player_name(a), fold_player_name(b)
    if not folded_a or not folded_b:
        return False
    if folded_a == folded_b:
        return True
    tokens_a, tokens_b = folded_a.split(), folded_b.split()
    shorter, longer = sorted((tokens_a, tokens_b), key=len)
    if all(token in longer for token in shorter) and any(len(token) >= 3 for token in shorter):
        return True  # "Vinicius Junior" within "Vinicius Jose Paixao de Oliveira Junior", "Saka" within "Bukayo Saka"
    if tokens_a[-1] == tokens_b[-1] and (
        len(tokens_a) == 1 or len(tokens_b) == 1 or tokens_a[0][0] == tokens_b[0][0]
    ):
        return True  # same surname, first name abbreviated or missing
    # A misspelt surname: only for full names, where near-identical text really is one
    # person -- on a single word ("Saka"/"Sakai") or a number ("Regular 3"/"Regular 4")
    # a high similarity score is just two different players.
    return (
        len(tokens_a) > 1
        and len(tokens_b) > 1
        and fuzz.ratio(tokens_a[-1], tokens_b[-1]) >= SURNAME_MATCH_THRESHOLD
        and fuzz.token_sort_ratio(folded_a, folded_b) >= threshold
    )


def expected_xi(squad: Sequence[ExpectedPlayer]) -> list[ExpectedPlayer]:
    """The XI the team usually fields: its most-used goalkeeper plus the ten
    outfielders with most minutes (so a rotation player never counts as missing)."""
    keepers = sorted((p for p in squad if p.is_keeper), key=lambda p: p.minutes, reverse=True)
    outfield = sorted((p for p in squad if not p.is_keeper), key=lambda p: p.minutes, reverse=True)
    xi = keepers[:1] + outfield[: EXPECTED_XI_SIZE - len(keepers[:1])]
    return [p for p in xi if p.minutes > 0]


def lineup_absences(
    squad: Sequence[ExpectedPlayer], xi_names: Sequence[str], threshold: int = NAME_MATCH_THRESHOLD
) -> list[LineupAbsence]:
    """Members of the usual XI that none of `xi_names` is."""
    return [
        LineupAbsence(player.name, player.position, player.weight)
        for player in expected_xi(squad)
        if not any(same_player(player.name, name, threshold) for name in xi_names)
    ]


def adjust_for_lineup(
    base_attack: float,
    base_defense: float,
    absences: Sequence[LineupAbsence],
    attack_factor: float = LINEUP_ATTACK_FACTOR,
    defense_factor: float = LINEUP_DEFENSE_FACTOR,
) -> tuple[float, float]:
    """Same shape as injury_adjustment.adjust_strength (an absent attacker pulls
    attack down, an absent defender or keeper pushes the goals-conceded
    multiplier up, within the same floor and ceiling), with its own factors."""
    attack_hit = sum(a.weight for a in absences if a.position == "attack") * attack_factor
    defense_hit = sum(a.weight for a in absences if a.position == "defense") * defense_factor
    return (
        base_attack * max(1.0 - attack_hit, MIN_MULTIPLIER),
        base_defense * min(1.0 + defense_hit, MAX_MULTIPLIER),
    )
