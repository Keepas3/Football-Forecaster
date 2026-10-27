"""Pure helpers for the Predictor page's "AI what-if" tab: turning what the AI
read from a note into the model's own adjustment inputs, the "model only" v
"with this note" table, and plain-English lines for each adjustment. No
Streamlit, no database, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from soccer_predictor.ai.whatif_parser import WhatIfEntries
from soccer_predictor.model.form_adjustment import FORM_IMPACT_CAP, FormNoteEntry
from soccer_predictor.model.injury_adjustment import (
    ATTACK_IMPACT_FACTOR,
    DEFENSE_IMPACT_FACTOR,
    MIN_MULTIPLIER,
    InjuryEntry,
)
from soccer_predictor.model.markets import MatchPrediction


@dataclass
class Adjustments:
    """The what-if entries split by side, ready for predict_fixture's extra_* arguments."""

    home_injuries: list[InjuryEntry] = field(default_factory=list)
    away_injuries: list[InjuryEntry] = field(default_factory=list)
    home_form: list[FormNoteEntry] = field(default_factory=list)
    away_form: list[FormNoteEntry] = field(default_factory=list)


def build_adjustments(
    entries: WhatIfEntries,
    home_id: int,
    injury_scale: list[float] | None = None,
    form_scale: list[float] | None = None,
) -> Adjustments:
    """`*_scale` is how much of each entry to count (0 to 1, one per entry, in
    order) -- the sliders that let a visitor soften the AI's numbers without
    asking again. Default: all of it."""
    out = Adjustments()
    for index, injury in enumerate(entries.injuries):
        scale = injury_scale[index] if injury_scale and index < len(injury_scale) else 1.0
        entry = InjuryEntry(injury.player_name, injury.position, injury.importance_weight * scale)
        (out.home_injuries if injury.team_id == home_id else out.away_injuries).append(entry)
    for index, form in enumerate(entries.form_notes):
        scale = form_scale[index] if form_scale and index < len(form_scale) else 1.0
        entry = FormNoteEntry(form.magnitude * scale, form.affects)
        (out.home_form if form.team_id == home_id else out.away_form).append(entry)
    return out


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _signed(value: float) -> str:
    """+0.40 / −0.40 (a real minus sign, like dashboard/note_format.py)."""
    return f"{'+' if value >= 0 else '−'}{abs(value):.2f}"


def _signed_pct(fraction: float) -> str:
    return f"{'+' if fraction >= 0 else '−'}{abs(fraction) * 100:.1f}%"


def _change_points(before: float, after: float) -> str:
    diff = (after - before) * 100
    if abs(diff) < 0.05:
        return "no change"
    return f"{'+' if diff > 0 else '−'}{abs(diff):.1f} pts"


def _score(prediction: MatchPrediction) -> str:
    home, away, _ = prediction.top_scorelines[0]
    return f"{home}-{away}"


def comparison_rows(base: MatchPrediction, adjusted: MatchPrediction) -> list[dict]:
    """The rows of the "model only" v "with this note" table."""
    rows = []
    for label, before, after in (
        ("Home win", base.home_win, adjusted.home_win),
        ("Draw", base.draw, adjusted.draw),
        ("Away win", base.away_win, adjusted.away_win),
        ("Over 2.5 goals", base.over_2_5, adjusted.over_2_5),
        ("Both teams score", base.both_teams_to_score, adjusted.both_teams_to_score),
    ):
        rows.append(
            {"": label, "Model only": _pct(before), "With this note": _pct(after), "Change": _change_points(before, after)}
        )
    before_score, after_score = _score(base), _score(adjusted)
    rows.append(
        {
            "": "Most likely score",
            "Model only": before_score,
            "With this note": after_score,
            "Change": "no change" if before_score == after_score else "changed",
        }
    )
    if base.breakdown and adjusted.breakdown:
        for label, before, after in (
            ("Expected home goals", base.breakdown.lambda_home, adjusted.breakdown.lambda_home),
            ("Expected away goals", base.breakdown.lambda_away, adjusted.breakdown.lambda_away),
        ):
            diff = after - before
            rows.append(
                {
                    "": label,
                    "Model only": f"{before:.2f}",
                    "With this note": f"{after:.2f}",
                    "Change": "no change" if abs(diff) < 0.005 else f"{'+' if diff > 0 else '−'}{abs(diff):.2f}",
                }
            )
    return rows


def describe_injury_entry(player: str, team: str, position: str, weight: float) -> tuple[str, str]:
    """(headline, maths) for an availability entry, using the model's own constants."""
    headline = f"🩹 {player} ({position} player, {team}) is treated as unavailable for this match"
    if position == "defense":
        maths = (
            f"importance {weight:.2f} × {DEFENSE_IMPACT_FACTOR:g} → {team} concede "
            f"{_pct(weight * DEFENSE_IMPACT_FACTOR)} more goals"
        )
    else:
        cut = min(weight * ATTACK_IMPACT_FACTOR, 1 - MIN_MULTIPLIER)
        maths = f"importance {weight:.2f} × {ATTACK_IMPACT_FACTOR:g} → {team}'s attack −{_pct(cut)}"
    return headline, maths


def describe_form_entry(summary: str, team: str, magnitude: float, affects: str) -> tuple[str, str]:
    mood = "📈 good form" if magnitude > 0 else "📉 bad form" if magnitude < 0 else "➖ neutral"
    headline = f"{mood} ({team}): {summary.rstrip('. ')}"
    shift = max(-1.0, min(1.0, magnitude)) * FORM_IMPACT_CAP
    effects = []
    if affects in ("attack", "both"):
        effects.append(f"attack {_signed_pct(shift)}")
    if affects in ("defense", "both"):
        effects.append(f"goals conceded {_signed_pct(-shift)}")
    maths = (
        f"strength {_signed(magnitude)} × {FORM_IMPACT_CAP:.0%} cap → "
        f"{' and '.join(effects) or 'no effect'} (a team's form notes together are capped at ±{FORM_IMPACT_CAP:.0%})"
    )
    return headline, maths
