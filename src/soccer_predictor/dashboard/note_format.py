"""Plain-English wording for the saved chat notes shown under "Active
chat-sourced notes": what the note is, on one line, and the maths it feeds
into the prediction on another -- using the same constants the model itself
applies (model/form_adjustment.py, model/injury_adjustment.py), so the
numbers shown are the numbers used.

Pure functions, no Streamlit.
"""

from __future__ import annotations

import datetime as dt

from soccer_predictor.model.form_adjustment import FORM_IMPACT_CAP
from soccer_predictor.model.injury_adjustment import (
    ATTACK_IMPACT_FACTOR,
    DEFENSE_IMPACT_FACTOR,
    MIN_MULTIPLIER,
)


def _date(d: dt.date) -> str:
    return d.strftime("%b %d, %Y").replace(" 0", " ")


def _countdown(target: dt.date, today: dt.date) -> str:
    days = (target - today).days
    if days <= 0:
        return "today"
    if days == 1:
        return "tomorrow"
    return f"in {days} days"


def _pct(fraction: float) -> str:
    """+9.0% / −9.0% (a real minus sign, so the direction is unmistakable)."""
    return f"{'+' if fraction >= 0 else '−'}{abs(fraction) * 100:.1f}%"


def describe_form_note(summary: str, magnitude: float, affects: str, expires_on: dt.date, today: dt.date) -> tuple[str, str]:
    """(headline, maths) for a team form note.

    The model turns a form note into a shift of `magnitude × 15%` on attack
    and/or defence (all of a team's notes together are capped at ±15%).
    Defence is a goals-conceded multiplier, so bad form makes it concede MORE
    -- shown as such rather than as a confusing negative.
    """
    mood = "📈 Good form" if magnitude > 0 else "📉 Bad form" if magnitude < 0 else "➖ Neutral"
    headline = f"**Form note** · {mood} — {summary.rstrip('. ')}"

    shift = max(-1.0, min(1.0, magnitude)) * FORM_IMPACT_CAP
    effects = []
    if affects in ("attack", "both"):
        effects.append(f"attack strength {_pct(shift)}")
    if affects in ("defense", "both"):
        effects.append(f"goals conceded {_pct(-shift)}")
    effect_text = " and ".join(effects) if effects else "no effect"

    maths = (
        f"Maths: strength {f'{magnitude:+.2f}'.replace('-', '−')} × {FORM_IMPACT_CAP:.0%} cap → {effect_text} "
        f"(all of a team's form notes together are capped at ±{FORM_IMPACT_CAP:.0%}) · "
        f"Expires {_date(expires_on)} ({_countdown(expires_on, today)})"
    )
    return headline, maths


def describe_injury(
    player_name: str,
    position: str,
    importance_weight: float,
    expected_return_date: dt.date | None,
    today: dt.date,
) -> tuple[str, str]:
    """(headline, maths) for a player availability note."""
    if expected_return_date is None:
        status = "no return date set, so it counts until you delete it"
    else:
        status = f"expected back {_date(expected_return_date)} ({_countdown(expected_return_date, today)})"
    headline = f"**Injury** · 🩹 {player_name} ({position} player) — {status}"

    if position == "defense":
        hit = importance_weight * DEFENSE_IMPACT_FACTOR
        effect_text = f"goals conceded {_pct(hit)}"
        factor = DEFENSE_IMPACT_FACTOR
    else:
        hit = importance_weight * ATTACK_IMPACT_FACTOR
        effect_text = f"attack strength {_pct(-min(hit, 1 - MIN_MULTIPLIER))}"
        factor = ATTACK_IMPACT_FACTOR

    maths = (
        f"Maths: importance {importance_weight:.2f} × {factor:g} → {effect_text} while they're out "
        f"(a team's rating can't be cut by more than {1 - MIN_MULTIPLIER:.0%} in total)"
    )
    return headline, maths
