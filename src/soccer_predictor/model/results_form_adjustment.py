"""Nudges a team's rating for how its last few results compare with an average side's.

The fitted ratings average a team's whole history, with older matches counting
for less (a half-life of about a year), so they are slow to notice a run of good
or bad results. A walk-forward test found that a team's points from its last six
league matches still say something the ratings do not: over 5,566 matches in the
five big European leagues since August 2023, scaling attack and goals conceded by
5% for every point per match above or below the league average improved the
ranked probability score by 0.0006 (95% interval 0.0002 to 0.0010), and the same
setting, chosen on those leagues, did as well on 2,025 MLS matches (0.0008). It
is a real effect but a very small one: it was not better in every league or
season (Ligue 1 and 2025-26 were slightly worse), and a team on 0 points from 6
is rated only about 7% down. Longer memory for everyone made forecasts worse, so
this is deliberately a small add-on, not a re-fit.

The size is capped by the arithmetic: 0 to 18 points from 6 matches is at most
about 8% either way. It overlaps a little with the current-season attacking
output nudge (model/current_form_adjustment.py), which cannot be backtested.

Only results in the same competition count, and only a full run of six that
started no more than MAX_SPAN_DAYS before the match, so a summer break or a
sparse fixture list does not stretch "recent" over most of a year.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

RECENT_MATCHES = 6
AVERAGE_POINTS_PER_MATCH = 1.375  # a team's mean over a league season: (3 x decisive + 2 x draws) / 2
RESULTS_FORM_STRENGTH = 0.05  # rating change per point per match above (below) the average
MAX_SPAN_DAYS = 130  # the oldest of the six results is at most this many days before the match


@dataclass(frozen=True)
class ResultsForm:
    points: int  # points won in the `matches` latest matches (3 a win, 1 a draw)
    matches: int
    shift: float  # fractional rating change: attack up by this, goals conceded down by about this


def results_form(
    results: Sequence[tuple[dt.date, int]], fixture_date: dt.date
) -> ResultsForm | None:
    """`results` is (date, points won) newest first. None unless there are
    RECENT_MATCHES of them, all recent enough to count as form."""
    latest = list(results)[:RECENT_MATCHES]
    if len(latest) < RECENT_MATCHES or (fixture_date - latest[-1][0]).days > MAX_SPAN_DAYS:
        return None
    points = sum(p for _, p in latest)
    shift = RESULTS_FORM_STRENGTH * (points / RECENT_MATCHES - AVERAGE_POINTS_PER_MATCH)
    return ResultsForm(points=points, matches=RECENT_MATCHES, shift=shift)


def adjust_for_results_form(
    base_attack: float, base_defense: float, form: ResultsForm | None
) -> tuple[float, float]:
    """Good form lifts attack and tightens defense (defense is a goals-conceded
    multiplier, so it is divided); poor form does the reverse. No form leaves
    the rating untouched, like every other adjustment here."""
    if form is None:
        return base_attack, base_defense
    return base_attack * (1.0 + form.shift), base_defense / (1.0 + form.shift)
