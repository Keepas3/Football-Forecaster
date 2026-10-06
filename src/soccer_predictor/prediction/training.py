"""Offline job: fit Dixon-Coles for a league and persist it.

Run this after new results come in (roughly weekly) via scripts/run_training.py.
The dashboard never fits live -- it only reads the latest persisted row here,
so browsing fixtures/injuries stays fast (see prediction/service.py).
"""

from __future__ import annotations

import datetime as dt
import re

from sqlalchemy.orm import Session

from soccer_predictor.config import League, load_leagues
from soccer_predictor.model.dixon_coles import DixonColesParams, fit_league
from soccer_predictor.model.time_weighting import DEFAULT_XI
from soccer_predictor.storage.models import FittedParams
from soccer_predictor.storage.repository import matches_for_league, team_group_names_for_season

# Nations League divisions, strongest first. A team missing from the latest
# edition (e.g. one that has since left the competition) is treated as League C.
DIVISION_LETTERS = "ABCD"
DEFAULT_DIVISION = 2

_DIVISION_RE = re.compile(r"League ([A-D])|Group ([A-D])\d")


def division_index(group_name: str | None) -> int | None:
    """0-3 for League A-D from a group label -- "Group B2" and "League D -
    Group 1" (how ESPN labels the 2018-19 edition) both work -- or None for
    a label that names no division."""
    if not isinstance(group_name, str) or not group_name:
        return None  # also a pandas NaN from an untagged (knockout) match
    match = _DIVISION_RE.search(group_name)
    if match is None:
        return None
    return DIVISION_LETTERS.index(match.group(1) or match.group(2))


def division_groups(session: Session, league: League) -> dict[int, int]:
    """{team_id: division index} for the league's latest edition."""
    groups = team_group_names_for_season(session, league.code, league.seasons[-1])
    divisions = {team_id: division_index(name) for team_id, name in groups.items()}
    return {team_id: division for team_id, division in divisions.items() if division is not None}


def train_league(session: Session, league_code: str, xi: float | None = None) -> DixonColesParams:
    """Fits and stores the league's ratings. How -- memory length, smoothing,
    division prior -- comes from the league's config (see League.rating_decay_xi
    and friends), so a competition whose teams play rarely can be tuned
    without touching the leagues that are fine as they are. `xi` overrides the
    memory length explicitly."""
    league = load_leagues().get(league_code)
    if xi is None:
        xi = (league.rating_decay_xi if league else None) or DEFAULT_XI
    ridge = league.rating_ridge if league else 0.0
    team_groups = division_groups(session, league) if league and league.division_prior else None

    matches = matches_for_league(session, league_code)
    params = fit_league(
        matches,
        league_code,
        xi=xi,
        ridge=ridge,
        team_groups=team_groups,
        default_group=DEFAULT_DIVISION,
    )

    session.add(
        FittedParams(
            league_code=league_code,
            fitted_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
            params_json=params.to_json(),
            time_window_desc=f"{matches.date.min()}..{matches.date.max()} ({len(matches)} matches)",
        )
    )
    return params
