"""Offline job: fit Dixon-Coles for a league and persist it.

Run this after new results come in (roughly weekly) via scripts/run_training.py.
The dashboard never fits live -- it only reads the latest persisted row here,
so browsing fixtures/injuries stays fast (see prediction/service.py).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from soccer_predictor.model.dixon_coles import DixonColesParams, fit_league
from soccer_predictor.model.time_weighting import DEFAULT_XI
from soccer_predictor.storage.models import FittedParams
from soccer_predictor.storage.repository import matches_for_league


def train_league(session: Session, league_code: str, xi: float = DEFAULT_XI) -> DixonColesParams:
    matches = matches_for_league(session, league_code)
    params = fit_league(matches, league_code, xi=xi)

    session.add(
        FittedParams(
            league_code=league_code,
            fitted_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
            params_json=params.to_json(),
            time_window_desc=f"{matches.date.min()}..{matches.date.max()} ({len(matches)} matches)",
        )
    )
    return params
