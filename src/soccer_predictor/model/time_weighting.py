"""Exponential recency weighting, standard in Dixon-Coles fits.

Older matches still inform the fit but contribute less, so squad/form
changes show up without needing a separate hand-built "form" feature.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

DEFAULT_XI = 0.0018  # ~half-life of ~1 year; matches values used in DC literature.


def match_weights(
    dates: pd.Series, as_of: dt.date | None = None, xi: float = DEFAULT_XI
) -> np.ndarray:
    as_of = as_of or dates.max()
    days_since = (pd.Timestamp(as_of) - pd.to_datetime(dates)).dt.days.to_numpy()
    days_since = np.clip(days_since, a_min=0, a_max=None)
    return np.exp(-xi * days_since)
