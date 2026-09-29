"""Tracks how good this app's own predictions actually are.

A forward-snapshot system, not retrospective grading: the first time a
fixture is seen as upcoming with a trained model available, its predicted
scoreline is locked into `PredictionRecord` and never touched again, even
if the model is retrained before the match is played (see
storage.repository.insert_prediction_record). Once the real result lands in
the `matches` table, that locked-in snapshot can be graded against it.

Deliberately NOT retrospective (re-running the *current* model against
already-completed matches) -- Dixon-Coles is fit on the full match history,
so grading a match with a model partly trained on that match's own result
isn't a genuine blind-forecast test, just a fit-quality check. A real
snapshot, taken before the outcome was known, is the only honest way to
measure this.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.prediction.service import predict_fixture
from soccer_predictor.storage.repository import (
    all_prediction_records,
    fixtures_for_league,
    has_prediction_record,
    insert_prediction_record,
    matches_for_league,
    teams_for_league,
)

# Wide enough to mean "every upcoming fixture currently known," not the
# dashboard's usual 14-day display window -- this just needs to find rows
# to snapshot, not decide what's worth showing on a page.
SNAPSHOT_WINDOW_DAYS = 365


def snapshot_upcoming_predictions(session: Session, league: League, params: DixonColesParams) -> int:
    """Locks in a prediction for every fixture in `league` that doesn't
    already have one. Returns how many new snapshots were created.
    predict_fixture may fetch current-season stats (Understat/American
    Soccer Analysis, see model/current_form_adjustment.py) for leagues
    those cover -- disk-cached 24h, so only the first fixture per league
    in a given run costs a real fetch; every other lookup in this loop
    hits the same warm cache. Degrades to no adjustment (never raises) if
    that fetch fails, same contract as every other adjustment here.
    """
    team_names = teams_for_league(session, league.code)
    today = dt.date.today()
    upcoming = fixtures_for_league(session, league.code, today, today + dt.timedelta(days=SNAPSHOT_WINDOW_DAYS))

    created = 0
    for row in upcoming.itertuples(index=False):
        if has_prediction_record(session, league.code, row.date, row.home_team_id, row.away_team_id):
            continue
        home_name = team_names.get(row.home_team_id, f"team#{row.home_team_id}")
        away_name = team_names.get(row.away_team_id, f"team#{row.away_team_id}")
        prediction = predict_fixture(
            session, params, row.home_team_id, row.away_team_id, home_name, away_name, fixture_date=row.date
        )
        predicted_home_goals, predicted_away_goals, _ = prediction.top_scorelines[0]
        insert_prediction_record(
            session, league.code, row.date, row.home_team_id, row.away_team_id, predicted_home_goals, predicted_away_goals
        )
        created += 1
    return created


@dataclass
class PredictionAccuracy:
    exact: int = 0
    correct_outcome: int = 0
    wrong: int = 0
    pending: int = 0  # snapshotted but the match hasn't been played yet

    @property
    def graded_total(self) -> int:
        return self.exact + self.correct_outcome + self.wrong

    @property
    def hit_rate(self) -> float | None:
        """(exact + correct_outcome) / graded_total, or None if nothing's
        been graded yet -- callers must handle None explicitly rather than
        showing a misleading 0%."""
        if self.graded_total == 0:
            return None
        return (self.exact + self.correct_outcome) / self.graded_total


def _outcome_sign(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 1
    if home_goals < away_goals:
        return -1
    return 0


def compute_prediction_accuracy(session: Session) -> PredictionAccuracy:
    """Reconciles every locked-in PredictionRecord against the matches
    table, grouped by league (matches_for_league is per-league, so this
    loops the leagues actually represented in the snapshot data rather than
    requiring a new "all matches, every league" query).
    """
    records = all_prediction_records(session)
    accuracy = PredictionAccuracy()
    if not records:
        return accuracy

    records_by_league: dict[str, list] = {}
    for record in records:
        records_by_league.setdefault(record.league_code, []).append(record)

    for league_code, league_records in records_by_league.items():
        matches_df = matches_for_league(session, league_code)
        actual_by_key = {
            (row.date, row.home_team_id, row.away_team_id): (row.home_goals, row.away_goals)
            for row in matches_df.itertuples(index=False)
        }
        for record in league_records:
            actual = actual_by_key.get((record.date, record.home_team_id, record.away_team_id))
            if actual is None:
                accuracy.pending += 1
                continue
            actual_home_goals, actual_away_goals = actual
            if (
                record.predicted_home_goals == actual_home_goals
                and record.predicted_away_goals == actual_away_goals
            ):
                accuracy.exact += 1
            elif _outcome_sign(record.predicted_home_goals, record.predicted_away_goals) == _outcome_sign(
                actual_home_goals, actual_away_goals
            ):
                accuracy.correct_outcome += 1
            else:
                accuracy.wrong += 1

    return accuracy
