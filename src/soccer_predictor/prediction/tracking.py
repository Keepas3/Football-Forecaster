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
import math
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
    prediction_records_missing_probabilities,
    teams_for_league,
)

# Wide enough to mean "every upcoming fixture currently known," not the
# dashboard's usual 14-day display window -- this just needs to find rows
# to snapshot, not decide what's worth showing on a page.
SNAPSHOT_WINDOW_DAYS = 365


def _already_kicked_off(kickoff_utc, now: dt.datetime) -> bool:
    """True when a kickoff time is known and has passed. A prediction made
    after kickoff is no longer a forecast, so snapshots and probability
    backfills skip such matches even if no result is stored yet (still in
    progress). The model never reads live scores, so this is about keeping
    the record honest, not about leaked information."""
    return kickoff_utc is not None and not _is_missing(kickoff_utc) and kickoff_utc <= now


def _is_missing(value) -> bool:
    # pandas gives NaT/NaN for an empty kickoff column
    return value != value


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

    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    created = 0
    for row in upcoming.itertuples(index=False):
        if _already_kicked_off(row.kickoff_utc, now):
            continue
        if has_prediction_record(session, league.code, row.date, row.home_team_id, row.away_team_id):
            continue
        home_name = team_names.get(row.home_team_id, f"team#{row.home_team_id}")
        away_name = team_names.get(row.away_team_id, f"team#{row.away_team_id}")
        prediction = predict_fixture(
            session, params, row.home_team_id, row.away_team_id, home_name, away_name, fixture_date=row.date
        )
        predicted_home_goals, predicted_away_goals, _ = prediction.top_scorelines[0]
        insert_prediction_record(
            session,
            league.code,
            row.date,
            row.home_team_id,
            row.away_team_id,
            predicted_home_goals,
            predicted_away_goals,
            p_home=prediction.home_win,
            p_draw=prediction.draw,
            p_away=prediction.away_win,
        )
        created += 1
    return created


def backfill_prediction_probabilities(session: Session, league: League, params: DixonColesParams) -> int:
    """Saves win/draw/loss probabilities on records locked in before they were
    stored -- but only for matches that have NOT been played yet, so the
    probabilities are still a genuine forecast made before the result. (Doing
    it for a finished match would mean scoring a model that has already seen
    the outcome.) The locked-in scoreline itself is never touched. Returns how
    many records were filled in."""
    team_names = teams_for_league(session, league.code)
    # A match dated today may already be over: skip anything with a result.
    played = {
        (row.date, row.home_team_id, row.away_team_id)
        for row in matches_for_league(session, league.code).itertuples(index=False)
    }
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    today = dt.date.today()
    kickoff_by_match = {
        (row.date, row.home_team_id, row.away_team_id): row.kickoff_utc
        for row in fixtures_for_league(session, league.code, today, today + dt.timedelta(days=SNAPSHOT_WINDOW_DAYS)).itertuples(index=False)
    }
    filled = 0
    for record in prediction_records_missing_probabilities(session, league.code, today):
        key = (record.date, record.home_team_id, record.away_team_id)
        if key in played or _already_kicked_off(kickoff_by_match.get(key), now):
            continue
        if record.date == today and kickoff_by_match.get(key) is None:
            continue  # today's match with no known kickoff: can't prove it hasn't started
        prediction = predict_fixture(
            session,
            params,
            record.home_team_id,
            record.away_team_id,
            team_names.get(record.home_team_id, f"team#{record.home_team_id}"),
            team_names.get(record.away_team_id, f"team#{record.away_team_id}"),
            fixture_date=record.date,
        )
        record.p_home, record.p_draw, record.p_away = prediction.home_win, prediction.draw, prediction.away_win
        filled += 1
    return filled


# What "no skill" looks like: just guessing the usual football rates (home win,
# draw, away win -- measured on 3,500 league matches), the same every match.
NO_SKILL_PROBS = (0.43, 0.25, 0.32)


@dataclass
class PredictionAccuracy:
    exact: int = 0
    correct_outcome: int = 0
    wrong: int = 0
    pending: int = 0  # snapshotted but the match hasn't been played yet
    # Graded predictions that also have saved probabilities, and their summed
    # scores (see _rps / _log_loss) next to the same sums for NO_SKILL_PROBS.
    scored: int = 0
    rps_total: float = 0.0
    log_loss_total: float = 0.0
    baseline_rps_total: float = 0.0
    baseline_log_loss_total: float = 0.0

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

    @property
    def mean_rps(self) -> float | None:
        return self.rps_total / self.scored if self.scored else None

    @property
    def baseline_rps(self) -> float | None:
        return self.baseline_rps_total / self.scored if self.scored else None

    @property
    def mean_log_loss(self) -> float | None:
        return self.log_loss_total / self.scored if self.scored else None

    @property
    def baseline_log_loss(self) -> float | None:
        return self.baseline_log_loss_total / self.scored if self.scored else None


def _outcome_sign(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 1
    if home_goals < away_goals:
        return -1
    return 0


def most_likely_outcome(record) -> int:
    """The result the model thought most likely: 1 home win, 0 draw, -1 away
    win. From the saved probabilities when there are any; for older records
    (no probabilities) from the locked-in scoreline.

    The scoreline alone is a poor stand-in: the single likeliest scoreline is
    very often 0-0 or 1-1 even when a win is the likeliest result (it has to
    split a win's probability across 1-0, 2-0, 2-1...), so grading it counted
    "draw predicted" for matches the model actually favoured one side in."""
    if record.p_home is not None and record.p_draw is not None and record.p_away is not None:
        probs = {1: record.p_home, 0: record.p_draw, -1: record.p_away}
        return max(probs, key=probs.get)
    return _outcome_sign(record.predicted_home_goals, record.predicted_away_goals)


def _rps(probs: tuple[float, float, float], actual: int) -> float:
    """Ranked probability score for (home, draw, away) probabilities: 0 is
    perfect, lower is better. Unlike right/wrong it rewards a forecast for
    putting probability near the real result (a 45% draw call on a 1-0 home win
    is far less wrong than a 90% away call)."""
    observed = (1.0, 0.0, 0.0) if actual == 1 else (0.0, 1.0, 0.0) if actual == 0 else (0.0, 0.0, 1.0)
    cum_p1, cum_o1 = probs[0], observed[0]
    cum_p2, cum_o2 = probs[0] + probs[1], observed[0] + observed[1]
    return 0.5 * ((cum_p1 - cum_o1) ** 2 + (cum_p2 - cum_o2) ** 2)


def _log_loss(probs: tuple[float, float, float], actual: int) -> float:
    p = probs[0] if actual == 1 else probs[1] if actual == 0 else probs[2]
    return -math.log(max(p, 1e-9))


def compute_prediction_accuracy(session: Session) -> PredictionAccuracy:
    """Reconciles every locked-in PredictionRecord against the matches
    table, grouped by league (matches_for_league is per-league, so this
    loops the leagues actually represented in the snapshot data rather than
    requiring a new "all matches, every league" query).

    A graded record is "exact" if the scoreline matched, else "correct
    outcome" if the result the model thought most likely (most_likely_outcome)
    happened, else "wrong". Records with saved probabilities are also scored
    with RPS and log loss against the no-skill guess.
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
            actual_outcome = _outcome_sign(actual_home_goals, actual_away_goals)
            if (
                record.predicted_home_goals == actual_home_goals
                and record.predicted_away_goals == actual_away_goals
            ):
                accuracy.exact += 1
            elif most_likely_outcome(record) == actual_outcome:
                accuracy.correct_outcome += 1
            else:
                accuracy.wrong += 1

            if record.p_home is not None and record.p_draw is not None and record.p_away is not None:
                probs = (record.p_home, record.p_draw, record.p_away)
                accuracy.scored += 1
                accuracy.rps_total += _rps(probs, actual_outcome)
                accuracy.log_loss_total += _log_loss(probs, actual_outcome)
                accuracy.baseline_rps_total += _rps(NO_SKILL_PROBS, actual_outcome)
                accuracy.baseline_log_loss_total += _log_loss(NO_SKILL_PROBS, actual_outcome)
    return accuracy
