"""Tracks how good this app's own predictions actually are.

A forward-snapshot system, not retrospective grading: a match's final
forecast is locked into `PredictionRecord` when its starting XI is announced
(usually 60-75 minutes before kickoff) and never touched again, even if the
model is retrained before the match is played (see
storage.repository.insert_prediction_record). Once the real result lands in
the `matches` table, that locked-in snapshot can be graded against it. Until
the lineup is out the dashboard shows the injury-based forecast as "Early".

Deliberately NOT retrospective (re-running the *current* model against
already-completed matches) -- Dixon-Coles is fit on the full match history,
so grading a match with a model partly trained on that match's own result
isn't a genuine blind-forecast test, just a fit-quality check. A real
snapshot, taken before the outcome was known, is the only honest way to
measure this.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import lineups
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.prediction.service import predict_fixture
from soccer_predictor.storage.repository import (
    all_prediction_records,
    espn_event_id_for_match,
    fixtures_for_league,
    insert_prediction_record,
    matches_for_league,
    prediction_record_for_match,
    prediction_records_missing_probabilities,
    teams_for_league,
)

logger = logging.getLogger(__name__)

# The final forecast waits for the announced starting XI. Clubs name it about an
# hour before kickoff, so from this long before kickoff the lock job looks for it
# on every run and locks the moment it appears.
LINEUP_WINDOW_MINUTES = 90
# A match with still no XI this close to kickoff (no feed, a late team sheet, a
# competition ESPN does not cover) is locked on its early, injury-based forecast
# instead, so every match is still graded. Wide enough for two runs of a job that
# is pinged every ten minutes.
EARLY_FALLBACK_MINUTES = 20
LINEUP_BASIS = "lineup"
EARLY_BASIS = "early"
# Wide enough to mean "every upcoming fixture currently known" -- used where a
# lookup just has to find the rows, not decide what gets locked in.
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


def has_real_kickoff(league: League, fixture) -> bool:
    """Whether `fixture`'s kickoff time is a real one. football-data.org gives a
    fixture it has not scheduled yet (status SCHEDULED) a placeholder 00:00 UTC
    kickoff, and only a TIMED one a real time; ESPN's times are always real (an MLS
    match at 00:00 UTC is simply an 8 pm Eastern kickoff)."""
    if _is_missing(fixture.kickoff_utc) or fixture.kickoff_utc is None:
        return False
    return league.data_source == "espn" or fixture.status != "SCHEDULED"


def _named_xi(session: Session, league: League, fixture, home_name: str, away_name: str):
    """The starting XI ESPN lists for `fixture` right now, or None. Never raises:
    one match's feed trouble must not stop the others from being locked."""
    try:
        stored = espn_event_id_for_match(
            session, league.code, fixture.date, fixture.home_team_id, fixture.away_team_id
        )
        event_id = lineups.find_event(league, fixture.date, home_name, away_name, stored)
        return lineups.fetch_confirmed_xi(league, event_id, home_name, away_name) if event_id else None
    except Exception:  # noqa: BLE001 -- the lineup is a bonus; the early forecast stands without it
        logger.exception("Could not read the lineup for %s v %s", home_name, away_name)
        return None


def lineup_json(xi: lineups.ConfirmedXi, breakdown) -> str:
    """What is saved with a lineup lock: both XIs and who was missing from each
    side's usual XI, so the lock can be shown and audited later."""
    return json.dumps(
        {
            "event_id": xi.event_id,
            "home": xi.home,
            "away": xi.away,
            "home_formation": xi.home_formation,
            "away_formation": xi.away_formation,
            "home_absent": [asdict(a) for a in breakdown.home_lineup_absences or []],
            "away_absent": [asdict(a) for a in breakdown.away_lineup_absences or []],
        }
    )


def stored_lineup(record) -> tuple[list[str] | None, list[str] | None]:
    """(home XI, away XI) saved with a lineup lock, or (None, None) for any other
    record or an unreadable one -- so a locked match's workings can be shown for the
    lineup it was locked with."""
    if record is None or record.basis != LINEUP_BASIS or not record.lineup_json:
        return None, None
    try:
        saved = json.loads(record.lineup_json)
        return list(saved["home"]), list(saved["away"])
    except (ValueError, KeyError, TypeError):
        return None, None


def lock_due_predictions(
    session: Session,
    league: League,
    params: DixonColesParams,
    now: dt.datetime | None = None,
    named_xi: Callable | None = None,
) -> int:
    """Locks in the final forecast of every match in `league` that kicks off within
    LINEUP_WINDOW_MINUTES and has none yet. Returns how many locks were made.

    Each such match is looked up on ESPN. With both starting XIs named, the forecast
    is made from them (basis "lineup", see prediction.service.predict_fixture) and
    locked at once. With none yet the match is left alone until
    EARLY_FALLBACK_MINUTES before kickoff, when the early forecast is locked instead
    (basis "early"); a lineup that cannot be rated (no player stats for either club)
    is locked the same way, since it changes nothing. A match locked by the older
    48-hour rule that has not kicked off is replaced; one already played, or whose
    kickoff time is still a placeholder, is never touched, and neither is a match
    that has started: a forecast made after kickoff is not a forecast.

    predict_fixture may fetch current-season stats (Understat/American Soccer
    Analysis, see model/current_form_adjustment.py), disk-cached 24h, and degrades
    to no adjustment (never raises) if that fails.

    `now` and `named_xi` (called with the session, league, fixture row, home and
    away name) exist for tests; the defaults are the real clock and ESPN.
    """
    now = now or dt.datetime.now(dt.UTC).replace(tzinfo=None)
    named_xi = named_xi or _named_xi
    window_end = now + dt.timedelta(minutes=LINEUP_WINDOW_MINUTES)
    team_names = teams_for_league(session, league.code)
    upcoming = fixtures_for_league(session, league.code, now.date(), window_end.date())

    created = 0
    for row in upcoming.itertuples(index=False):
        if not has_real_kickoff(league, row) or not now < row.kickoff_utc <= window_end:
            continue
        if row.home_team_id not in params.attack or row.away_team_id not in params.attack:
            continue  # a club the model has no rating for yet
        existing = prediction_record_for_match(session, league.code, row.date, row.home_team_id, row.away_team_id)
        if existing is not None and existing.basis is not None:
            continue
        home_name = team_names.get(row.home_team_id, f"team#{row.home_team_id}")
        away_name = team_names.get(row.away_team_id, f"team#{row.away_team_id}")
        xi = named_xi(session, league, row, home_name, away_name)
        if xi is None and (row.kickoff_utc - now) > dt.timedelta(minutes=EARLY_FALLBACK_MINUTES):
            continue  # the lineup may still come
        prediction = predict_fixture(
            session,
            params,
            row.home_team_id,
            row.away_team_id,
            home_name,
            away_name,
            fixture_date=row.date,
            home_lineup=xi.home if xi else None,
            away_lineup=xi.away if xi else None,
        )
        breakdown = prediction.breakdown
        rated = xi is not None and (
            breakdown.home_lineup_absences is not None or breakdown.away_lineup_absences is not None
        )
        predicted_home_goals, predicted_away_goals, _ = prediction.top_scorelines[0]
        if insert_prediction_record(
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
            basis=LINEUP_BASIS if rated else EARLY_BASIS,
            lineup_json=lineup_json(xi, breakdown) if rated else None,
            replace_older_rule=True,
        ):
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
    today = now.date()  # UTC, like kickoff times and fixture dates (not the machine's local date)
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


# What a locked record was made from, in the order the Track Record lists them:
# the announced lineup, the early forecast locked for want of one, and the
# forecast the older rule locked 48 hours ahead.
LOCK_GROUPS = (LINEUP_BASIS, EARLY_BASIS, "legacy")


def lock_group(record) -> str:
    return record.basis if record.basis in (LINEUP_BASIS, EARLY_BASIS) else "legacy"


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
    # The same figures for each kind of lock (LOCK_GROUPS keys), so the lineup
    # forecasts can be told apart from the no-lineup and older 48-hour ones.
    groups: dict[str, PredictionAccuracy] = field(default_factory=dict)

    def record_result(self, record, actual: tuple[int, int] | None) -> None:
        """Counts one locked-in record: pending without a result, else graded
        (exact / right outcome / wrong) and, with saved probabilities, scored."""
        if actual is None:
            self.pending += 1
            return
        actual_home_goals, actual_away_goals = actual
        actual_outcome = _outcome_sign(actual_home_goals, actual_away_goals)
        if record.predicted_home_goals == actual_home_goals and record.predicted_away_goals == actual_away_goals:
            self.exact += 1
        elif most_likely_outcome(record) == actual_outcome:
            self.correct_outcome += 1
        else:
            self.wrong += 1

        if record.p_home is not None and record.p_draw is not None and record.p_away is not None:
            probs = (record.p_home, record.p_draw, record.p_away)
            self.scored += 1
            self.rps_total += _rps(probs, actual_outcome)
            self.log_loss_total += _log_loss(probs, actual_outcome)
            self.baseline_rps_total += _rps(NO_SKILL_PROBS, actual_outcome)
            self.baseline_log_loss_total += _log_loss(NO_SKILL_PROBS, actual_outcome)

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
            group = accuracy.groups.setdefault(lock_group(record), PredictionAccuracy())
            accuracy.record_result(record, actual)
            group.record_result(record, actual)
    return accuracy
