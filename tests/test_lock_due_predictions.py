"""lock_due_predictions: the final forecast is locked when the lineup is out (or,
failing that, just before kickoff), never earlier and never after kickoff."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import lineups
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.model.lineup_adjustment import ExpectedPlayer
from soccer_predictor.prediction import service, tracking
from soccer_predictor.prediction.tracking import (
    EARLY_BASIS,
    EARLY_FALLBACK_MINUTES,
    LINEUP_BASIS,
    LINEUP_WINDOW_MINUTES,
    has_real_kickoff,
    lock_due_predictions,
)
from soccer_predictor.storage.models import Base, PredictionRecord
from soccer_predictor.storage.repository import (
    all_prediction_records,
    get_or_create_team,
    insert_prediction_record,
    upsert_fixture,
    upsert_match,
)

LEAGUE = League(code="EPL", name="Premier League", seasons=["2627"])
MLS = League(code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1")
NOW = dt.datetime(2026, 10, 31, 14, 0)

SQUAD = [ExpectedPlayer("Keeper", "defense", 0.9, 1000, True)] + [
    ExpectedPlayer(f"Player {chr(65 + i)}", "attack", 0.2, 900 - i * 10) for i in range(10)
]
SAKA = ExpectedPlayer("Bukayo Saka", "attack", 0.5, 2000)
SAKA_BENCHED = [p.name for p in SQUAD]  # the usual XI minus Saka
SAKA_STARTS = [p.name for p in [SAKA, *SQUAD][:11]]  # the usual XI
OPPONENT_XI = [f"Villa Player {i}" for i in range(11)]  # a club with no stats


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(service, "resolve_current_attack_strength", lambda *a, **k: None)
    monkeypatch.setattr(service, "expected_squad", lambda name, league: [SAKA, *SQUAD] if name == "Arsenal" else [])
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture
def teams(session):
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    villa = get_or_create_team(session, "Aston Villa", "EPL")
    session.commit()
    return arsenal.id, villa.id


def _params(a: int, b: int, league_code: str = "EPL") -> DixonColesParams:
    return DixonColesParams(
        league_code=league_code,
        attack={a: 1.4, b: 0.9},
        defense={a: 1.0, b: 1.2},
        home_advantage=1.3,
        rho=-0.05,
        xi=0.0018,
        fitted_at="2026-10-30T00:00:00",
        n_matches=1000,
    )


def _xi(home_names, away_names=OPPONENT_XI):
    return lineups.ConfirmedXi("401", list(home_names), list(away_names), "4-3-3", "4-4-2", ("Arsenal", "Aston Villa"))


def _announces(xi):
    """A stand-in for ESPN that returns `xi` and records which matches it was asked about."""
    asked = []

    def named_xi(session, league, fixture, home_name, away_name):
        asked.append((home_name, away_name))
        return xi

    named_xi.asked = asked
    return named_xi


def _fixture(session, teams, kickoff, status="TIMED", league_code="EPL"):
    a, b = teams
    upsert_fixture(session, league_code, kickoff.date(), a, b, status, kickoff_utc=kickoff)
    session.commit()


def _lock(session, teams, named_xi, now=NOW, league=LEAGUE):
    created = lock_due_predictions(session, league, _params(*teams, league.code), now=now, named_xi=named_xi)
    session.commit()
    return created


# --- when a lock happens ------------------------------------------------------------------


def test_a_match_outside_the_lineup_window_is_left_alone(session, teams):
    _fixture(session, teams, NOW + dt.timedelta(minutes=LINEUP_WINDOW_MINUTES + 30))
    named_xi = _announces(_xi(SAKA_STARTS))

    assert _lock(session, teams, named_xi) == 0
    assert all_prediction_records(session) == []
    assert named_xi.asked == []  # not even looked up


def test_an_announced_lineup_locks_the_forecast_with_that_lineup(session, teams):
    _fixture(session, teams, NOW + dt.timedelta(minutes=70))

    assert _lock(session, teams, _announces(_xi(SAKA_BENCHED))) == 1

    [record] = all_prediction_records(session)
    assert record.basis == LINEUP_BASIS
    assert record.p_home is not None and record.p_draw is not None and record.p_away is not None
    saved = json.loads(record.lineup_json)
    assert saved["home"] == SAKA_BENCHED and saved["away"] == OPPONENT_XI
    assert [a["name"] for a in saved["home_absent"]] == ["Bukayo Saka"]
    assert saved["away_absent"] == []  # no stats for Villa: nothing to compare
    assert saved["home_formation"] == "4-3-3"


def test_the_lineup_changes_the_locked_probabilities(session, teams):
    """Saka on the bench lowers Arsenal's chance compared with the usual XI."""
    _fixture(session, teams, NOW + dt.timedelta(minutes=70))
    _lock(session, teams, _announces(_xi(SAKA_BENCHED)))
    benched = all_prediction_records(session)[0].p_home

    session.query(PredictionRecord).delete()
    session.commit()
    _lock(session, teams, _announces(_xi(SAKA_STARTS)))
    assert benched < all_prediction_records(session)[0].p_home


def test_no_lineup_yet_means_wait_until_just_before_kickoff(session, teams):
    _fixture(session, teams, NOW + dt.timedelta(minutes=60))
    assert _lock(session, teams, _announces(None)) == 0
    assert all_prediction_records(session) == []


def test_no_lineup_by_the_fallback_time_locks_the_early_forecast(session, teams):
    _fixture(session, teams, NOW + dt.timedelta(minutes=EARLY_FALLBACK_MINUTES - 5))

    assert _lock(session, teams, _announces(None)) == 1

    [record] = all_prediction_records(session)
    assert record.basis == EARLY_BASIS and record.lineup_json is None
    assert record.p_home is not None


def test_a_lineup_that_cannot_be_rated_locks_as_early_straight_away(session, teams, monkeypatch):
    # neither club has player stats here, so the XI would change nothing
    monkeypatch.setattr(service, "expected_squad", lambda name, league: [])
    _fixture(session, teams, NOW + dt.timedelta(minutes=70))

    assert _lock(session, teams, _announces(_xi([f"Arsenal Player {i}" for i in range(11)]))) == 1

    [record] = all_prediction_records(session)
    assert record.basis == EARLY_BASIS and record.lineup_json is None


def test_a_match_that_has_started_is_never_locked(session, teams):
    _fixture(session, teams, NOW - dt.timedelta(minutes=10))
    assert _lock(session, teams, _announces(_xi(SAKA_STARTS))) == 0
    assert all_prediction_records(session) == []


def test_a_lock_is_made_once_and_never_changed(session, teams):
    _fixture(session, teams, NOW + dt.timedelta(minutes=70))
    assert _lock(session, teams, _announces(_xi(SAKA_BENCHED))) == 1
    before = all_prediction_records(session)[0]
    saved = (before.p_home, before.lineup_json)

    # a later run sees a different XI (late change): the lock stands
    assert _lock(session, teams, _announces(_xi(SAKA_STARTS)), now=NOW + dt.timedelta(minutes=20)) == 0
    after = all_prediction_records(session)
    assert len(after) == 1 and (after[0].p_home, after[0].lineup_json) == saved


def test_a_club_the_model_has_no_rating_for_is_skipped(session, teams):
    a, b = teams
    _fixture(session, teams, NOW + dt.timedelta(minutes=70))
    params = _params(a, b)
    del params.attack[b]
    assert lock_due_predictions(session, LEAGUE, params, now=NOW, named_xi=_announces(_xi(SAKA_STARTS))) == 0


# --- which kickoff times count -----------------------------------------------------------


def test_a_placeholder_midnight_kickoff_is_not_a_real_kickoff(session, teams):
    # football-data.org: a SCHEDULED fixture is "00:00" until a time is set
    late = dt.datetime(2026, 10, 31, 23, 30)
    _fixture(session, teams, dt.datetime(2026, 11, 1, 0, 0), status="SCHEDULED")

    assert _lock(session, teams, _announces(_xi(SAKA_STARTS)), now=late) == 0

    _fixture(session, teams, dt.datetime(2026, 11, 1, 0, 0), status="TIMED")  # the same match, now with a real time
    assert _lock(session, teams, _announces(_xi(SAKA_STARTS)), now=late) == 1


def test_an_espn_leagues_midnight_utc_kickoff_is_real(session):
    home, away = get_or_create_team(session, "Arsenal", "MLS"), get_or_create_team(session, "Aston Villa", "MLS")
    session.commit()
    teams = (home.id, away.id)
    late = dt.datetime(2026, 10, 31, 23, 30)
    _fixture(session, teams, dt.datetime(2026, 11, 1, 0, 0), status="SCHEDULED", league_code="MLS")

    assert _lock(session, teams, _announces(_xi(SAKA_STARTS)), now=late, league=MLS) == 1


def test_has_real_kickoff():
    class Row:
        def __init__(self, kickoff, status):
            self.kickoff_utc, self.status = kickoff, status

    midnight = dt.datetime(2026, 11, 1)
    assert has_real_kickoff(LEAGUE, Row(dt.datetime(2026, 11, 1, 15, 0), "TIMED"))
    assert not has_real_kickoff(LEAGUE, Row(midnight, "SCHEDULED"))
    assert not has_real_kickoff(LEAGUE, Row(None, "TIMED"))
    assert not has_real_kickoff(LEAGUE, Row(float("nan"), "TIMED"))
    assert has_real_kickoff(MLS, Row(midnight, "SCHEDULED"))


# --- records locked by the old 48-hour rule ------------------------------------------------


def test_an_old_rule_lock_for_an_unplayed_match_is_replaced_at_lineup_time(session, teams):
    a, b = teams
    kickoff = NOW + dt.timedelta(minutes=70)
    _fixture(session, teams, kickoff)
    insert_prediction_record(session, "EPL", kickoff.date(), a, b, 1, 0, p_home=0.5, p_draw=0.3, p_away=0.2)
    session.commit()

    assert _lock(session, teams, _announces(_xi(SAKA_BENCHED))) == 1

    [record] = all_prediction_records(session)
    assert record.basis == LINEUP_BASIS


def test_an_old_rule_lock_stays_while_the_lineup_is_still_to_come(session, teams):
    a, b = teams
    kickoff = NOW + dt.timedelta(minutes=70)
    _fixture(session, teams, kickoff)
    insert_prediction_record(session, "EPL", kickoff.date(), a, b, 1, 0, p_home=0.5, p_draw=0.3, p_away=0.2)
    session.commit()

    assert _lock(session, teams, _announces(None)) == 0
    assert all_prediction_records(session)[0].basis is None


def test_a_graded_old_lock_is_never_touched(session, teams):
    a, b = teams
    played = NOW - dt.timedelta(days=3)
    insert_prediction_record(session, "EPL", played.date(), a, b, 1, 1, p_home=0.4, p_draw=0.3, p_away=0.3)
    upsert_match(session, "EPL", "2627", played.date(), a, b, 1, 1)
    session.commit()

    assert _lock(session, teams, _announces(_xi(SAKA_STARTS))) == 0
    [record] = all_prediction_records(session)
    assert record.basis is None and record.p_home == 0.4


# --- reading the lineup from ESPN ---------------------------------------------------------


class _Row:
    date = dt.date(2026, 10, 31)
    home_team_id, away_team_id = 1, 2


def test_a_failing_lineup_lookup_means_no_lineup_not_a_crash(session, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("ESPN is down")

    monkeypatch.setattr(lineups, "find_event", boom)
    assert tracking._named_xi(session, MLS, _Row(), "Arsenal", "Aston Villa") is None


def test_the_stored_espn_event_id_is_used_to_fetch_the_lineup(session, teams, monkeypatch):
    a, b = teams
    seen = {}
    upsert_fixture(session, "EPL", NOW.date(), a, b, "TIMED", kickoff_utc=NOW + dt.timedelta(minutes=70), espn_event_id="777")
    session.commit()
    xi = _xi(SAKA_STARTS)
    monkeypatch.setattr(lineups, "find_event", lambda league, d, h, aw, stored=None: seen.setdefault("stored", stored))
    monkeypatch.setattr(lineups, "fetch_confirmed_xi", lambda league, event_id, h, aw: xi if event_id == "777" else None)

    row = type("R", (), {"date": NOW.date(), "home_team_id": a, "away_team_id": b})()
    assert tracking._named_xi(session, LEAGUE, row, "Arsenal", "Aston Villa") is xi
    assert seen["stored"] == "777"
