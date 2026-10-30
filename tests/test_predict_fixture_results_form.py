"""A side's last six results nudge its rating in predict_fixture, only in the
leagues that use them, and only from matches played before the fixture."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.model.results_form_adjustment import (
    AVERAGE_POINTS_PER_MATCH,
    RESULTS_FORM_STRENGTH,
)
from soccer_predictor.prediction.service import predict_fixture
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, recent_results_for_team, upsert_match

FIXTURE_DAY = dt.date(2026, 10, 31)


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(
        "soccer_predictor.prediction.service.resolve_current_attack_strength", lambda *a, **k: None
    )
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _teams(session, league="EPL"):
    home = get_or_create_team(session, "Chelsea", league)
    away = get_or_create_team(session, "Manchester United", league)
    other = get_or_create_team(session, "Everton", league)
    session.commit()
    return home, away, other


def _params(home_id, away_id, league="EPL"):
    return DixonColesParams(
        league_code=league,
        attack={home_id: 1.2, away_id: 1.1},
        defense={home_id: 1.0, away_id: 1.0},
        home_advantage=1.3,
        rho=-0.1,
        xi=0.0018,
        fitted_at="2026-01-01T00:00:00",
        n_matches=100,
    )


def _play(session, league, team, opponent, results, *, start=FIXTURE_DAY - dt.timedelta(days=7)):
    """`team` plays `opponent` once a week going back from `start`, newest first,
    alternating home and away; each entry of `results` is (goals for, goals against)."""
    for i, (scored, conceded) in enumerate(results):
        day = start - dt.timedelta(days=7 * i)
        if i % 2 == 0:
            upsert_match(session, league, "2627", day, team.id, opponent.id, scored, conceded)
        else:
            upsert_match(session, league, "2627", day, opponent.id, team.id, conceded, scored)
    session.commit()


def _predict(session, home, away, league="EPL"):
    return predict_fixture(
        session, _params(home.id, away.id, league), home.id, away.id, home.canonical_name, away.canonical_name, FIXTURE_DAY
    )


def test_recent_results_are_points_newest_first_from_either_side_of_the_fixture_list(session):
    home, away, other = _teams(session)
    _play(session, "EPL", home, other, [(2, 1), (0, 0), (0, 3), (1, 1), (4, 0), (0, 1), (5, 0)])
    results = recent_results_for_team(session, home.id, FIXTURE_DAY, 6)
    assert [p for _, p in results] == [3, 1, 0, 1, 3, 0]
    assert [d for d, _ in results] == sorted((d for d, _ in results), reverse=True)


def test_a_match_on_the_fixture_day_itself_is_not_form(session):
    home, away, other = _teams(session)
    upsert_match(session, "EPL", "2627", FIXTURE_DAY, home.id, other.id, 5, 0)
    session.commit()
    assert recent_results_for_team(session, home.id, FIXTURE_DAY, 6) == []


def test_a_poor_run_lowers_attack_and_raises_goals_conceded(session):
    home, away, other = _teams(session)
    _play(session, "EPL", home, other, [(0, 1)] * 6)
    plain = _params(home.id, away.id)
    p = _predict(session, home, away)
    shift = -RESULTS_FORM_STRENGTH * AVERAGE_POINTS_PER_MATCH
    assert p.breakdown.home_results_form.points == 0 and p.breakdown.home_results_form.shift == pytest.approx(shift)
    assert p.breakdown.home_attack == pytest.approx(plain.attack[home.id] * (1 + shift))
    assert p.breakdown.home_defense == pytest.approx(plain.defense[home.id] / (1 + shift))
    assert p.breakdown.away_results_form is None  # the other side has no results at all
    assert p.breakdown.away_attack == pytest.approx(plain.attack[away.id])


def test_a_good_run_does_the_reverse_and_changes_the_forecast(session):
    home, away, other = _teams(session)
    _play(session, "EPL", away, other, [(2, 0)] * 6)
    p = _predict(session, home, away)
    assert p.breakdown.away_results_form.points == 18
    assert p.breakdown.away_attack > _params(home.id, away.id).attack[away.id]
    assert p.breakdown.away_defense < _params(home.id, away.id).defense[away.id]
    # Home is untouched, so the visitors' run is all that moved.
    no_form = predict_fixture(
        session, _params(home.id, away.id), home.id, away.id, home.canonical_name, away.canonical_name, FIXTURE_DAY - dt.timedelta(days=400)
    )
    assert p.away_win > no_form.away_win


def test_fewer_than_six_results_leaves_the_prediction_alone(session):
    home, away, other = _teams(session)
    _play(session, "EPL", home, other, [(0, 3)] * 5)
    p = _predict(session, home, away)
    assert p.breakdown.home_results_form is None
    assert p.breakdown.home_attack == pytest.approx(_params(home.id, away.id).attack[home.id])


def test_a_league_that_does_not_use_results_form_is_untouched(session):
    # The Champions League was never tested; six results there can span half a year.
    home, away, other = _teams(session, league="UCL")
    _play(session, "UCL", home, other, [(0, 3)] * 6)
    p = _predict(session, home, away, league="UCL")
    assert p.breakdown.home_results_form is None
    assert p.breakdown.home_attack == pytest.approx(_params(home.id, away.id, "UCL").attack[home.id])
