"""Regression tests for prediction.service.predict_fixture attaching a
PredictionBreakdown -- the actual attack/defense/lambda numbers the
dashboard shows so a user can redo the Poisson math themselves.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.prediction.service import predict_fixture
from soccer_predictor.storage.models import Base, Injury
from soccer_predictor.storage.repository import get_or_create_team


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _params(home_id: int, away_id: int) -> DixonColesParams:
    return DixonColesParams(
        league_code="EPL",
        attack={home_id: 1.4, away_id: 1.0},
        defense={home_id: 0.9, away_id: 1.1},
        home_advantage=1.3,
        rho=-0.1,
        xi=0.0018,
        fitted_at="2026-01-01T00:00:00",
        n_matches=100,
    )


def test_breakdown_matches_manually_computed_lambdas(session):
    home = get_or_create_team(session, "Arsenal", "EPL")
    away = get_or_create_team(session, "Chelsea", "EPL")
    session.commit()
    params = _params(home.id, away.id)

    prediction = predict_fixture(session, params, home.id, away.id, "Arsenal", "Chelsea")

    b = prediction.breakdown
    assert b is not None
    # No injuries/form notes on record -- adjusted values equal the base fit.
    assert b.home_attack == pytest.approx(params.attack[home.id])
    assert b.away_defense == pytest.approx(params.defense[away.id])

    expected_lambda_home = params.attack[home.id] * params.defense[away.id] * params.home_advantage
    expected_lambda_away = params.attack[away.id] * params.defense[home.id]
    assert b.lambda_home == pytest.approx(expected_lambda_home)
    assert b.lambda_away == pytest.approx(expected_lambda_away)
    assert b.rho == params.rho
    assert b.home_advantage == params.home_advantage

    # Provenance -- proves these ratings trace back to the league-wide fit
    # (all matches, not just these two teams), not to any player-level data.
    assert b.n_matches == params.n_matches
    assert b.fitted_at == params.fitted_at
    assert b.xi == params.xi


def test_breakdown_reflects_injury_adjustment(session):
    home = get_or_create_team(session, "Arsenal", "EPL")
    away = get_or_create_team(session, "Chelsea", "EPL")
    session.commit()
    session.add(
        Injury(
            team_id=home.id,
            player_name="Star Striker",
            position="attack",
            importance_weight=1.0,
            source="api",
            note="",
            fetched_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
        )
    )
    session.commit()
    params = _params(home.id, away.id)

    prediction = predict_fixture(session, params, home.id, away.id, "Arsenal", "Chelsea")
    b = prediction.breakdown

    # base_* stays the raw fitted value; the adjusted home_attack must be
    # strictly lower once an attack-position injury with nonzero weight is
    # on record -- and lambda_home must reflect that lower value too.
    assert b.base_home_attack == pytest.approx(params.attack[home.id])
    assert b.home_attack < b.base_home_attack
    assert b.lambda_home == pytest.approx(b.home_attack * b.away_defense * b.home_advantage)
