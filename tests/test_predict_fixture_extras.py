"""predict_fixture's in-memory what-if entries: they change that one prediction
exactly as saved notes would, and are never written anywhere."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.model.form_adjustment import FORM_IMPACT_CAP, FormNoteEntry
from soccer_predictor.model.injury_adjustment import ATTACK_IMPACT_FACTOR, DEFENSE_IMPACT_FACTOR, InjuryEntry
from soccer_predictor.prediction.service import predict_fixture
from soccer_predictor.storage.models import Base, Injury, PredictionRecord, TeamFormNote
from soccer_predictor.storage.repository import get_or_create_team


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(
        "soccer_predictor.prediction.service.resolve_current_attack_strength", lambda *a, **k: None
    )
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture
def teams(session):
    home = get_or_create_team(session, "Arsenal", "EPL")
    away = get_or_create_team(session, "Chelsea", "EPL")
    session.commit()
    return home, away


def _params(home_id, away_id):
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


def _predict(session, teams, **extra):
    home, away = teams
    return predict_fixture(session, _params(home.id, away.id), home.id, away.id, "Arsenal", "Chelsea", **extra)


def test_no_extras_is_exactly_the_plain_prediction(session, teams):
    plain = _predict(session, teams)
    empty = _predict(session, teams, extra_home_injuries=[], extra_away_form=())
    assert plain.home_win == empty.home_win and plain.breakdown.lambda_home == empty.breakdown.lambda_home


def test_an_extra_injury_cuts_that_sides_attack_like_a_saved_one(session, teams):
    plain = _predict(session, teams)
    hurt = _predict(session, teams, extra_home_injuries=[InjuryEntry("Saka", "attack", 0.4)])
    assert hurt.breakdown.home_attack == pytest.approx(plain.breakdown.home_attack * (1 - 0.4 * ATTACK_IMPACT_FACTOR))
    assert hurt.breakdown.away_attack == pytest.approx(plain.breakdown.away_attack)
    assert hurt.home_win < plain.home_win


def test_an_extra_defender_injury_makes_that_side_concede_more(session, teams):
    plain = _predict(session, teams)
    hurt = _predict(session, teams, extra_away_injuries=[InjuryEntry("Keeper", "defense", 0.5)])
    assert hurt.breakdown.away_defense == pytest.approx(plain.breakdown.away_defense * (1 + 0.5 * DEFENSE_IMPACT_FACTOR))
    assert hurt.breakdown.lambda_home > plain.breakdown.lambda_home


def test_extra_form_moves_a_side_within_the_form_cap(session, teams):
    plain = _predict(session, teams)
    flat = _predict(session, teams, extra_home_form=[FormNoteEntry(-5.0, "attack")])  # absurd on purpose
    assert flat.breakdown.home_attack == pytest.approx(plain.breakdown.home_attack * (1 - FORM_IMPACT_CAP))


def test_extras_are_never_stored(session, teams):
    _predict(
        session,
        teams,
        extra_home_injuries=[InjuryEntry("Saka", "attack", 0.4)],
        extra_away_form=[FormNoteEntry(-0.5, "both")],
    )
    session.commit()
    for model in (Injury, TeamFormNote, PredictionRecord):
        assert session.scalar(select(func.count()).select_from(model)) == 0
    # ...and a later plain prediction is unaffected.
    assert _predict(session, teams).breakdown.home_attack == pytest.approx(1.4)
