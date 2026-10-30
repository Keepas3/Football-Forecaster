"""predict_fixture with a named starting XI: missing regulars replace that side's
injuries; with no XI (or no stats to rate it) nothing changes."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.model.injury_adjustment import ATTACK_IMPACT_FACTOR
from soccer_predictor.model.lineup_adjustment import LINEUP_ATTACK_FACTOR, LINEUP_DEFENSE_FACTOR, ExpectedPlayer
from soccer_predictor.prediction import service
from soccer_predictor.prediction.service import predict_fixture
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, replace_injuries

SQUAD = [ExpectedPlayer("Keeper", "defense", 0.9, 1000, True)] + [
    ExpectedPlayer(f"Player {chr(65 + i)}", "attack", 0.2, 900 - i * 10) for i in range(10)
]
SAKA = ExpectedPlayer("Bukayo Saka", "attack", 0.5, 2000)  # the busiest outfielder


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
    home = get_or_create_team(session, "Arsenal", "EPL")
    away = get_or_create_team(session, "Chelsea", "EPL")
    session.commit()
    return home, away


def _predict(session, teams, **kwargs):
    home, away = teams
    params = DixonColesParams(
        league_code="EPL",
        attack={home.id: 1.4, away.id: 1.0},
        defense={home.id: 0.9, away.id: 1.1},
        home_advantage=1.3,
        rho=-0.1,
        xi=0.0018,
        fitted_at="2026-01-01T00:00:00",
        n_matches=100,
    )
    return predict_fixture(session, params, home.id, away.id, "Arsenal", "Chelsea", **kwargs)


def test_no_lineup_is_exactly_the_plain_prediction(session, teams):
    plain = _predict(session, teams)
    assert plain.breakdown.home_lineup_absences is None and plain.breakdown.away_lineup_absences is None
    assert _predict(session, teams, home_lineup=None, away_lineup=[]).home_win == plain.home_win


def test_the_usual_xi_playing_changes_nothing(session, teams):
    plain = _predict(session, teams)
    full = _predict(session, teams, home_lineup=[p.name for p in [SAKA, *SQUAD][:11]])
    assert full.breakdown.home_lineup_absences == []
    assert full.breakdown.home_attack == pytest.approx(plain.breakdown.home_attack)
    assert full.home_win == pytest.approx(plain.home_win)


def test_a_missing_regular_cuts_that_sides_attack_by_his_weight(session, teams):
    plain = _predict(session, teams)
    without_saka = _predict(session, teams, home_lineup=[p.name for p in SQUAD] + ["Rotation Player"])
    absences = without_saka.breakdown.home_lineup_absences
    assert [a.name for a in absences] == ["Bukayo Saka"]
    expected = plain.breakdown.home_attack * (1 - 0.5 * LINEUP_ATTACK_FACTOR)
    assert without_saka.breakdown.home_attack == pytest.approx(expected)
    assert without_saka.home_win < plain.home_win


def test_a_missing_defender_or_keeper_raises_goals_conceded(session, teams):
    plain = _predict(session, teams)
    no_keeper = _predict(session, teams, home_lineup=["Bukayo Saka"] + [p.name for p in SQUAD[1:]] + ["Aaron Ramsdale"])
    assert [a.name for a in no_keeper.breakdown.home_lineup_absences] == ["Keeper"]
    assert no_keeper.breakdown.home_defense == pytest.approx(plain.breakdown.home_defense * (1 + 0.9 * LINEUP_DEFENSE_FACTOR))


def test_the_named_xi_replaces_that_teams_injuries(session, teams):
    home, _ = teams
    replace_injuries(
        session, home.id, source="api",
        injuries=[{"player_name": "Bukayo Saka", "position": "attack", "importance_weight": 0.5, "note": "Knock"}],
    )
    session.commit()
    injured = _predict(session, teams)
    assert injured.breakdown.home_attack < injured.breakdown.base_home_attack  # the injury counts without a lineup
    starts_anyway = _predict(session, teams, home_lineup=[p.name for p in [SAKA, *SQUAD][:11]])
    assert starts_anyway.breakdown.home_attack == pytest.approx(starts_anyway.breakdown.base_home_attack)  # he is named: no penalty


def test_a_side_without_usable_stats_keeps_its_injuries(session, teams):
    _, away = teams
    replace_injuries(
        session, away.id, source="api",
        injuries=[{"player_name": "Reece James", "position": "defense", "importance_weight": 0.6, "note": "Knock"}],
    )
    session.commit()
    plain = _predict(session, teams)
    rated = _predict(session, teams, away_lineup=[f"Chelsea Player {i}" for i in range(11)])  # no stats for Chelsea
    assert rated.breakdown.away_lineup_absences is None
    assert rated.breakdown.away_defense == pytest.approx(plain.breakdown.away_defense)
    assert rated.breakdown.away_defense > rated.breakdown.base_away_defense  # the injury still applies


def test_what_if_extras_still_apply_on_top_of_a_lineup(session, teams):
    from soccer_predictor.model.injury_adjustment import InjuryEntry

    xi = [p.name for p in [SAKA, *SQUAD][:11]]
    plain = _predict(session, teams, home_lineup=xi)
    extra = _predict(session, teams, home_lineup=xi, extra_home_injuries=[InjuryEntry("Someone", "attack", 0.4)])
    assert extra.breakdown.home_attack == pytest.approx(plain.breakdown.home_attack * (1 - 0.4 * ATTACK_IMPACT_FACTOR))
