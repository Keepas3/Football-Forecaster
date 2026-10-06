"""Per-league rating settings (memory length, smoothing, Nations League
division prior) flow from config/leagues.yaml into the fit."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League, load_leagues
from soccer_predictor.model.time_weighting import DEFAULT_XI
from soccer_predictor.prediction import training
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, upsert_fixture, upsert_match


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Group A1", 0), ("Group B4", 1), ("Group C2", 2), ("Group D1", 3),
        ("League A - Group 3", 0), ("League D - Group 1", 3),
        ("Group E", None), ("Second Group Stage - Group A", None), ("", None), (None, None),
        (float("nan"), None),  # an untagged knockout match read through pandas
    ],
)
def test_division_index_reads_every_label_style(label, expected):
    assert training.division_index(label) == expected


def test_the_real_league_configs():
    leagues = load_leagues()
    nl, wc, euro, epl = leagues["NL"], leagues["WC"], leagues["EURO"], leagues["EPL"]
    # National teams: long memory (default half-life is ~1 year; these are far longer)...
    assert nl.rating_decay_xi < DEFAULT_XI and wc.rating_decay_xi < DEFAULT_XI and euro.rating_decay_xi < DEFAULT_XI
    # ...and the Nations League also starts each team from its division's level.
    assert nl.division_prior and nl.rating_ridge > 0
    assert not wc.division_prior and not euro.division_prior
    # Domestic leagues keep the standard fit.
    assert (epl.rating_decay_xi, epl.rating_ridge, epl.division_prior) == (None, 0.0, False)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _nl_league(**overrides):
    return League(code="NL", name="UEFA Nations League", seasons=["2425", "2627"], has_groups=True, **overrides)


def test_division_groups_read_the_latest_editions_labels(session):
    spain = get_or_create_team(session, "Spain", "NL").id
    malta = get_or_create_team(session, "Malta", "NL").id
    andorra = get_or_create_team(session, "Andorra", "NL").id
    france = get_or_create_team(session, "France", "NL").id
    day = dt.date(2026, 9, 5)
    upsert_match(session, "NL", "2627", day, spain, france, 1, 0, group_name="Group A1")
    upsert_match(session, "NL", "2425", day, malta, andorra, 1, 0, group_name="Group B1")  # an older edition
    upsert_fixture(session, "NL", dt.date(2026, 10, 9), malta, andorra, "SCHEDULED", group_name="Group D2")
    session.commit()

    groups = training.division_groups(session, _nl_league())

    # Spain/France: League A from the latest edition's match; Malta/Andorra: League D from its
    # fixture -- the old edition's "Group B1" is ignored.
    assert groups == {spain: 0, france: 0, malta: 3, andorra: 3}


def _capture_fit(monkeypatch):
    calls = {}

    def fake_fit(matches, league_code, **kwargs):
        calls.update(kwargs)
        calls["league_code"] = league_code

        class Params:
            n_matches = len(matches)

            def to_json(self):
                return "{}"

        return Params()

    monkeypatch.setattr(training, "fit_league", fake_fit)
    return calls


def _seed_matches(session, league_code="NL"):
    a = get_or_create_team(session, "A", league_code).id
    b = get_or_create_team(session, "B", league_code).id
    upsert_match(session, league_code, "2627", dt.date(2026, 9, 5), a, b, 1, 0, group_name="Group C1")
    session.commit()
    return a, b


def test_train_uses_the_leagues_own_settings(monkeypatch, session):
    league = _nl_league(rating_decay_xi=0.0005, rating_ridge=0.15, division_prior=True)
    monkeypatch.setattr(training, "load_leagues", lambda: {"NL": league})
    calls = _capture_fit(monkeypatch)
    a, b = _seed_matches(session)

    training.train_league(session, "NL")

    assert calls["xi"] == 0.0005 and calls["ridge"] == 0.15
    assert calls["team_groups"] == {a: 2, b: 2}  # League C
    assert calls["default_group"] == training.DEFAULT_DIVISION


def test_a_league_without_settings_gets_the_standard_fit(monkeypatch, session):
    league = League(code="EPL", name="English Premier League", seasons=["2627"], csv_code="E0")
    monkeypatch.setattr(training, "load_leagues", lambda: {"EPL": league})
    calls = _capture_fit(monkeypatch)
    _seed_matches(session, "EPL")

    training.train_league(session, "EPL")

    assert calls["xi"] == DEFAULT_XI and calls["ridge"] == 0.0 and calls["team_groups"] is None


def test_an_explicit_xi_overrides_the_leagues_setting(monkeypatch, session):
    league = _nl_league(rating_decay_xi=0.0005)
    monkeypatch.setattr(training, "load_leagues", lambda: {"NL": league})
    calls = _capture_fit(monkeypatch)
    _seed_matches(session)

    training.train_league(session, "NL", xi=0.01)

    assert calls["xi"] == 0.01
