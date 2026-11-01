"""The leaderboard lists the clubs in the league's current season, not every club the model has
ever rated: storage/repository.py::team_ids_in_season, leaderboard_dataframe's `team_ids` and
components.current_season_leaderboard."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.dashboard import components
from soccer_predictor.dashboard import squad_strength as ss
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, team_ids_in_season, upsert_fixture, upsert_match

TODAY = dt.date(2026, 10, 30)
EPL = League(code="EPL", name="Premier League", seasons=["2526", "2627"])
WORLD_CUP = League(code="WC", name="World Cup", seasons=["2022", "2030"], season_display="single_year")


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _teams(session, code, *names):
    teams = {name: get_or_create_team(session, name, code).id for name in names}
    session.commit()
    return teams


def _fixture(session, code, day, home, away):
    upsert_fixture(session, code, day, home, away, "SCHEDULED", None)
    session.commit()


def _played(session, code, season, day, home, away):
    upsert_match(session, code, season, day, home, away, 1, 0)
    session.commit()


# --- which clubs are in the season ----------------------------------------------------------------


def test_the_clubs_in_a_season_are_those_with_a_result_in_it_or_a_fixture_still_to_play(session):
    t = _teams(session, "EPL", "Arsenal", "Chelsea", "Promoted", "Watford", "Leicester")
    _played(session, "EPL", "2627", dt.date(2026, 9, 1), t["Arsenal"], t["Chelsea"])
    _fixture(session, "EPL", dt.date(2026, 11, 7), t["Promoted"], t["Arsenal"])  # not played a game yet
    _played(session, "EPL", "2526", dt.date(2026, 4, 1), t["Watford"], t["Leicester"])  # last season: gone
    assert team_ids_in_season(session, "EPL", "2627", TODAY) == {t["Arsenal"], t["Chelsea"], t["Promoted"]}


def test_a_fixture_already_in_the_past_does_not_keep_a_club_in_the_season(session):
    t = _teams(session, "EPL", "Arsenal", "Chelsea", "Stale", "Other")
    _fixture(session, "EPL", dt.date(2026, 8, 1), t["Stale"], t["Other"])
    _fixture(session, "EPL", TODAY, t["Arsenal"], t["Chelsea"])  # today still counts
    assert team_ids_in_season(session, "EPL", "2627", TODAY) == {t["Arsenal"], t["Chelsea"]}


def test_another_leagues_matches_and_fixtures_are_not_counted(session):
    _teams(session, "EPL", "Arsenal", "Chelsea")
    u = _teams(session, "LALIGA", "Real Madrid", "Barcelona")
    _played(session, "LALIGA", "2627", dt.date(2026, 9, 1), u["Real Madrid"], u["Barcelona"])
    _fixture(session, "LALIGA", dt.date(2026, 11, 7), u["Barcelona"], u["Real Madrid"])
    assert team_ids_in_season(session, "EPL", "2627", TODAY) == set()
    assert team_ids_in_season(session, "LALIGA", "2627", TODAY) == set(u.values())


# --- the leaderboard ------------------------------------------------------------------------------


def _params(ids):
    return DixonColesParams(
        league_code="EPL",
        attack={team_id: 1.0 + team_id / 10 for team_id in ids},
        defense={team_id: 1.0 for team_id in ids},
        home_advantage=1.3,
        rho=-0.1,
        xi=0.0018,
        fitted_at="2026-01-01T00:00:00",
        n_matches=100,
    )


@pytest.fixture(autouse=True)
def _no_current_form(monkeypatch):
    monkeypatch.setattr(components, "resolve_current_attack_strength", lambda *a, **k: None)


NAMES = {1: "Arsenal", 2: "Chelsea", 3: "Watford", 4: "Leicester"}


def test_only_the_named_teams_are_listed_and_ranked():
    board = components.leaderboard_dataframe(_params(NAMES), NAMES, {1, 2})
    assert list(board["Team"]) == ["Chelsea", "Arsenal"]  # strongest first


def test_no_filter_lists_every_rated_team():
    assert len(components.leaderboard_dataframe(_params(NAMES), NAMES)) == 4


@pytest.mark.parametrize("ids", [set(), {1}, {99, 98}])
def test_a_filter_that_leaves_fewer_than_two_teams_is_ignored(ids):
    assert len(components.leaderboard_dataframe(_params(NAMES), NAMES, ids)) == 4


def test_the_current_season_leaderboard_drops_clubs_that_have_left_the_league(session):
    t = _teams(session, "EPL", "Arsenal", "Chelsea", "Watford")
    _played(session, "EPL", "2627", dt.date(2026, 9, 1), t["Arsenal"], t["Chelsea"])
    _played(session, "EPL", "2526", dt.date(2026, 4, 1), t["Watford"], t["Chelsea"])
    names = {v: k for k, v in t.items()}
    board = components.current_season_leaderboard(session, EPL, _params(names), names)
    assert set(board["Team"]) == {"Arsenal", "Chelsea"}


def test_an_edition_that_has_not_started_lists_every_rated_team(session):
    t = _teams(session, "WC", "Brazil", "France", "Chile")
    _played(session, "WC", "2022", dt.date(2022, 11, 25), t["Brazil"], t["France"])  # the 2030 edition has nothing yet
    names = {v: k for k, v in t.items()}
    board = components.current_season_leaderboard(session, WORLD_CUP, _params(names), names)
    assert set(board["Team"]) == {"Brazil", "France", "Chile"}


def test_a_page_with_no_league_config_lists_every_rated_team(session):
    t = _teams(session, "EPL", "Arsenal", "Chelsea")
    names = {v: k for k, v in t.items()}
    assert len(components.current_season_leaderboard(session, None, _params(names), names)) == 2


# --- the team page's own line ---------------------------------------------------------------------


def test_a_club_that_is_not_on_the_leaderboard_still_gets_its_net_strength():
    strength = ss.TeamStrength(attack=1.0, attack_results_only=1.0, defense=1.2, net=-0.2, rank=None)
    line = ss.strength_line(strength)
    assert line.startswith("**Net strength −0.20** (attack minus defense) · not on the Predictor's Team Strength Leaderboard")
    assert "this season's clubs" in line
