"""The Predictor page's Upcoming Fixtures tab: the same HTML table as the Leagues page (team,
kickoff and predicted-score links), the workings of the match picked by its link, and the
sidebar date range that has to survive the page reload those links cause."""

from __future__ import annotations

import datetime as dt
import re
from contextlib import contextmanager

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from streamlit.testing.v1 import AppTest

from soccer_predictor.dashboard import fixture_workings
from soccer_predictor.dashboard.views import predictor
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team

TODAY = dt.date.today()
MATCH_DAY = TODAY + dt.timedelta(days=3)


@pytest.fixture
def page(monkeypatch):
    """The Predictor page over an in-memory database with two teams and one fixture;
    returns a function that runs it and the (start, end) ranges the page asked fixtures for."""
    engine = create_engine("sqlite://", future=True, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = Session(engine)
    home = get_or_create_team(session, "Arsenal", "EPL")
    away = get_or_create_team(session, "Chelsea", "EPL")
    session.commit()

    @contextmanager
    def scope():
        yield session
        session.flush()

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
    asked: list[tuple[dt.date, dt.date]] = []

    def fixtures(_session, _league_code, start, end):
        asked.append((start, end))
        kickoff = dt.datetime.combine(MATCH_DAY, dt.time(15, 0))
        return pd.DataFrame(
            [(MATCH_DAY, home.id, away.id, "scheduled", kickoff)],
            columns=["date", "home_team_id", "away_team_id", "status", "kickoff_utc"],
        )

    monkeypatch.setattr(predictor, "session_scope", scope)
    monkeypatch.setattr(fixture_workings, "session_scope", scope)
    monkeypatch.setattr(predictor, "load_latest_params", lambda *a, **k: params)
    monkeypatch.setattr(predictor, "fixtures_for_league", fixtures)
    monkeypatch.setattr("soccer_predictor.prediction.service.resolve_current_attack_strength", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def run(**query):
        def app():
            from soccer_predictor.dashboard.views import predictor as view

            view.render()

        at = AppTest.from_function(app, default_timeout=60)
        at.query_params["league"] = "EPL"
        for name, value in query.items():
            at.query_params[name] = value
        return at.run()

    yield run, asked, (home.id, away.id)
    session.close()


def _table(at) -> str | None:
    return next((m.value for m in at.markdown if 'class="fb-fixtures"' in m.value), None)


def _hrefs(markup: str) -> list[str]:
    return re.findall(r'href="([^"]*)"', markup)


def _close_buttons(at) -> list[str]:
    return [b.key for b in at.button if b.key and b.key.startswith("close_fixture_breakdown")]


def test_upcoming_fixtures_are_a_linked_html_table_not_a_dataframe(page):
    run, _, (home_id, away_id) = page
    at = run()
    assert not at.exception
    markup = _table(at)
    assert markup is not None and "Arsenal" in markup and "Chelsea" in markup
    assert not any("Home win %" in d.value.columns for d in at.dataframe)  # no canvas grid of fixtures any more
    hrefs = _hrefs(markup)
    assert f"/team_detail?league=EPL&amp;team={home_id}" in " ".join(hrefs)
    assert f"/team_detail?league=EPL&amp;team={away_id}" in " ".join(hrefs)
    score_link = next(h for h in hrefs if h.startswith("?"))
    assert f"fixture={home_id}-{away_id}-{MATCH_DAY.isoformat()}" in score_link
    assert any("predicted score to see how it was calculated" in c.value for c in at.caption)
    assert 'class="selected"' not in markup and not _close_buttons(at)  # nothing open until a score is clicked


def test_the_predicted_score_link_carries_the_date_range(page):
    run, asked, _ = page
    at = run(**{"from": "2026-12-01", "to": "2026-12-10"})
    assert not at.exception
    assert asked == [(dt.date(2026, 12, 1), dt.date(2026, 12, 10))]
    assert at.sidebar.date_input[0].value == (dt.date(2026, 12, 1), dt.date(2026, 12, 10))
    score_link = next(h for h in _hrefs(_table(at)) if h.startswith("?"))
    assert "&amp;from=2026-12-01&amp;to=2026-12-10&amp;fixture=" in score_link


def test_without_a_range_in_the_link_the_next_two_weeks_are_shown(page):
    run, asked, _ = page
    at = run()
    assert asked == [(TODAY, TODAY + dt.timedelta(days=14))]
    score_link = next(h for h in _hrefs(_table(at)) if h.startswith("?"))
    assert f"from={TODAY.isoformat()}&amp;to={(TODAY + dt.timedelta(days=14)).isoformat()}" in score_link


def test_the_fixture_in_the_link_opens_its_workings_under_the_table(page):
    run, _, (home_id, away_id) = page
    at = run(fixture=f"{home_id}-{away_id}-{MATCH_DAY.isoformat()}")
    assert not at.exception
    assert 'class="selected"' in _table(at)
    assert any("How Arsenal v Chelsea was predicted" in m.value for m in at.markdown)
    assert _close_buttons(at) == ["close_fixture_breakdown_EPL"]


def test_a_fixture_that_is_not_in_the_table_is_dropped_from_the_link(page):
    run, _, _ = page
    at = run(fixture="9-9-2026-01-01")
    assert not at.exception
    assert 'class="selected"' not in _table(at)
    assert "fixture" not in at.query_params
    assert not any(m.value.startswith("#### How ") for m in at.markdown) and not _close_buttons(at)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ({"from": "2026-11-01", "to": "2026-11-20"}, (dt.date(2026, 11, 1), dt.date(2026, 11, 20))),
        ({"from": "2026-11-05", "to": "2026-11-05"}, (dt.date(2026, 11, 5), dt.date(2026, 11, 5))),
        ({"from": "2026-11-20", "to": "2026-11-01"}, None),  # backwards
        ({"from": "not-a-date", "to": "2026-11-01"}, None),
        ({"from": "2026-11-01"}, None),
        ({}, None),
    ],
)
def test_the_date_range_in_a_link_is_used_only_when_it_is_a_real_range(query, expected):
    default = (dt.date(2026, 10, 29), dt.date(2026, 11, 12))
    assert predictor.date_range_from_url(query, default) == (expected or default)
