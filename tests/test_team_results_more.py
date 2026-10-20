"""Team Detail's recent results: 10 by default, a button reveals up to 25."""

from __future__ import annotations

import datetime as dt

from streamlit.testing.v1 import AppTest

from soccer_predictor.dashboard.views import team_detail


def _page(monkeypatch, matches):
    import pandas as pd

    frame = pd.DataFrame(
        [{"date": dt.date(2026, 10, 1) - dt.timedelta(days=i), "home_team_id": 1, "away_team_id": 2,
          "home_goals": 2, "away_goals": 1} for i in range(matches)],
        columns=["date", "home_team_id", "away_team_id", "home_goals", "away_goals"],
    )
    monkeypatch.setattr(team_detail, "matches_for_team", lambda session, team_id: frame)

    def app():
        from soccer_predictor.dashboard.views import team_detail as page

        page.render()

    at = AppTest.from_function(app, default_timeout=60)
    at.query_params["team"] = "441"
    return at


def _result_rows(at):
    tables = [d.value for d in at.dataframe if "Result" in getattr(d.value, "columns", [])]
    return len(tables[0]) if tables else 0


def _more_button(at):
    return next((b for b in at.button if "more results" in b.label or "fewer results" in b.label), None)


def test_ten_results_by_default_and_a_button_reveals_up_to_twenty_five(monkeypatch):
    at = _page(monkeypatch, 40).run()
    assert not at.exception and _result_rows(at) == 10
    assert _more_button(at).label == "Show more results (up to 25)"

    _more_button(at).click().run()
    assert _result_rows(at) == 25 and _more_button(at).label == "Show fewer results"

    _more_button(at).click().run()
    assert _result_rows(at) == 10  # and back again


def test_the_button_counts_what_there_actually_is_and_is_absent_for_ten_or_fewer(monkeypatch):
    at = _page(monkeypatch, 14).run()
    assert _more_button(at).label == "Show more results (up to 14)"
    _more_button(at).click().run()
    assert _result_rows(at) == 14

    assert _more_button(_page(monkeypatch, 10).run()) is None
    assert _more_button(_page(monkeypatch, 3).run()) is None
