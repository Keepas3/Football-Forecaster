from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from types import SimpleNamespace

from soccer_predictor.config import League
from soccer_predictor.dashboard.components import (
    _predicted_outcome_styles,
    fixture_columns_with_kickoff_first,
    format_kickoff,
    season_already_concluded,
    season_not_yet_scheduled,
    style_fixture_predictions,
)

EASTERN = ZoneInfo("America/New_York")


def test_format_kickoff_converts_utc_to_target_timezone():
    # 2026-10-10 19:00 UTC in October is EDT (UTC-4) -> 3:00 PM
    kickoff = dt.datetime(2026, 10, 10, 19, 0)
    result = format_kickoff(kickoff, dt.date(2026, 10, 10), EASTERN)
    assert "03:00 PM" in result
    assert "EDT" in result


def test_format_kickoff_falls_back_to_date_only_when_missing():
    result = format_kickoff(None, dt.date(2026, 10, 10), EASTERN)
    assert result == "Oct 10, 2026"


def test_format_kickoff_falls_back_on_nat():
    result = format_kickoff(pd.NaT, dt.date(2026, 10, 10), EASTERN)
    assert result == "Oct 10, 2026"


def test_format_kickoff_shows_live_marker_during_match_window():
    kickoff = dt.datetime(2026, 10, 10, 19, 0)
    now = kickoff + dt.timedelta(hours=1)  # mid-match
    result = format_kickoff(kickoff, dt.date(2026, 10, 10), EASTERN, now=now)
    assert result.startswith("🔴 LIVE · ")
    assert "03:00 PM" in result


def test_format_kickoff_no_live_marker_before_kickoff():
    kickoff = dt.datetime(2026, 10, 10, 19, 0)
    now = kickoff - dt.timedelta(minutes=5)
    result = format_kickoff(kickoff, dt.date(2026, 10, 10), EASTERN, now=now)
    assert "LIVE" not in result


def test_format_kickoff_no_live_marker_after_match_window():
    kickoff = dt.datetime(2026, 10, 10, 19, 0)
    now = kickoff + dt.timedelta(hours=3)  # well past a normal match length
    result = format_kickoff(kickoff, dt.date(2026, 10, 10), EASTERN, now=now)
    assert "LIVE" not in result


def test_format_kickoff_live_marker_at_exact_kickoff():
    kickoff = dt.datetime(2026, 10, 10, 19, 0)
    result = format_kickoff(kickoff, dt.date(2026, 10, 10), EASTERN, now=kickoff)
    assert result.startswith("🔴 LIVE · ")


def _row(predicted_home_goals, predicted_away_goals, p_home=0.4, p_draw=0.3, p_away=0.3):
    # p_home/p_draw/p_away deliberately don't have to agree with the
    # predicted scoreline -- that's the exact real-world case this module
    # has to get right (see _predicted_outcome_styles' docstring/comment).
    return pd.Series(
        {
            "Home": "Team A",
            "Away": "Team B",
            "Home win %": p_home,
            "Draw %": p_draw,
            "Away win %": p_away,
            "_predicted_home_goals": predicted_home_goals,
            "_predicted_away_goals": predicted_away_goals,
        }
    )


def test_predicted_outcome_styles_home_win():
    styles = _predicted_outcome_styles(_row(2, 0))
    assert "2e7d32" in styles["Home"]  # green
    assert "c62828" in styles["Away"]  # red


def test_predicted_outcome_styles_away_win():
    styles = _predicted_outcome_styles(_row(0, 2))
    assert "c62828" in styles["Home"]
    assert "2e7d32" in styles["Away"]


def test_predicted_outcome_styles_draw():
    styles = _predicted_outcome_styles(_row(1, 1))
    assert "9e9e9e" in styles["Home"]
    assert "9e9e9e" in styles["Away"]


def test_predicted_outcome_styles_follows_scoreline_not_aggregate_probabilities():
    # The real bug this guards against: Home win % > Draw % > Away win % here
    # (a plausible Poisson output), but the single most likely exact
    # scoreline is a 1-1 draw -- coloring must follow the scoreline shown
    # in "Predicted score", not the aggregate 1X2 split, or the two columns
    # visually contradict each other.
    styles = _predicted_outcome_styles(_row(1, 1, p_home=0.414, p_draw=0.260, p_away=0.326))
    assert "9e9e9e" in styles["Home"]
    assert "9e9e9e" in styles["Away"]


def test_style_fixture_predictions_returns_styler_with_correct_row_count():
    rows = [
        {
            "Home": "A", "Away": "B", "Home win %": 0.6, "Draw %": 0.2, "Away win %": 0.2,
            "_predicted_home_goals": 2, "_predicted_away_goals": 0,
        },
        {
            "Home": "C", "Away": "D", "Home win %": 0.2, "Draw %": 0.2, "Away win %": 0.6,
            "_predicted_home_goals": 0, "_predicted_away_goals": 2,
        },
    ]
    styler = style_fixture_predictions(rows)
    assert len(styler.data) == 2


def test_style_fixture_predictions_hides_internal_columns():
    rows = [
        {
            "Home": "A", "Away": "B", "Home win %": 0.6, "Draw %": 0.2, "Away win %": 0.2,
            "_predicted_home_goals": 2, "_predicted_away_goals": 0,
        },
    ]
    styler = style_fixture_predictions(rows)
    rendered_columns = styler.columns.tolist()
    hidden = getattr(styler, "hidden_columns", None)
    # Whichever internal attribute this pandas version exposes, the hidden
    # column names must show up as excluded from what actually renders.
    assert "_predicted_home_goals" not in (rendered_columns if hidden is None else [])
    html = styler.to_html()
    assert "_predicted_home_goals" not in html
    assert "_predicted_away_goals" not in html


def test_fixture_columns_with_kickoff_first_excludes_hidden_keys():
    row = {
        "Home": "A", "Away": "B", "Kickoff": "Oct 10", "Home win %": 0.5,
        "_predicted_home_goals": 1, "_predicted_away_goals": 0,
    }
    columns = fixture_columns_with_kickoff_first(row)
    assert columns[0] == "Kickoff"
    assert "_predicted_home_goals" not in columns
    assert "_predicted_away_goals" not in columns
    assert set(columns) == {"Kickoff", "Home", "Away", "Home win %"}


_EURO = League(
    code="EURO",
    name="European Championship",
    api_competition_id=2018,
    seasons=["2024", "2028"],
    season_display="single_year",
)
_EPL = League(code="EPL", name="English Premier League", api_competition_id=2021, seasons=["2425"], csv_code="E0")


def test_season_not_yet_scheduled_true_for_future_tournament_year():
    assert season_not_yet_scheduled(_EURO, "2028", today=dt.date(2026, 9, 27)) is True


def test_season_not_yet_scheduled_false_for_past_or_current_tournament_year():
    assert season_not_yet_scheduled(_EURO, "2024", today=dt.date(2026, 9, 27)) is False
    assert season_not_yet_scheduled(_EURO, "2028", today=dt.date(2028, 6, 1)) is False


def test_season_not_yet_scheduled_false_for_range_display_leagues():
    # A domestic league always has a continuously-running current season --
    # this must never apply to season_display="range" leagues, even for a
    # nominally "future" season code like next year's.
    assert season_not_yet_scheduled(_EPL, "2627", today=dt.date(2026, 9, 27)) is False


def test_season_already_concluded_true_for_past_tournament_year():
    assert season_already_concluded(_EURO, "1996", today=dt.date(2026, 9, 27)) is True


def test_season_already_concluded_false_for_current_or_future_tournament_year():
    # The in-progress tournament year isn't "concluded" even with nothing
    # upcoming in the next 14 days -- more fixtures could still be added.
    assert season_already_concluded(_EURO, "2024", today=dt.date(2024, 1, 1)) is False
    assert season_already_concluded(_EURO, "2028", today=dt.date(2026, 9, 27)) is False


def test_season_already_concluded_false_for_range_display_leagues():
    assert season_already_concluded(_EPL, "1516", today=dt.date(2026, 9, 27)) is False


def test_fixture_prediction_row_uses_plain_english_percent_columns():
    from soccer_predictor.dashboard.components import fixture_prediction_row
    from soccer_predictor.model.markets import MatchPrediction

    prediction = MatchPrediction(
        home_win=0.4523, draw=0.2611, away_win=0.2866, over_2_5=0.5712, under_2_5=0.4288,
        both_teams_to_score=0.5049, top_scorelines=[(1, 0, 0.12)],
    )
    row = fixture_prediction_row("Arsenal", "Chelsea", prediction)

    assert row["Home win %"] == 45.23
    assert row["Draw %"] == 26.11
    assert row["Away win %"] == 28.66
    assert row["Over 2.5 goals %"] == 57.12
    assert row["Both teams score %"] == 50.49
    assert not any(key.startswith("P(") for key in row)


def test_sort_fixtures_by_kickoff_orders_within_a_day_and_puts_unknown_times_last():
    import datetime as dt

    import pandas as pd

    from soccer_predictor.dashboard.components import sort_fixtures_by_kickoff

    day = dt.date(2026, 10, 4)
    df = pd.DataFrame(
        {
            "date": [day, day, day, dt.date(2026, 10, 5), day],
            "home_team_id": [1, 2, 3, 4, 5],
            "kickoff_utc": [
                dt.datetime(2026, 10, 4, 18, 45),
                dt.datetime(2026, 10, 4, 16, 0),
                None,
                dt.datetime(2026, 10, 5, 13, 0),
                dt.datetime(2026, 10, 4, 18, 45),
            ],
        }
    )
    ordered = sort_fixtures_by_kickoff(df)["home_team_id"].tolist()
    # 16:00, then the two 18:45s in their original order, then no-time, then next day.
    assert ordered == [2, 1, 5, 3, 4]


def test_percentage_columns_show_two_decimals_everywhere():
    from soccer_predictor.dashboard import components

    prediction = SimpleNamespace(
        home_win=0.6412345, draw=0.2, away_win=0.1587655, over_2_5=0.5, both_teams_to_score=0.123456,
        top_scorelines=[(2, 1, 0.1)],
    )
    row = components.fixture_prediction_row("A", "B", prediction)
    assert row[components.HOME_WIN_COLUMN] == 64.12 and row[components.BTTS_COLUMN] == 12.35

    html = components.style_fixture_predictions([row]).to_html()
    assert ">64.12<" in html and ">20.00<" in html and ">12.35<" in html  # a whole number keeps its zeros
    assert set(components.fixture_percent_column_config()) == set(components.PERCENT_COLUMNS)


def test_a_locked_in_match_shows_the_locked_forecast_and_an_early_one_the_current_model():
    from soccer_predictor.dashboard import components
    from soccer_predictor.storage.repository import LockedPrediction

    live = SimpleNamespace(
        home_win=0.50, draw=0.25, away_win=0.25, over_2_5=0.55, both_teams_to_score=0.5, top_scorelines=[(1, 0, 0.12)],
    )
    early = components.fixture_prediction_row("A", "B", live)
    assert early[components.STATUS_COLUMN] == components.EARLY_LABEL
    assert early["Predicted score"] == "1-0" and early[components.HOME_WIN_COLUMN] == 50.0

    locked = components.fixture_prediction_row("A", "B", live, LockedPrediction(2, 2, 0.30, 0.40, 0.30))
    assert locked[components.STATUS_COLUMN] == components.LOCKED_LABEL
    assert locked["Predicted score"] == "2-2" and locked[components.DRAW_COLUMN] == 40.0
    assert locked["_predicted_home_goals"] == 2  # the row colouring follows the locked score too
    # An old record with no stored probabilities keeps the locked score but the live percentages.
    old = components.fixture_prediction_row("A", "B", live, LockedPrediction(0, 1, None, None, None))
    assert old["Predicted score"] == "0-1" and old[components.HOME_WIN_COLUMN] == 50.0
    assert components.STATUS_COLUMN in components.fixture_columns_with_kickoff_first({**locked, "Kickoff": "x"})
