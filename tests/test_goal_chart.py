"""The goals-over-time chart and the ESPN event parsing behind it."""

from __future__ import annotations

import pytest

from soccer_predictor.dashboard import goal_chart
from soccer_predictor.ingest import espn_client
from soccer_predictor.ingest.espn_client import MatchDetail, MatchEvent, TeamMatchInfo


def _detail(events, state="post"):
    home = TeamMatchInfo(espn_id="1", name="Palace", home_away="home", formation=None)
    away = TeamMatchInfo(espn_id="2", name="Forest", home_away="away", formation=None)
    return MatchDetail("e", state, "FT", home, away, 3, 1, None, None, None, events=events)


def _event(minute, kind, team, player="Someone"):
    return MatchEvent(minute=minute, kind=kind, text="t", team_id=team, player=player)


@pytest.mark.parametrize("label, expected", [("25'", 25), ("45'+4'", 49), ("90’+3’", 93), ("106'", 106), ("", None), ("HT", None)])
def test_event_minutes_count_stoppage_time_on_from_the_half(label, expected):
    assert goal_chart.event_minute(label) == expected


def test_goals_are_counted_per_side_in_match_order_and_substitutions_are_left_out():
    events = [
        _event("59'", "goal", "1"),
        _event("3'", "goal", "2", "Ndoye"),
        _event("41'", "goal", "1"),
        _event("45'+4'", "goal", "1"),
        _event("50'", "sub", "1"),
        _event("70'", "red", "2", "Gibbs-White"),
        _event("80'", "yellow", "1"),
        _event("", "goal", "1"),  # no readable minute: skipped
        _event("30'", "goal", "9"),  # a team that isn't in the match: skipped
    ]
    chart = goal_chart.chart_events(_detail(events))
    assert [(e.minute, e.kind, e.side, e.goals_after) for e in chart] == [
        (3, "goal", "away", 1),
        (41, "goal", "home", 1),
        (49, "goal", "home", 2),
        (59, "goal", "home", 3),
        (70, "red", "away", 1),  # a card sits on the line at that team's current goal count
        (80, "yellow", "home", 3),
    ]
    assert chart[0].label == "3' Goal: Ndoye (Forest)"
    assert chart[4].label == "70' Red card: Gibbs-White (Forest)"


def test_the_figure_has_a_blue_home_line_and_a_red_away_line_stepping_at_each_goal():
    fig = goal_chart.goal_timeline_figure(
        _detail([_event("10'", "goal", "1"), _event("40'", "goal", "1"), _event("20'", "goal", "2"), _event("70'", "red", "2")])
    )
    lines = {t.name: t for t in fig.data if t.mode == "lines"}
    home, away = lines["Palace"], lines["Forest"]
    assert home.line.color == goal_chart.HOME_COLOUR and away.line.color == goal_chart.AWAY_COLOUR
    assert home.line.shape == "hv"
    assert list(home.x) == [0, 10, 40, 91] and [round(v) for v in home.y] == [0, 1, 2, 2]
    assert list(away.x) == [0, 20, 91] and [round(v) for v in away.y] == [0, 1, 1]
    # Level scores don't hide one line behind the other.
    assert home.y[0] > 0 > away.y[0]
    # Axes: minutes along the bottom, whole goals up the side.
    assert fig.layout.xaxis.title.text == "Minute" and fig.layout.yaxis.title.text == "Goals" and fig.layout.yaxis.dtick == 1
    hover = [t for t in fig.data if t.mode == "markers"]
    assert any("Red card" in text for t in hover for text in t.text)


def test_a_match_still_in_play_ends_at_the_last_event_and_one_with_nothing_draws_no_chart():
    live = goal_chart.goal_timeline_figure(_detail([_event("23'", "goal", "1")], state="in"))
    assert max(live.data[0].x) == 24
    assert goal_chart.goal_timeline_figure(_detail([])) is None


# --- parsing --------------------------------------------------------------------------------


def test_every_kind_of_goal_is_a_goal_and_shootout_kicks_are_not():
    def ev(kind, minute, **extra):
        return {"type": {"type": kind}, "clock": {"displayValue": minute}, "team": {"id": "1"},
                "participants": [{"athlete": {"displayName": "P"}}], "text": "x", **extra}

    parsed = espn_client._parse_match_events(
        [
            ev("goal", "3'", scoringPlay=True),
            ev("goal---header", "45'+4'", scoringPlay=True),
            ev("goal---free-kick", "50'", scoringPlay=True),
            ev("penalty---scored", "60'", scoringPlay=True),
            ev("own-goal", "70'", scoringPlay=True),
            ev("something-new", "80'", scoringPlay=True),  # any scoring play counts
            ev("penalty---missed", "85'", scoringPlay=False),
            ev("penalty---scored", "120'", scoringPlay=True, shootout=True),
            ev("yellow-card", "10'"),
            ev("halftime", "45'"),
        ]
    )
    assert [(e.minute, e.kind) for e in parsed] == [
        ("3'", "goal"), ("45'+4'", "goal"), ("50'", "goal"), ("60'", "goal"), ("70'", "goal"), ("80'", "goal"), ("10'", "yellow"),
    ]
    assert parsed[0].player == "P"
