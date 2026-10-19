"""The match page's goals-over-time chart: minutes played along the bottom,
goals scored up the side, a blue line for the home team and a red one for the
away team. Hover a marker for what happened (who scored, who was carded).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import plotly.graph_objects as go

from soccer_predictor.ingest.espn_client import MatchDetail, MatchEvent

HOME_COLOUR = "#2f6bff"
AWAY_COLOUR = "#e5342f"
YELLOW = "#f2c200"
RED_CARD = "#b71c1c"
# Both teams are level at 0-0 and often afterwards, so one line would hide the
# other; each is nudged this far (in goals) off its true value, up for home and
# down for away. Hover text always shows the real numbers.
LINE_OFFSET = 0.05
REGULATION_MINUTES = 90

_MINUTE = re.compile(r"(\d+)(?:'|’)?(?:\s*\+\s*(\d+))?")


def event_minute(label: str) -> float | None:
    """"25'" -> 25, "45'+2'" -> 47 (stoppage time counted on from the half)."""
    match = _MINUTE.match((label or "").strip())
    if not match:
        return None
    return float(int(match.group(1)) + int(match.group(2) or 0))


@dataclass
class ChartEvent:
    minute: float
    kind: str  # "goal" | "yellow" | "red"
    side: str  # "home" | "away"
    goals_after: int  # that side's goals once this event has happened
    label: str  # the hover text


def _describe(event: MatchEvent, kind: str, team_name: str) -> str:
    who = event.player or "Unknown player"
    action = {"goal": "Goal", "yellow": "Yellow card", "red": "Red card"}[kind]
    return f"{event.minute} {action}: {who} ({team_name})"


def chart_events(detail: MatchDetail) -> list[ChartEvent]:
    """Goals and cards in match order, each tagged with the side it belongs to
    and that side's goal count at the time. Substitutions are left out (they
    would bury the goals). An event with no readable minute or team is skipped."""
    sides = {detail.home.espn_id: ("home", detail.home.name), detail.away.espn_id: ("away", detail.away.name)}
    timed = []
    for index, event in enumerate(detail.events):
        minute = event_minute(event.minute)
        if event.kind not in ("goal", "yellow", "red") or minute is None or event.team_id not in sides:
            continue
        timed.append((minute, index, event))
    timed.sort(key=lambda item: (item[0], item[1]))

    goals = {"home": 0, "away": 0}
    events = []
    for minute, _, event in timed:
        side, team_name = sides[event.team_id]
        if event.kind == "goal":
            goals[side] += 1
        events.append(
            ChartEvent(minute, event.kind, side, goals[side], _describe(event, event.kind, team_name))
        )
    return events


def goal_timeline_figure(detail: MatchDetail) -> go.Figure | None:
    """None when there is nothing to draw (no goals or cards, e.g. before kickoff)."""
    events = chart_events(detail)
    if not events:
        return None
    end = max(REGULATION_MINUTES if detail.state == "post" else 0, max(e.minute for e in events)) + 1

    fig = go.Figure()
    for side, name, colour, offset in (
        ("home", detail.home.name, HOME_COLOUR, LINE_OFFSET),
        ("away", detail.away.name, AWAY_COLOUR, -LINE_OFFSET),
    ):
        goal_events = [e for e in events if e.side == side and e.kind == "goal"]
        xs = [0.0, *[e.minute for e in goal_events], end]
        ys = [0, *[e.goals_after for e in goal_events], goal_events[-1].goals_after if goal_events else 0]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[y + offset for y in ys],
                mode="lines",
                line=dict(color=colour, width=3, shape="hv"),  # a step: the count jumps at the goal
                name=name,
                hoverinfo="skip",
            )
        )
        for kind, symbol, marker_colour, size, label in (
            ("goal", "circle", colour, 13, "Goal"),
            ("yellow", "square", YELLOW, 10, "Yellow card"),
            ("red", "square", RED_CARD, 11, "Red card"),
        ):
            chosen = [e for e in events if e.side == side and e.kind == kind]
            if not chosen:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[e.minute for e in chosen],
                    y=[e.goals_after + offset for e in chosen],
                    mode="markers",
                    marker=dict(symbol=symbol, size=size, color=marker_colour, line=dict(color=colour, width=2)),
                    name=f"{name}: {label.lower()}s",
                    showlegend=False,
                    text=[e.label for e in chosen],
                    hovertemplate="%{text}<br>%{customdata} goal(s) so far<extra></extra>",
                    customdata=[e.goals_after for e in chosen],
                )
            )

    top = max(e.goals_after for e in events)
    fig.add_vline(x=45, line_dash="dot", line_color="gray", opacity=0.6, annotation_text="Half-time", annotation_position="top")
    fig.update_layout(
        height=340,
        margin=dict(t=30, b=40, l=40, r=20),
        xaxis=dict(title="Minute", range=[0, end], dtick=15, zeroline=False),
        yaxis=dict(title="Goals", range=[-0.4, top + 0.6], dtick=1, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="closest",
    )
    return fig
