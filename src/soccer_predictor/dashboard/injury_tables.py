"""The Predictor's Injuries tab as sortable HTML tables (the data is dashboard/injury_table.py's).

A team's name opens its page and a player's name opens their stats on the Players page. In the team-impact table the number of players listed is a link
too: it opens who is out for that team under the table, without reloading the page
(components.inject_injury_links).
"""

from __future__ import annotations

import html

import pandas as pd

from soccer_predictor.dashboard.components import (
    INJURY_PARAM,
    TableCell,
    TableColumn,
    _signed_number,
    page_link_start,
    sortable_table_html,
    team_page_href,
)
from soccer_predictor.dashboard.players_table import player_page_href

# No return date sorts after every real one when the soonest is shown first.
_NO_DATE = 10**7

_CHANCE_HINT = "How likely he is to play the next match."
_BACK_HINT = 'Some feeds only say roughly when ("mid-November"); those show the middle of the month.'
_IMPACT_HINT = (
    "How much the model discounts the team for him (an attacker lowers its attack, a defender or keeper "
    "raises the goals it concedes). A doubtful player counts for less."
)


def _text(value: object) -> str:
    return html.escape("" if pd.isna(value) else str(value))


def _team_cell(team: str, team_ids: dict[str, int], league_code: str) -> TableCell:
    team_id = team_ids.get(team)
    if team_id is None:
        return TableCell(_text(team), team.casefold())
    href = html.escape(team_page_href(league_code, team_id))
    return TableCell(f'<a class="team" href="{href}" target="_self">{_text(team)}</a>', team.casefold())


def _player_cell(name: str, team: str, league_code: str) -> TableCell:
    """The player's name, linked to the Players page with them open. A plain link (not the in-place
    kind): it is another page, and the squad list may not spell the name the way the feed does, in
    which case that page says so."""
    href = html.escape(player_page_href(league_code, name, team))
    return TableCell(f'<a class="profile" href="{href}" target="_self">{_text(name)}</a>', name.casefold())


def _player_columns(show_team: bool) -> list[TableColumn]:
    columns = [
        TableColumn("Team", left=True, sort="text", first="asc", hint="Click to sort; click a name for the team's page"),
        TableColumn("Player", left=True, sort="text", first="asc", hint="Click to sort; click a name for their stats"),
        TableColumn("Status", left=True, sort="text", first="asc"),
        TableColumn("Problem", left=True, sort="text", first="asc"),
        TableColumn("Chance of playing", sort="num", first="desc", hint=_CHANCE_HINT),
        TableColumn("Expected back", sort="num", first="asc", hint=_BACK_HINT),
        TableColumn("Position", left=True, sort="text", first="asc"),
        TableColumn("Impact", sort="num", first="desc", hint=_IMPACT_HINT),
        TableColumn("Source", left=True, sort="text", first="asc"),
    ]
    return columns if show_team else columns[1:]


def _impact_cell(impact: float) -> TableCell:
    """A bar as wide as the impact (full width is 1.0), then the number. Its style is in the shared
    table stylesheet: a second <style> block in front of that one would end the markdown's HTML
    block early and print the rest of the CSS on the page."""
    width = min(100.0, max(0.0, impact * 100))
    return TableCell(f'<span class="impact"><i style="width:{width:.0f}%"></i></span>{impact:.2f}', impact)


def injury_players_table_html(
    frame: pd.DataFrame, team_ids: dict[str, int], league_code: str, show_team: bool = True
) -> str:
    """One row per unavailable player (injury_table.players_frame's columns). `team_ids` maps a
    team's name to its id, for the link to its page; `show_team=False` drops the Team column, for
    a list that is already one team's."""
    rows = []
    for row in frame.to_dict("records"):
        chance, back = row["Chance of playing"], row["Expected back"]
        cells = [
            _team_cell(row["Team"], team_ids, league_code),
            _player_cell(str(row["Player"]), str(row["Team"]), league_code),
            TableCell(_text(row["Status"]), str(row["Status"]).casefold()),
            TableCell(_text(row["Problem"]), str(row["Problem"]).casefold()),
            TableCell("–" if pd.isna(chance) else f"{int(chance)}%", None if pd.isna(chance) else int(chance)),
            TableCell(
                "–" if pd.isna(back) else f"{back.day} {back:%b %Y}", _NO_DATE if pd.isna(back) else back.toordinal()
            ),
            TableCell(_text(row["Position"]), str(row["Position"]).casefold()),
            _impact_cell(float(row["Impact"])),
            TableCell(_text(row["Source"]), str(row["Source"]).casefold()),
        ]
        rows.append(cells if show_team else cells[1:])
    return sortable_table_html(_player_columns(show_team), rows)


def _percent(value: float) -> str:
    return f"{_signed_number(value, 1)}%"


TEAM_IMPACT_COLUMNS = [
    TableColumn("Team", left=True, sort="text", first="asc", hint="Click to sort; click a name for the team's page"),
    TableColumn("Players listed", sort="num", hint="Click a number to see who is out"),
    TableColumn("Attack", sort="num"),
    TableColumn("Attack with injuries", sort="num"),
    TableColumn("Attack change", sort="num", first="asc", hint="How much the injuries lower the team's attack."),
    TableColumn("Defence (goals conceded)", sort="num", first="asc"),
    TableColumn("Defence with injuries", sort="num", first="asc"),
    TableColumn(
        "Defence change", sort="num", hint="How much the injuries raise the goals the team concedes: a rise is worse."
    ),
]


def team_impact_table_html(
    frame: pd.DataFrame,
    team_ids: dict[str, int],
    league_code: str,
    tz_name: str = "",
    page_params: str = "",
    selected: int | None = None,
) -> str:
    """The model's attack and defence for each team before and after its injuries
    (injury_table.team_impact_frame). A team's number of players listed reloads this page with
    `injuries=<team id>` set, which lists them under the table; `selected` is the team whose
    list is open (its row and number are marked). `tz_name` and `page_params` ("&name=value..."
    already URL-encoded) travel in that link so the page's timezone and date range survive a
    reload."""
    page_start = page_link_start(league_code, tz_name, page_params)
    rows, selected_row = [], None
    for index, row in enumerate(frame.to_dict("records")):
        team = row["Team"]
        team_id = team_ids.get(team)
        listed = int(row["Players listed"])
        if team_id is None:
            listed_cell = TableCell(str(listed), listed)
        else:
            if team_id == selected:
                selected_row = index
            picked = " on" if team_id == selected else ""
            href = html.escape(f"{page_start}&{INJURY_PARAM}={team_id}")
            listed_cell = TableCell(
                f'<a class="injured{picked}" href="{href}" target="_self" title="Who is out">{listed}</a>', listed
            )
        rows.append(
            [
                _team_cell(team, team_ids, league_code),
                listed_cell,
                TableCell(f"{row['Attack']:.2f}", row["Attack"]),
                TableCell(f"{row['Attack with injuries']:.2f}", row["Attack with injuries"]),
                TableCell(_percent(row["Attack change"]), row["Attack change"]),
                TableCell(f"{row['Defence (goals conceded)']:.2f}", row["Defence (goals conceded)"]),
                TableCell(f"{row['Defence with injuries']:.2f}", row["Defence with injuries"]),
                TableCell(_percent(row["Defence change"]), row["Defence change"]),
            ]
        )
    return sortable_table_html(TEAM_IMPACT_COLUMNS, rows, selected=selected_row)


def parse_injury_team(text: str | None, team_ids: dict[str, int], listed: set[str]) -> str | None:
    """The name of the team the `injuries` URL parameter (a team id) names, provided it is one of
    the teams with someone listed; None otherwise."""
    try:
        wanted = int(text or "")
    except ValueError:
        return None
    return next((team for team in listed if team_ids.get(team) == wanted), None)
