"""The Players page's two lists (a league's top players, and the results of a name search) as
sortable HTML tables. A player's name opens their stats under the table without reloading the
page (components.inject_player_links); a club's name opens that team's page.

The top list comes from a stats source (Understat, American Soccer Analysis, ...) that spells
clubs its own way, so `club_team_ids` works out which of our teams each one is; a club it cannot
place is shown as plain text rather than linked to the wrong team.
"""

from __future__ import annotations

import html
import urllib.parse
from collections.abc import Iterable

from rapidfuzz import fuzz, process

from soccer_predictor.dashboard.components import (
    PLAYER_PARAM,
    TableCell,
    TableColumn,
    sortable_table_html,
    team_page_href,
)
from soccer_predictor.ingest.asa_client import TEAM_NAME_OVERRIDES
from soccer_predictor.ingest.squad import PlayerSearchResult
from soccer_predictor.ingest.top_players import normalize_name
from soccer_predictor.ingest.understat_client import TEAM_NAME_ALIASES

# Stricter than the 75 the stats clients use to pair names: those pick the best of a few dozen
# clubs for a team that must exist, whereas an unplaced club here just goes without a link.
CLUB_MATCH_THRESHOLD = 85


def player_key(name: str, club: str) -> str:
    """The `player` URL parameter that selects a player: "<name>|<club>"."""
    return f"{name}|{club}"


def parse_player_key(text: str | None) -> tuple[str, str] | None:
    """(name, club) from a `player` URL parameter, None if it names nobody. The club may be empty."""
    name, _, club = (text or "").partition("|")
    return (name.strip(), club.strip()) if name.strip() else None


def club_team_ids(
    clubs: Iterable[str], teams: dict[int, str], aliases: Iterable[tuple[str, int]] = ()
) -> dict[str, int]:
    """Our team behind each club name in `clubs`, for those that can be placed.

    `teams` is {team id: our name}; `aliases` are other spellings already on record as
    (text, team id). The sources' own spellings of a few of our names are known too
    (the Understat and ASA tables in their clients). A name matches on its accent- and
    case-folded text; failing that, on a close fuzzy match."""
    by_name: dict[str, int] = {}
    ours = {name: team_id for team_id, name in teams.items()}
    for team_id, name in teams.items():
        by_name.setdefault(normalize_name(name), team_id)
    for spellings in (TEAM_NAME_ALIASES, TEAM_NAME_OVERRIDES):
        for our_name, theirs in spellings.items():
            if our_name in ours:
                by_name.setdefault(normalize_name(theirs), ours[our_name])
    for text, team_id in aliases:
        if team_id in teams:
            by_name.setdefault(normalize_name(text), team_id)

    labels = sorted(by_name)
    found: dict[str, int] = {}
    for club in dict.fromkeys(clubs):
        wanted = normalize_name(club)
        if not wanted:
            continue
        team_id = by_name.get(wanted)
        if team_id is None:
            match = process.extractOne(wanted, labels, scorer=fuzz.WRatio, score_cutoff=CLUB_MATCH_THRESHOLD)
            team_id = by_name[match[0]] if match else None
        if team_id is not None:
            found[club] = team_id
    return found


def player_page_href(league_code: str, name: str, club: str) -> str:
    """A link to the Players page with `name` (at `club`, one of our team names) already open, for a
    table on another page."""
    return f"/players?league={urllib.parse.quote(league_code)}&{PLAYER_PARAM}={urllib.parse.quote(player_key(name, club), safe='')}"


def _player_link(name: str, club: str, league_code: str, selected: str | None) -> str:
    key = player_key(name, club)
    href = f"?league={urllib.parse.quote(league_code)}&{PLAYER_PARAM}={urllib.parse.quote(key, safe='')}"
    picked = " on" if key == selected else ""
    return (
        f'<a class="player{picked}" href="{html.escape(href)}" target="_self" '
        f'title="Show their stats below the table">{html.escape(name)}</a>'
    )


def _club_cell(club: str, team_id: int | None, league_code: str) -> TableCell:
    text = _text(club)
    if team_id is None:
        return TableCell(text, club.casefold())
    href = html.escape(team_page_href(league_code, team_id))
    return TableCell(f'<a class="club" href="{href}" target="_self">{text}</a>', club.casefold())


def _text(value: object) -> str:
    # "$" too: a table in st.markdown reads a pair of them (two salaries) as a formula.
    return html.escape(str(value)).replace("$", "&#36;")


def _number(text: object) -> float | None:
    """The number in a cell of the top table ("12", "0.85", "$1,200,000"); None for "–"."""
    try:
        value = float(str(text).replace("$", "").replace(",", ""))
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def _selected_index(keys: list[str], selected: str | None) -> int | None:
    return keys.index(selected) if selected in keys else None


def top_players_table_html(
    rows: list[dict], league_code: str, club_ids: dict[str, int], selected: str | None = None
) -> str:
    """The top-players list (ingest.top_players.table_rows) as a sortable table. The first
    columns are the rank, player and club; whatever follows (goals, assists, the ranked
    metric...) is a number. `selected` is the `player` parameter's value, to mark that row."""
    extra = [label for label in (rows[0] if rows else {}) if label not in ("#", "Player", "Club")]
    columns = [
        TableColumn("#", sort="num", first="asc"),
        TableColumn("Player", left=True, sort="text", first="asc", hint="Click to sort; click a name for their stats"),
        TableColumn("Club", left=True, sort="text", first="asc", hint="Click to sort; click a club for the team's page"),
        *(TableColumn(label, sort="num") for label in extra),
    ]
    table_rows, keys = [], []
    for row in rows:
        keys.append(player_key(row["Player"], row["Club"]))
        table_rows.append(
            [
                TableCell(str(row["#"]), row["#"]),
                TableCell(_player_link(row["Player"], row["Club"], league_code, selected), row["Player"].casefold()),
                _club_cell(row["Club"], club_ids.get(row["Club"]), league_code),
                *(TableCell(_text(row[label]), _number(row[label])) for label in extra),
            ]
        )
    return sortable_table_html(columns, table_rows, selected=_selected_index(keys, selected))


SEARCH_COLUMNS = [
    TableColumn("Name", left=True, sort="text", first="asc", hint="Click to sort; click a name for their stats"),
    TableColumn("Position", left=True, sort="text", first="asc"),
    TableColumn("Club", left=True, sort="text", first="asc", hint="Click to sort; click a club for the team's page"),
]


def search_results_table_html(
    results: list[PlayerSearchResult], league_code: str, selected: str | None = None
) -> str:
    """The squad players a name search found, as a sortable table in the same style."""
    table_rows, keys = [], []
    for result in results:
        player = result.player
        keys.append(player_key(player.name, result.team_name))
        table_rows.append(
            [
                TableCell(_player_link(player.name, result.team_name, league_code, selected), player.name.casefold()),
                TableCell(_text(player.position), player.position.casefold()),
                _club_cell(result.team_name, result.team_id, league_code),
            ]
        )
    return sortable_table_html(SEARCH_COLUMNS, table_rows, selected=_selected_index(keys, selected))
