"""Club injuries shown on the player's national team.

The injury feeds are club feeds, so Bukayo Saka is listed as injured for Arsenal and nowhere
else, though he is also one of England's players. This finds the club players on the injury
lists who play for a country and lists them under that country in the national competitions
(Nations League, World Cup, Euros, Asian Cup), as rows with source "club".

Who plays for a country comes from football-data.org's tournament squads (the World Cup and the
Euros, see squad.py): the same player keeps the same football-data.org id in his club's squad
and in his country's, so no nationality or spelling has to be matched. A country that was in
neither tournament, or a player not named in either squad, is not found: such a player is
simply not shown.

These rows are for display only. They carry a weight of 0, and prediction/service.py leaves
source "club" out of every forecast, so a country's ratings and predictions do not change. A
club suspension does not carry over to the national team and is left out; so are MLS clubs,
whose squads come from ESPN and have no football-data.org id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from soccer_predictor.config import League, load_leagues
from soccer_predictor.ingest import squad as squad_module
from soccer_predictor.ingest.top_players import normalize_name
from soccer_predictor.storage.repository import injuries_for_team, replace_injuries, teams_for_league

CLUB_SOURCE = "club"

# The same country spelt differently by different sources (ESPN's Nations League, football-data.org's
# World Cup, the Euros archive).
_NATION_ALIASES = {
    "turkey": "turkiye",
    "czech republic": "czechia",
    "republic of ireland": "ireland",
    "korea republic": "south korea",
    "ivory coast": "cote d'ivoire",
    "usa": "united states",
}


# Letters that do not come apart into a letter and an accent, which normalize_name would drop whole
# ("Odegaard" against "Ødegaard").
_SPECIAL_LETTERS = str.maketrans(
    {
        "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L",
        "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "ı": "i",
    }
)


def fold(text: str) -> str:
    """Lowercase, without accents or special letters, for comparing spellings."""
    return normalize_name(text.translate(_SPECIAL_LETTERS)).strip()


def nation_key(name: str) -> str:
    key = fold(name)
    return _NATION_ALIASES.get(key, key)


@dataclass(frozen=True)
class ClubInjured:
    player_id: int
    name: str
    club: str
    position: str  # "attack" | "defense"
    role: str | None
    note: str


def find_squad_player(name: str, players: list) -> object | None:
    """The one player of a club's squad that `name` (as a feed spells it: sometimes the whole
    name, sometimes only a surname or a short form) refers to; None when nobody or more than
    one could be meant."""
    wanted = fold(name)
    if not wanted:
        return None
    exact = [p for p in players if fold(p.name) == wanted]
    if exact:
        return exact[0]
    tokens = wanted.split()
    ends_with = [p for p in players if fold(p.name).endswith(" " + wanted)]
    candidates = ends_with or [p for p in players if all(t in fold(p.name).split() for t in tokens)]
    return candidates[0] if len(candidates) == 1 else None


_PREFIX = re.compile(r"^[A-Za-z0-9 ]{2,20}:\s+")


def club_note(original: str, club: str) -> str:
    """"FPL: Ankle injury - Expected back 24 Oct" as "Club: Ankle injury (Arsenal) - Expected back 24 Oct",
    which the injury table splits back into a problem and a return date as it does any other note."""
    body = _PREFIX.sub("", original or "").strip()
    first, separator, rest = body.partition(" - ")
    first = first.strip() or "Not specified"
    return f"Club: {first} ({club}){separator}{rest}"


def _is_suspension(note: str) -> bool:
    return "suspend" in (note or "").lower()


def fetch_national_squads(leagues: dict[str, League]) -> dict[str, set[int]]:
    """nation_key -> football-data.org ids of the players named in that country's tournament squads."""
    squads: dict[str, set[int]] = {}
    for league in leagues.values():
        if league.team_type != "national" or league.api_competition_id is None:
            continue
        for entry in squad_module._fetch_raw_teams(league):
            ids = {p["id"] for p in entry.get("squad") or [] if p.get("id") is not None}
            if ids and entry.get("name"):
                squads.setdefault(nation_key(entry["name"]), set()).update(ids)
    return squads


def collect_club_injured(session: Session, leagues: dict[str, League]) -> list[ClubInjured]:
    """Every automatically listed club injury (not suspensions) whose player is in a football-data.org squad."""
    found: dict[int, ClubInjured] = {}  # by player id: the same man is listed under his league and the Champions League
    for league in leagues.values():
        if league.team_type != "club" or league.api_competition_id is None or league.data_source == "espn":
            continue
        entries = squad_module._raw_entries_by_team_id(session, league)
        for team_id, team_name in teams_for_league(session, league.code).items():
            squad_players = squad_module.fetch_squad_for_team(session, league, team_id, entries) or []
            if not squad_players:
                continue
            for row in injuries_for_team(session, team_id):
                if row.source != "api" or _is_suspension(row.note):
                    continue
                player = find_squad_player(row.player_name, squad_players)
                if player is None or player.player_id is None or player.player_id in found:
                    continue
                found[player.player_id] = ClubInjured(
                    player_id=player.player_id,
                    name=player.name,
                    club=team_name,
                    position=row.position,
                    role=row.role,
                    note=row.note,
                )
    return list(found.values())


def sync_club_injuries_to_national_teams(session: Session, league: League) -> tuple[int, int]:
    """Lists the club injuries of each country's players under that country in `league` (a
    national competition). Returns (countries with someone listed, countries cleared);
    (0, 0) and nothing changed if no country's squad could be read -- yesterday's rows
    beat wiping them."""
    if league.team_type != "national":
        return 0, 0
    leagues = load_leagues()
    squads = fetch_national_squads(leagues)
    if not squads:
        return 0, 0
    injured = collect_club_injured(session, leagues)

    team_by_key = {nation_key(name): team_id for team_id, name in teams_for_league(session, league.code).items()}
    entries_by_team: dict[int, list[dict]] = {team_id: [] for team_id in team_by_key.values()}
    for player in injured:
        for key, ids in squads.items():
            team_id = team_by_key.get(key)
            if team_id is None or player.player_id not in ids:
                continue
            entries_by_team[team_id].append(
                {
                    "player_name": player.name,
                    "position": player.position,
                    "role": player.role,
                    "importance_weight": 0.0,  # display only: see the module docstring
                    "note": club_note(player.note, player.club),
                }
            )
    listed = cleared = 0
    for team_id, entries in entries_by_team.items():
        replace_injuries(session, team_id, source=CLUB_SOURCE, injuries=entries)
        listed, cleared = (listed + 1, cleared) if entries else (listed, cleared + 1)
    return listed, cleared
