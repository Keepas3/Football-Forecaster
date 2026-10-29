"""The Injuries tab's data: one row per unavailable player with real columns
(team, status, what's wrong, chance of playing, expected back, how much the
model discounts the team for it), instead of a sentence per player. The feeds
store their wording as one note string ("FPL: Hip injury - 75% chance of
playing", "FPL: Suspended until 29 Nov"), so it is split back into columns here.

Pure functions, no Streamlit.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import pandas as pd

from soccer_predictor.model.injury_adjustment import adjust_strength

OUT, DOUBTFUL, SUSPENDED = "Out", "Doubtful", "Suspended"
STATUS_ORDER = [OUT, DOUBTFUL, SUSPENDED]
SOURCE_LABELS = {"api": "Automatic feed", "manual": "Manual list", "chat": "AI note"}

PLAYER_COLUMNS = [
    "Team", "Player", "Status", "Problem", "Chance of playing", "Expected back", "Position", "Impact", "Source",
]

_CHANCE = re.compile(r"(\d{1,3})\s*% chance of playing", re.IGNORECASE)
_BACK = re.compile(
    r"(?:expected back|suspended until|back)\s+(\d{1,2})\s+([A-Za-z]{3})[a-z]*(?:\s+(\d{4}))?", re.IGNORECASE
)
# "Expected back mid-November": a feed that only says roughly when.
_BACK_ROUGHLY = re.compile(r"(?:expected back|back)\s+(early|mid|late|end of)[\s-]+([A-Za-z]{3})", re.IGNORECASE)
_ROUGH_DAY = {"early": 5, "mid": 15, "late": 25, "end of": 28}
_UNCERTAIN_WORDS = ("doubt", "day-to-day", "questionable")


@dataclass
class ParsedNote:
    status: str
    problem: str
    chance: int | None  # 0-100, None if the source gave none
    back: dt.date | None


def _day_month(day: str, month: str, today: dt.date, year: str | None = None) -> dt.date | None:
    """"1", "Nov" -> the next such date (most feeds omit the year): a date more
    than two months back is taken to be next year's. A year the feed does give is
    taken as it is."""
    try:
        parsed = dt.datetime.strptime(f"{day} {month.title()} {year or today.year}", "%d %b %Y").date()
    except ValueError:
        return None
    if year:
        return parsed
    return parsed.replace(year=today.year + 1) if parsed < today - dt.timedelta(days=60) else parsed


def parse_note(note: str, today: dt.date, return_date: dt.date | None = None) -> ParsedNote:
    """Status, problem, chance of playing and return date out of a feed's note
    string. A stored return date (manual and AI notes carry one) wins over the
    note's own."""
    text = (note or "").strip()
    text = re.sub(r"^[A-Za-z ]{2,20}:\s+", "", text)  # a source prefix such as "FPL: "
    lowered = text.lower()

    chance_match = _CHANCE.search(text)
    chance = int(chance_match.group(1)) if chance_match else None
    back_match = _BACK.search(text)
    rough_match = _BACK_ROUGHLY.search(text)
    back = None
    if back_match:
        back = _day_month(back_match.group(1), back_match.group(2), today, back_match.group(3))
    elif rough_match:
        back = _day_month(str(_ROUGH_DAY[rough_match.group(1).lower()]), rough_match.group(2), today)

    if "suspend" in lowered:
        status = SUSPENDED
    elif chance is not None or any(word in lowered for word in _UNCERTAIN_WORDS):
        status = DOUBTFUL
    else:
        status = OUT

    first = text.split(" - ")[0].strip()
    if not first or first.lower().startswith("suspended") or _CHANCE.search(first):
        problem = "Suspension" if status == SUSPENDED else "Not specified"
    else:
        problem = first[:80]
    if status == OUT and chance is None:
        chance = 0
    if status == SUSPENDED:
        chance = 0
    return ParsedNote(status, problem, chance, return_date or back)


def player_rows(team_name: str, team_injuries, today: dt.date) -> list[dict]:
    """One row per player in a service.TeamInjuries."""
    rows = []
    for entry, source, return_date, note in zip(
        team_injuries.entries, team_injuries.sources, team_injuries.expected_return_dates, team_injuries.notes
    ):
        parsed = parse_note(note, today, return_date)
        rows.append(
            {
                "Team": team_name,
                "Player": entry.player_name,
                "Status": parsed.status,
                "Problem": parsed.problem,
                "Chance of playing": parsed.chance,
                "Expected back": parsed.back,
                "Position": "Defence" if entry.position == "defense" else "Attack",
                "Impact": round(entry.importance_weight, 3),
                "Source": SOURCE_LABELS.get(source, source),
            }
        )
    return rows


def players_frame(rows: list[dict]) -> pd.DataFrame:
    """The rows as a table, biggest impact first within each team."""
    frame = pd.DataFrame(rows, columns=PLAYER_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(["Team", "Impact"], ascending=[True, False], kind="stable").reset_index(drop=True)


def filter_players(
    frame: pd.DataFrame,
    teams: list[str] | None = None,
    statuses: list[str] | None = None,
    positions: list[str] | None = None,
    query: str = "",
) -> pd.DataFrame:
    """Rows matching every filter given (empty or None means no filter)."""
    mask = pd.Series(True, index=frame.index)
    if teams:
        mask &= frame["Team"].isin(teams)
    if statuses:
        mask &= frame["Status"].isin(statuses)
    if positions:
        mask &= frame["Position"].isin(positions)
    if query.strip():
        mask &= frame["Player"].str.contains(query.strip(), case=False, na=False, regex=False)
    return frame[mask].reset_index(drop=True)


def team_impact_frame(team_ratings: dict[str, tuple[float, float, list]]) -> pd.DataFrame:
    """Per team with anyone listed: how many, and the model's attack and
    defence before and after the injuries. `team_ratings` maps team name to
    (base attack, base defence, [InjuryEntry]). Defence is goals conceded, so
    a rise there is worse."""
    rows = []
    for team, (attack, defense, entries) in team_ratings.items():
        if not entries:
            continue
        adj_attack, adj_defense = adjust_strength(attack, defense, entries)
        rows.append(
            {
                "Team": team,
                "Players listed": len(entries),
                "Attack": round(attack, 2),
                "Attack with injuries": round(adj_attack, 2),
                "Attack change": round((adj_attack / attack - 1) * 100, 1) if attack else 0.0,
                "Defence (goals conceded)": round(defense, 2),
                "Defence with injuries": round(adj_defense, 2),
                "Defence change": round((adj_defense / defense - 1) * 100, 1) if defense else 0.0,
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "Team", "Players listed", "Attack", "Attack with injuries", "Attack change",
            "Defence (goals conceded)", "Defence with injuries", "Defence change",
        ],
    )
    if frame.empty:
        return frame
    frame["_size"] = frame["Attack change"].abs() + frame["Defence change"].abs()
    return frame.sort_values("_size", ascending=False, kind="stable").drop(columns="_size").reset_index(drop=True)


def summary_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "players": len(frame),
        "teams": int(frame["Team"].nunique()) if len(frame) else 0,
        OUT: int((frame["Status"] == OUT).sum()) if len(frame) else 0,
        DOUBTFUL: int((frame["Status"] == DOUBTFUL).sum()) if len(frame) else 0,
        SUSPENDED: int((frame["Status"] == SUSPENDED).sum()) if len(frame) else 0,
    }
