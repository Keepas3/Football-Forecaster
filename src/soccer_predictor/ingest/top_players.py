"""The Players page's default list: a league's top players, ranked by goals,
assists, goals + assists, xG + xA per 90, or (MLS only) salary.

Which statistics exist depends on the source, and all are free and already
integrated -- nothing here needs a new key or runs on a schedule:

* EPL, La Liga, Bundesliga, Serie A, Ligue 1 -- Understat: goals, assists, xG,
  xA and minutes for the whole league in one cached call.
* MLS -- American Soccer Analysis: the same, plus salaries (MLS Players
  Association release), in three cached calls.
* Champions League, World Cup, Euros -- football-data.org's top-scorers list
  (one call, within its 10 requests/minute): goals and assists.
* Nations League, Asian Cup -- ESPN's league statistics (one call): goals and
  assists.

There is no free, legitimate source of player ratings or contract/market values
outside MLS (the sites that publish them forbid scraping), so the page says so
rather than inventing them.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import requests

from soccer_predictor.config import League
from soccer_predictor.ingest import api_client, asa_client, espn_client, understat_client
from soccer_predictor.ingest.memo import ttl_memo

TOP_N = 15
# Per-90 rates need a real sample: a player with one 20-minute cameo and a
# goal would otherwise top the list.
MIN_MINUTES_PER_90 = 450
SCORERS_CACHE_TTL_SECONDS = 6 * 3600

GOALS = "goals"
ASSISTS = "assists"
GOALS_PLUS_ASSISTS = "goals_assists"
XG_XA_PER_90 = "xg_xa_90"
SALARY = "salary"
METRIC_LABELS = {
    GOALS: "Goals",
    ASSISTS: "Assists",
    GOALS_PLUS_ASSISTS: "Goals + assists",
    XG_XA_PER_90: "xG + xA per 90",
    SALARY: "Salary (guaranteed)",
}


@dataclass
class TopPlayer:
    name: str
    team: str
    goals: int
    assists: int
    minutes: int | None = None  # None: the source gives no minutes
    xg: float | None = None
    xa: float | None = None
    salary: float | None = None

    @property
    def goals_plus_assists(self) -> int:
        return self.goals + self.assists

    @property
    def xg_xa_per_90(self) -> float | None:
        if self.xg is None or self.xa is None or not self.minutes or self.minutes < MIN_MINUTES_PER_90:
            return None
        return (self.xg + self.xa) * 90 / self.minutes


@dataclass
class TopPlayers:
    players: list[TopPlayer]
    source: str  # shown under the table, e.g. "Understat"
    season: int | None  # start year the data is for, when the source says


def metric_value(player: TopPlayer, metric: str) -> float | None:
    return {
        GOALS: player.goals,
        ASSISTS: player.assists,
        GOALS_PLUS_ASSISTS: player.goals_plus_assists,
        XG_XA_PER_90: player.xg_xa_per_90,
        SALARY: player.salary,
    }[metric]


def available_metrics(players: list[TopPlayer]) -> list[str]:
    """The metrics this league's data can rank by (goals, assists and their sum
    always; xG per 90 and salary only where the source has them)."""
    metrics = [GOALS, ASSISTS, GOALS_PLUS_ASSISTS]
    if any(p.xg_xa_per_90 is not None for p in players):
        metrics.append(XG_XA_PER_90)
    if any(p.salary is not None for p in players):
        metrics.append(SALARY)
    return metrics


def rank_players(players: list[TopPlayer], metric: str, limit: int = TOP_N) -> list[TopPlayer]:
    """Best first by `metric`; players with no value for it are left out, ties
    broken by the other scoring numbers so the order is stable."""
    ranked = [p for p in players if metric_value(p, metric) is not None]
    ranked.sort(key=lambda p: (-metric_value(p, metric), -p.goals, -p.assists, p.name))
    return ranked[:limit]


def normalize_name(text: str) -> str:
    """Lowercase without accents, so Understat's "Julian Alvarez" matches a
    squad's "Julián Álvarez"."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()


def per_90(value: float, minutes: int | None) -> float | None:
    """`value` as a rate per 90 minutes; None without a real sample (one game's
    worth of minutes), where the rate would only be noise."""
    if not minutes or minutes < 90:
        return None
    return value * 90 / minutes


def find_top_player(players: list[TopPlayer], name: str, team: str = "") -> TopPlayer | None:
    """The listing's entry for a player by name (same club preferred)."""
    matches = [p for p in players if normalize_name(p.name) == normalize_name(name)]
    same_club = [p for p in matches if team and normalize_name(p.team) == normalize_name(team)]
    return (same_club or matches or [None])[0]


def league_rankings(players: list[TopPlayer], player: TopPlayer) -> list[tuple[str, int, int]]:
    """Where this player stands among the whole league's players, as
    (label, rank, how many are ranked) for each metric the source has. Ties
    share a rank (1 + the number strictly ahead)."""
    standings = []
    for metric in (GOALS, ASSISTS, GOALS_PLUS_ASSISTS, XG_XA_PER_90, SALARY):
        mine = metric_value(player, metric)
        if mine is None:
            continue
        values = [v for p in players if (v := metric_value(p, metric)) is not None]
        standings.append((METRIC_LABELS[metric], 1 + sum(v > mine for v in values), len(values)))
    return standings


def format_metric(metric: str, value: float | None) -> str:
    if value is None:
        return "–"
    if metric == SALARY:
        return f"${value:,.0f}"
    if metric == XG_XA_PER_90:
        return f"{value:.2f}"
    return f"{value:.0f}"


def table_rows(players: list[TopPlayer], metric: str) -> list[dict]:
    """The page's table: rank, player, club, goals and assists, plus the ranked
    metric's own column when it isn't one of those."""
    rows = []
    for rank, player in enumerate(rank_players(players, metric), start=1):
        row = {"#": rank, "Player": player.name, "Club": player.team, "Goals": player.goals, "Assists": player.assists}
        if metric in (XG_XA_PER_90, SALARY):
            row[METRIC_LABELS[metric]] = format_metric(metric, metric_value(player, metric))
        elif metric == GOALS_PLUS_ASSISTS:
            row["G + A"] = player.goals_plus_assists
        rows.append(row)
    return rows


@ttl_memo(SCORERS_CACHE_TTL_SECONDS)
def _football_data_scorers(competition_id: int) -> list[TopPlayer]:
    try:
        data = api_client.get(
            f"/competitions/{competition_id}/scorers",
            params={"limit": 30},
            cache_ttl_seconds=SCORERS_CACHE_TTL_SECONDS,
        )
    except (api_client.MissingApiKey, requests.RequestException):
        return []
    players = []
    for entry in data.get("scorers") or []:
        try:
            players.append(
                TopPlayer(
                    name=entry["player"]["name"],
                    team=(entry.get("team") or {}).get("name") or "",
                    goals=int(entry.get("goals") or 0),
                    assists=int(entry.get("assists") or 0),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return players


def fetch_top_players(league: League) -> TopPlayers:
    """The league's players with their numbers, from whichever source covers
    it (see the module docstring). Empty list when there is none or it fails."""
    season = league.api_season_year(league.seasons[-1])

    if league.code in understat_client.UNDERSTAT_LEAGUE_SLUG:
        rows = understat_client.fetch_league_players(league.code, season)
        if not rows:  # a new season with nothing played yet: show last season's
            season -= 1
            rows = understat_client.fetch_league_players(league.code, season)
        return TopPlayers(
            [
                TopPlayer(r.name, r.team_title, r.goals, r.assists, r.minutes, r.xg, r.xa)
                for r in rows
            ],
            "Understat",
            season,
        )

    if league.code == "MLS":
        rows = asa_client.fetch_league_players(season)
        return TopPlayers(
            [TopPlayer(r.name, r.team, r.goals, r.assists, r.minutes, r.xg, r.xa, r.salary) for r in rows],
            "American Soccer Analysis (salaries: MLS Players Association)",
            season,
        )

    if league.api_competition_id is not None:
        return TopPlayers(
            _football_data_scorers(league.api_competition_id), "football-data.org (top scorers)", None
        )

    if league.espn_league_slug:
        # The current edition first; a tournament that has not started yet
        # (the Asian Cup between editions) has none, so fall back to the last
        # two editions played.
        previous = [league.api_season_year(s) for s in reversed(league.seasons[:-1])][:2]
        for edition in (None, *previous):
            leaders = espn_client.fetch_stat_leaders(league.espn_league_slug, edition)
            if leaders:
                players = [TopPlayer(l.name, l.team, l.goals, l.assists) for l in leaders]
                return TopPlayers(players, "ESPN", edition)

    return TopPlayers([], "", None)
