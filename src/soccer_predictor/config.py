"""Loads .env and the YAML config files under config/."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "soccer.db"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class TableZone:
    """A position range on the league table with a promotion/relegation
    meaning -- e.g. positions 1-4 qualify for the Champions League.
    `kind` picks the marker shown: "qualify" (green), "playoff" (amber,
    e.g. Bundesliga's 16th-place relegation playoff spot), "relegation" (red).
    """

    start: int  # 1-indexed, inclusive
    end: int  # 1-indexed, inclusive
    kind: str
    label: str


@dataclass(frozen=True)
class League:
    code: str
    name: str
    seasons: list[str]
    # None for a competition with no football-data.co.uk historical CSV
    # (e.g. UEFA Champions League, European Championship) -- these rely
    # entirely on football-data.org's live API for match history, so they
    # never get a trained Dixon-Coles model (see supports_predictions).
    csv_code: str | None = None
    # None for a league football-data.org doesn't cover at all (e.g. MLS,
    # see data_source below) -- every football-data.org call site must
    # guard on this before building a URL from it.
    api_competition_id: int | None = None
    flag_url: str | None = None
    # A unicode flag emoji -- st.selectbox only renders plain text for its
    # options, so this is what shows next to the league name in the League
    # picker dropdown itself (flag_url is an <img>, only usable in the
    # markdown header below the picker, not inside the dropdown).
    flag_emoji: str | None = None
    zones: list[TableZone] = field(default_factory=list)
    # "range" (default): a season spans two calendar years, e.g. "2425" ->
    # 2024/25, the usual Aug-May domestic/UCL calendar. "single_year": the
    # season code IS the year, e.g. "2024" for Euro 2024 -- international
    # tournaments like the Euros run entirely within one calendar year and
    # only happen every 4 years, so they don't fit the two-year convention.
    season_display: str = "range"
    # "football_data_org" (default): fixtures/results/team-seeding come from
    # football-data.org (api_competition_id). "espn": from ESPN's public,
    # keyless site.api.espn.com endpoints instead (ingest/espn_client.py) --
    # used for leagues football-data.org doesn't cover at all, e.g. MLS.
    data_source: str = "football_data_org"
    # Only meaningful when data_source == "espn" -- ESPN's own league slug,
    # e.g. "usa.1" for MLS (verified live against site.api.espn.com; no
    # documented, stable list of these exists).
    espn_league_slug: str | None = None
    # Explicit opt-in for a league with a real historical-match source that
    # ISN'T a football-data.co.uk CSV (e.g. MLS via ESPN) -- see
    # supports_predictions. Defaults to False so every existing league's
    # behavior is unchanged unless a league's config explicitly sets this;
    # UEFA Champions League/Euros/World Cup deliberately leave it False even
    # though they also have a non-CSV live-API match history, since a
    # trained model was never wanted for them.
    trainable: bool = False

    @property
    def supports_predictions(self) -> bool:
        return self.csv_code is not None or self.trainable

    def api_season_year(self, season: str) -> int:
        """Converts one of this league's season codes to a live API's
        `season=` query value (a single start year) -- shared by
        football-data.org and ESPN, which both use this convention."""
        if self.season_display == "single_year":
            return int(season)
        return int(f"20{season[:2]}")


@dataclass(frozen=True)
class TeamAlias:
    canonical_name: str
    csv_name: str
    api_name: str
    league: str


@dataclass(frozen=True)
class ManualInjury:
    team: str
    player: str
    position: str  # "attack" | "defense"
    importance_weight: float
    note: str = ""


@dataclass(frozen=True)
class ManualCaptain:
    team: str
    player: str


@dataclass(frozen=True)
class ManualStarPlayer:
    team: str
    player: str
    note: str = ""


@lru_cache
def load_leagues() -> dict[str, League]:
    raw = yaml.safe_load((CONFIG_DIR / "leagues.yaml").read_text(encoding="utf-8"))
    return {
        entry["code"]: League(
            code=entry["code"],
            name=entry["name"],
            csv_code=entry.get("csv_code"),
            api_competition_id=entry.get("api_competition_id"),
            seasons=list(entry["seasons"]),
            flag_url=entry.get("flag_url"),
            flag_emoji=entry.get("flag_emoji"),
            zones=[
                TableZone(start=z["start"], end=z["end"], kind=z["kind"], label=z["label"])
                for z in entry.get("zones", [])
            ],
            season_display=entry.get("season_display", "range"),
            data_source=entry.get("data_source", "football_data_org"),
            espn_league_slug=entry.get("espn_league_slug"),
            trainable=entry.get("trainable", False),
        )
        for entry in raw["leagues"]
    }


@lru_cache
def load_team_aliases() -> list[TeamAlias]:
    raw = yaml.safe_load((CONFIG_DIR / "team_aliases.yaml").read_text(encoding="utf-8"))
    return [
        TeamAlias(
            canonical_name=entry["canonical_name"],
            csv_name=entry["csv"],
            api_name=entry["api"],
            league=entry["league"],
        )
        for entry in raw["teams"]
    ]


def load_manual_injuries() -> list[ManualInjury]:
    # Not cached: this file is meant to be hand-edited between dashboard runs.
    raw = yaml.safe_load((CONFIG_DIR / "injuries.yaml").read_text(encoding="utf-8"))
    return [
        ManualInjury(
            team=entry["team"],
            player=entry["player"],
            position=entry["position"],
            importance_weight=float(entry["importance_weight"]),
            note=entry.get("note", ""),
        )
        for entry in (raw.get("injuries") or [])
    ]


def load_manual_captains() -> list[ManualCaptain]:
    # Not cached: this file is meant to be hand-edited between dashboard runs.
    raw = yaml.safe_load((CONFIG_DIR / "captains.yaml").read_text(encoding="utf-8"))
    return [
        ManualCaptain(team=entry["team"], player=entry["player"]) for entry in (raw.get("captains") or [])
    ]


def load_manual_star_players() -> list[ManualStarPlayer]:
    # Not cached: this file is meant to be hand-edited between dashboard runs.
    raw = yaml.safe_load((CONFIG_DIR / "star_players.yaml").read_text(encoding="utf-8"))
    return [
        ManualStarPlayer(team=entry["team"], player=entry["player"], note=entry.get("note", ""))
        for entry in (raw.get("star_players") or [])
    ]


def football_data_org_api_key() -> str | None:
    return os.environ.get("FOOTBALL_DATA_ORG_API_KEY") or None


def api_football_key() -> str | None:
    return os.environ.get("API_FOOTBALL_KEY") or None


DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"


def anthropic_api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or None


def turso_database_url() -> str | None:
    """Just the host (e.g. "my-db-myorg.turso.io"), not a full URL -- the
    scheme is hardcoded in storage/db.py's connection string. Unset (None)
    means "use the local SQLite file," the default everywhere except a
    deployment where local disk doesn't persist (e.g. Streamlit Community
    Cloud) -- see storage/db.py::get_engine.
    """
    return os.environ.get("TURSO_DATABASE_URL") or None


def turso_auth_token() -> str | None:
    return os.environ.get("TURSO_AUTH_TOKEN") or None


def admin_password() -> str | None:
    """Unset (None) keeps dashboard/views/admin.py's hidden data-refresh
    page fully disabled -- it's off unless explicitly configured, matching
    every other optional capability in this app.
    """
    return os.environ.get("ADMIN_PASSWORD") or None


def claude_model() -> str:
    return os.environ.get("SOCCER_PREDICTOR_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL
