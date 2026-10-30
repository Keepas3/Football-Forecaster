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
    # (e.g. UEFA Champions League, European Championship) -- these get their
    # match history from football-data.org's API or a static archive instead
    # (see data_source / trainable / supports_predictions).
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
    # "archive_worldcup"/"archive_euro": HISTORICAL match/goal data comes
    # from a static archive (ingest/worldcup_archive.py,
    # ingest/euro_archive.py) instead -- football-data.org's free tier 403s
    # on every past WC/EURO season. These two still keep api_competition_id
    # set and go through the normal football-data.org path for their
    # current/next (not-yet-archived) tournament's live fixtures.
    data_source: str = "football_data_org"
    # ESPN's own league slug, e.g. "usa.1" for MLS (verified live against
    # site.api.espn.com; no documented, stable list of these exists). Required
    # when data_source == "espn"; for every other league it only switches on
    # the match detail page (lineups, team stats, timeline), which looks the
    # match up on ESPN by team names.
    espn_league_slug: str | None = None
    # Explicit opt-in for a league with a real historical-match source that
    # ISN'T a football-data.co.uk CSV (e.g. MLS via ESPN, World Cup/Euro via
    # their static archives -- see data_source) -- see supports_predictions.
    # Defaults to False so every existing league's behavior is unchanged
    # unless a league's config explicitly sets this.
    trainable: bool = False
    # Past seasons that exist only to train the model: stored by
    # scripts/fetch_historical_data.py but never offered in the season picker or
    # shown in the standings, so `seasons` stays what a visitor can browse. The
    # Champions League keeps its pre-2024 eight-group seasons here -- a flat
    # league-phase table cannot show them, yet they still tell the ratings how
    # strong a club has been. Season codes use the same format as `seasons`.
    history_seasons: list[str] = field(default_factory=list)
    # Search term for the Latest news section (ingest/news.py) when `name`
    # isn't what news sites actually call the competition, e.g. "English
    # Premier League" -> "Premier League". None falls back to `name`.
    news_query: str | None = None
    # True for a group competition (Nations League, World Cup, Euros): the
    # Leagues page shows one standings table per group (Match.group_name)
    # instead of a single overall ranking, and ingest records each match's
    # group. Off by default -- and kept off for MLS, whose ESPN "groups" are
    # its two conferences, which most matches cross (see Team.conference).
    has_groups: bool = False
    # How the Dixon-Coles ratings are fit (prediction/training.py). All three
    # default to "the standard fit": the football-data.co.uk leagues are tuned
    # fine as they are, but national teams play so rarely that the default
    # one-year memory throws away most of what is known about them.
    # `rating_decay_xi`: recency-weighting rate (model/time_weighting.py);
    # None keeps DEFAULT_XI (half-life ~1 year); smaller = longer memory.
    rating_decay_xi: float | None = None
    # `rating_ridge`: smoothing that pulls each team's rating toward the
    # average -- useful when teams have few matches (see dixon_coles.fit_league).
    rating_ridge: float = 0.0
    # `division_prior`: pull each team toward its own division's average
    # instead (Nations League League A-D, read from Match.group_name).
    division_prior: bool = False
    # Nudge each side's rating for its last six results (model/results_form_adjustment.py).
    # Opt-in: only leagues it was tested on (the five football-data.co.uk
    # leagues and MLS) -- not the Champions League or the national-team
    # competitions, where six results can span most of a year.
    recent_results_form: bool = False
    # What a "team" is in this competition: "club" (domestic leagues, the
    # Champions League, MLS) or "national" (World Cup, Euros, Nations League,
    # Asian Cup). Teams are only ever linked to the same club/country in other
    # competitions of the same kind (storage/team_links.py).
    team_type: str = "club"

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
            history_seasons=list(entry.get("history_seasons", [])),
            news_query=entry.get("news_query"),
            has_groups=entry.get("has_groups", False),
            rating_decay_xi=entry.get("rating_decay_xi"),
            rating_ridge=float(entry.get("rating_ridge", 0.0)),
            division_prior=entry.get("division_prior", False),
            recent_results_form=entry.get("recent_results_form", False),
            team_type=entry.get("team_type", "club"),
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


_yaml_by_mtime: dict[Path, tuple[tuple[int, int], dict]] = {}


def _load_hand_edited_yaml(path: Path) -> dict:
    """Parses a YAML file that is meant to be hand-edited between dashboard
    runs, re-reading it only when it has changed on disk. (Every prediction
    looks at the manual injuries, so parsing the file each time added up.)
    The returned dict is shared -- callers only read it."""
    stat = path.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _yaml_by_mtime.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _yaml_by_mtime[path] = (signature, raw)
    return raw


def load_manual_injuries() -> list[ManualInjury]:
    raw = _load_hand_edited_yaml(CONFIG_DIR / "injuries.yaml")
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
    raw = _load_hand_edited_yaml(CONFIG_DIR / "captains.yaml")
    return [
        ManualCaptain(team=entry["team"], player=entry["player"]) for entry in (raw.get("captains") or [])
    ]


def load_manual_star_players() -> list[ManualStarPlayer]:
    raw = _load_hand_edited_yaml(CONFIG_DIR / "star_players.yaml")
    return [
        ManualStarPlayer(team=entry["team"], player=entry["player"], note=entry.get("note", ""))
        for entry in (raw.get("star_players") or [])
    ]


def football_data_org_api_key() -> str | None:
    return os.environ.get("FOOTBALL_DATA_ORG_API_KEY") or None


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


def notes_password() -> str | None:
    """Password for the Predictor page's Notes chat, which spends the app
    owner's Anthropic credits on every note (see dashboard/notes_access.py).
    Unset or blank (None) keeps the chat disabled for everyone -- off unless
    explicitly configured, like admin_password. Anyone can still read the
    saved notes.
    """
    return os.environ.get("NOTES_PASSWORD") or None


def claude_model() -> str:
    return os.environ.get("SOCCER_PREDICTOR_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL


# A model on Google's free tier (checked against ai.google.dev's pricing page);
# set SOCCER_PREDICTOR_GEMINI_MODEL to move to another without a code change.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def gemini_api_key() -> str | None:
    """Free-tier Google AI Studio key for the Predictor page's public "AI
    what-if" tab (ai/whatif_parser.py). Unset (None) turns that tab off."""
    return os.environ.get("GEMINI_API_KEY") or None


def gemini_model() -> str:
    return os.environ.get("SOCCER_PREDICTOR_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
