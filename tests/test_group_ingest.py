"""Group labels on ingested matches: the World Cup and Euro archives,
football-data.org, the league config flag, the repository upserts, and the
automatic schema upgrade for databases that predate Match.group_name."""

from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from soccer_predictor.config import League, load_leagues
from soccer_predictor.ingest import euro_archive, worldcup_archive
from soccer_predictor.ingest.fixtures import _football_data_group
from soccer_predictor.storage import db
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import (
    get_or_create_team,
    matches_for_league,
    set_match_group_names,
    upsert_match,
)


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


# --- config ----------------------------------------------------------------------


def test_has_groups_defaults_to_false():
    league = League(code="EPL", name="English Premier League", seasons=["2425"], csv_code="E0")
    assert league.has_groups is False


def test_nations_league_is_configured_as_a_trainable_espn_group_competition():
    nl = load_leagues()["NL"]
    assert nl.data_source == "espn"
    assert nl.espn_league_slug == "uefa.nations"
    assert nl.has_groups is True
    assert nl.supports_predictions is True
    # Editions are biennial and ESPN keys them by start year.
    assert [nl.api_season_year(s) for s in nl.seasons] == [2018, 2020, 2022, 2024, 2026]


def test_world_cup_and_euros_use_group_tables_but_leagues_and_mls_do_not():
    leagues = load_leagues()
    assert leagues["WC"].has_groups and leagues["EURO"].has_groups
    assert not leagues["EPL"].has_groups
    assert not leagues["MLS"].has_groups  # its ESPN "groups" are conferences most matches cross
    assert not leagues["UCL"].has_groups


# --- World Cup archive ---------------------------------------------------------------


def _wc_df(rows):
    base = {
        "tournament_id": "WC-2022",
        "tournament_name": "2022 FIFA Men's World Cup",
        "match_date": "2022-11-22",
        "home_team_name": "Argentina",
        "away_team_name": "Saudi Arabia",
        "home_team_score": 1,
        "away_team_score": 2,
        "stage_name": "group stage",
        "group_name": "Group C",
        "group_stage": 1,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_world_cup_group_stage_matches_get_their_group():
    out = worldcup_archive.parse_matches(_wc_df([{}]))
    assert out.iloc[0]["group_name"] == "Group C"


def test_world_cup_knockout_matches_have_no_group():
    df = _wc_df([{"stage_name": "final", "group_name": "not applicable", "group_stage": 0}])
    assert worldcup_archive.parse_matches(df).iloc[0]["group_name"] is None


def test_world_cup_second_group_stage_is_kept_apart_from_the_first():
    # 1974/78/82 reuse "Group A" in their second round-robin stage.
    df = _wc_df(
        [
            {"stage_name": "group stage", "group_name": "Group A"},
            {"stage_name": "second group stage", "group_name": "Group A"},
        ]
    )
    labels = worldcup_archive.parse_matches(df)["group_name"].tolist()
    assert labels == ["Group A", "Second Group Stage - Group A"]


def test_world_cup_1950_final_round_is_a_single_table():
    df = _wc_df([{"stage_name": "final round", "group_name": "not applicable"}])
    assert worldcup_archive.parse_matches(df).iloc[0]["group_name"] == "Final Round"


# --- Euro archive -----------------------------------------------------------------------

EURO_TEXT = """\
Group A  |  Germany   Scotland     Hungary   Switzerland
Group B  |  Spain     Croatia      Italy     Albania

▪ Matchday 1 | Fri Jun 14 - Tue Jun 18

▪ Group A
Fri Jun 14
  21:00         Germany   5-1   Scotland     @ Munchen

▪ Group B
Sat Jun 15
  18:00         Spain   3-0   Croatia     @ Berlin

▪ Round of 16
Sat Jun 29
  18:00         Switzerland   2-0   Italy     @ Berlin

▪ Final
Sun Jul 14
  21:00         Spain   2-1   England     @ Berlin
"""


def test_euro_group_matches_get_their_group_and_knockouts_do_not():
    matches = {(m["home_team_name"], m["away_team_name"]): m["group_name"] for m in euro_archive.parse_matches(EURO_TEXT, 2024)}
    assert matches[("Germany", "Scotland")] == "Group A"
    assert matches[("Spain", "Croatia")] == "Group B"
    assert matches[("Switzerland", "Italy")] is None
    assert matches[("Spain", "England")] is None


def test_euro_matchday_summary_line_does_not_end_or_start_a_group():
    # "▪ Matchday 1 | dates" precedes the groups -- it must not leave a
    # stale group on, nor clear one in the middle of a group's matches.
    text = "▪ Group A\n" + "▪ Matchday 2 | Jun 19\n" + "Fri Jun 14\n  21:00   Germany   5-1   Scotland\n"
    assert euro_archive.parse_matches(text, 2024)[0]["group_name"] == "Group A"


def test_euro_files_with_no_group_headers_have_no_groups():
    text = "▪ Semi-finals\nWed Jun 15\n  20:00   France   5-4   Portugal\n"
    assert euro_archive.parse_matches(text, 1960)[0]["group_name"] is None


# --- football-data.org --------------------------------------------------------------------

WC = League(code="WC", name="FIFA World Cup", seasons=["2026"], has_groups=True)
EPL = League(code="EPL", name="English Premier League", seasons=["2425"], csv_code="E0")


def test_football_data_group_stage_match_becomes_a_group_label():
    assert _football_data_group({"stage": "GROUP_STAGE", "group": "GROUP_A"}, WC) == "Group A"


def test_football_data_knockout_or_ungrouped_match_has_no_group():
    assert _football_data_group({"stage": "LAST_16", "group": None}, WC) is None
    assert _football_data_group({"stage": "GROUP_STAGE", "group": None}, WC) is None


def test_football_data_group_ignored_for_a_league_without_groups():
    assert _football_data_group({"stage": "GROUP_STAGE", "group": "GROUP_A"}, EPL) is None


# --- repository ---------------------------------------------------------------------------


def _two_teams(s):
    return get_or_create_team(s, "Spain", "NL").id, get_or_create_team(s, "Croatia", "NL").id


def _upsert(s, home, away, **kw):
    upsert_match(s, "NL", "2425", dt.date(2024, 9, 5), home, away, 2, 0, **kw)
    s.flush()


def test_upsert_match_stores_the_group_and_exposes_it_in_the_league_frame():
    with _session() as s:
        home, away = _two_teams(s)
        _upsert(s, home, away, group_name="Group A1")
        assert matches_for_league(s, "NL").iloc[0]["group_name"] == "Group A1"


def test_upsert_match_never_clears_an_existing_group():
    with _session() as s:
        home, away = _two_teams(s)
        _upsert(s, home, away, group_name="Group A1")
        _upsert(s, home, away)  # a later sync that doesn't know the group
        assert matches_for_league(s, "NL").iloc[0]["group_name"] == "Group A1"


def test_a_fallback_group_fills_a_missing_group_but_never_replaces_a_real_one():
    with _session() as s:
        home, away = _two_teams(s)
        _upsert(s, home, away)
        _upsert(s, home, away, group_name="League A - Group 1", group_is_fallback=True)
        assert matches_for_league(s, "NL").iloc[0]["group_name"] == "League A - Group 1"

        _upsert(s, home, away, group_name="Group A1")  # a real label wins
        _upsert(s, home, away, group_name="League A - Group 1", group_is_fallback=True)
        assert matches_for_league(s, "NL").iloc[0]["group_name"] == "Group A1"


def test_set_match_group_names_updates_only_matching_pairs():
    with _session() as s:
        home, away = _two_teams(s)
        _upsert(s, home, away)

        assert set_match_group_names(s, "NL", "2425", {(home, away): "Group B2"}) == 1
        assert set_match_group_names(s, "NL", "2425", {(home, away): "Group B2"}) == 0  # already set
        assert set_match_group_names(s, "NL", "2425", {(away, home): "Group Z"}) == 0  # other direction
        assert matches_for_league(s, "NL").iloc[0]["group_name"] == "Group B2"


# --- schema upgrade --------------------------------------------------------------------------


def test_ensure_group_name_columns_upgrades_an_old_database_and_is_idempotent():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        # The matches/fixtures tables as they were before group_name existed.
        conn.execute(text("CREATE TABLE matches (id INTEGER PRIMARY KEY, league_code VARCHAR)"))
        conn.execute(text("CREATE TABLE fixtures (id INTEGER PRIMARY KEY, league_code VARCHAR)"))

    assert db.ensure_group_name_columns(engine) == ["matches", "fixtures"]
    for table in ("matches", "fixtures"):
        assert "group_name" in {c["name"] for c in inspect(engine).get_columns(table)}
    assert db.ensure_group_name_columns(engine) == []
