from __future__ import annotations

import pandas as pd

from soccer_predictor.ingest import worldcup_archive


def _matches_df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "tournament_id": "WC-1930",
        "tournament_name": "1930 FIFA Men's World Cup",
        "match_date": "1930-07-13",
        "home_team_name": "France",
        "away_team_name": "Mexico",
        "home_team_score": 4,
        "away_team_score": 1,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def _goals_df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "tournament_id": "WC-1930",
        "tournament_name": "1930 FIFA Men's World Cup",
        "match_date": "1930-07-13",
        "player_team_name": "France",
        "given_name": "Lucien",
        "family_name": "Laurent",
        "minute_regulation": 19,
        "own_goal": 0,
        "penalty": 0,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_parse_matches_produces_canonical_columns():
    df = _matches_df([{}])
    out = worldcup_archive.parse_matches(df)

    assert list(out.columns) == ["date", "home_team_name", "away_team_name", "home_goals", "away_goals", "season"]
    row = out.iloc[0]
    assert row["home_team_name"] == "France"
    assert row["away_team_name"] == "Mexico"
    assert row["home_goals"] == 4
    assert row["away_goals"] == 1
    assert row["season"] == "1930"
    assert str(row["date"]) == "1930-07-13"


def test_parse_matches_excludes_womens_world_cup():
    df = _matches_df(
        [
            {},
            {
                "tournament_id": "WC-1991",
                "tournament_name": "1991 FIFA Women's World Cup",
                "home_team_name": "United States",
                "away_team_name": "Sweden",
            },
        ]
    )
    out = worldcup_archive.parse_matches(df)
    assert len(out) == 1
    assert out.iloc[0]["home_team_name"] == "France"


def test_parse_matches_derives_season_from_tournament_id():
    df = _matches_df([{"tournament_id": "WC-2022", "tournament_name": "2022 FIFA Men's World Cup"}])
    out = worldcup_archive.parse_matches(df)
    assert out.iloc[0]["season"] == "2022"


def test_parse_goals_hurst_hat_trick_produces_three_rows():
    df = _goals_df(
        [
            {
                "tournament_id": "WC-1966",
                "tournament_name": "1966 FIFA Men's World Cup",
                "match_date": "1966-07-30",
                "player_team_name": "England",
                "given_name": "Geoff",
                "family_name": "Hurst",
                "minute_regulation": 18,
            },
            {
                "tournament_id": "WC-1966",
                "tournament_name": "1966 FIFA Men's World Cup",
                "match_date": "1966-07-30",
                "player_team_name": "England",
                "given_name": "Geoff",
                "family_name": "Hurst",
                "minute_regulation": 101,
            },
            {
                "tournament_id": "WC-1966",
                "tournament_name": "1966 FIFA Men's World Cup",
                "match_date": "1966-07-30",
                "player_team_name": "England",
                "given_name": "Geoff",
                "family_name": "Hurst",
                "minute_regulation": 120,
            },
        ]
    )
    goals = worldcup_archive.parse_goals(df)

    assert len(goals) == 3
    assert all(g["player_name"] == "Geoff Hurst" for g in goals)
    assert all(g["team_name"] == "England" for g in goals)
    assert all(g["season"] == "1966" for g in goals)
    assert {g["minute"] for g in goals} == {18, 101, 120}


def test_parse_goals_excludes_womens_world_cup():
    df = _goals_df(
        [
            {},
            {
                "tournament_id": "WC-1991",
                "tournament_name": "1991 FIFA Women's World Cup",
                "player_team_name": "United States",
                "given_name": "Michelle",
                "family_name": "Akers",
            },
        ]
    )
    goals = worldcup_archive.parse_goals(df)
    assert len(goals) == 1
    assert goals[0]["player_name"] == "Lucien Laurent"


def test_parse_goals_handles_missing_given_name():
    df = _goals_df([{"given_name": "not applicable", "family_name": "Preguinho"}])
    goals = worldcup_archive.parse_goals(df)
    assert goals[0]["player_name"] == "Preguinho"


def test_parse_goals_carries_own_goal_and_penalty_flags():
    df = _goals_df([{"own_goal": 1, "penalty": 0}, {"own_goal": 0, "penalty": 1}])
    goals = worldcup_archive.parse_goals(df)
    assert goals[0]["own_goal"] is True
    assert goals[0]["penalty"] is False
    assert goals[1]["own_goal"] is False
    assert goals[1]["penalty"] is True


def test_parse_goals_own_goal_credited_to_scoring_players_own_team():
    # An own goal's "team_name" (the credited/beneficiary team) differs from
    # "player_team_name" (the scorer's own real team) -- this must key off
    # the latter, so the goal shows up on the scorer's OWN roster page, not
    # their opponent's.
    df = _goals_df([{"player_team_name": "Mexico", "own_goal": 1}])
    goals = worldcup_archive.parse_goals(df)
    assert goals[0]["team_name"] == "Mexico"
