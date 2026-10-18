"""The Players page's default top-players list: per-source parsing, ranking, page."""

from __future__ import annotations

import pytest
import requests

from soccer_predictor.config import League
from soccer_predictor.dashboard.views import players as players_view
from soccer_predictor.ingest import api_client, asa_client, espn_client, top_players, understat_client
from soccer_predictor.ingest.squad import PlayerSearchResult
from soccer_predictor.ingest.top_players import (
    GOALS,
    GOALS_PLUS_ASSISTS,
    SALARY,
    XG_XA_PER_90,
    TopPlayer,
    available_metrics,
    fetch_top_players,
    rank_players,
    table_rows,
)

EPL = League(code="EPL", name="English Premier League", seasons=["2526", "2627"], csv_code="E0")
MLS = League(code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1", season_display="single_year")
UCL = League(code="UCL", name="UEFA Champions League", seasons=["2627"], api_competition_id=2001)
NL = League(code="NL", name="UEFA Nations League", seasons=["2627"], data_source="espn", espn_league_slug="uefa.nations")
ASIA = League(
    code="ASIA", name="AFC Asian Cup", seasons=["2019", "2023", "2027"], data_source="espn",
    espn_league_slug="afc.asian.cup", season_display="single_year",
)


# --- ranking and metrics ---------------------------------------------------------------------


def _p(name, goals, assists, minutes=None, xg=None, xa=None, salary=None):
    return TopPlayer(name, "Club", goals, assists, minutes, xg, xa, salary)


def test_rank_by_goals_assists_and_their_sum_with_stable_tie_breaks():
    players = [_p("A", 5, 1), _p("B", 5, 3), _p("C", 2, 9), _p("D", 1, 0)]
    assert [p.name for p in rank_players(players, GOALS)] == ["B", "A", "C", "D"]  # tie on goals: more assists first
    assert [p.name for p in rank_players(players, GOALS_PLUS_ASSISTS)] == ["C", "B", "A", "D"]
    assert [p.name for p in rank_players(players, GOALS, limit=2)] == ["B", "A"]


def test_per_90_needs_a_real_sample_and_salary_only_ranks_players_who_have_one():
    players = [
        _p("Cameo", 1, 0, minutes=20, xg=0.9, xa=0.0),  # a huge per-90 rate over 20 minutes: ignored
        _p("Regular", 4, 2, minutes=900, xg=4.5, xa=1.8, salary=1_000_000),
        _p("NoSalary", 9, 9, minutes=900, xg=9.0, xa=9.0),
    ]
    assert [p.name for p in rank_players(players, XG_XA_PER_90)] == ["NoSalary", "Regular"]
    assert rank_players(players, XG_XA_PER_90)[1].xg_xa_per_90 == pytest.approx(0.63)
    assert [p.name for p in rank_players(players, SALARY)] == ["Regular"]


def test_available_metrics_depend_on_what_the_source_provides():
    assert available_metrics([_p("A", 1, 1)]) == [GOALS, "assists", GOALS_PLUS_ASSISTS]
    assert XG_XA_PER_90 in available_metrics([_p("A", 1, 1, minutes=900, xg=1.0, xa=1.0)])
    assert SALARY in available_metrics([_p("A", 1, 1, salary=5.0)])


def test_table_rows_add_the_ranked_metrics_own_column():
    players = [_p("A", 5, 1, minutes=900, xg=4.0, xa=1.0, salary=2_500_000.4)]
    assert table_rows(players, GOALS)[0] == {"#": 1, "Player": "A", "Club": "Club", "Goals": 5, "Assists": 1}
    assert table_rows(players, SALARY)[0]["Salary (guaranteed)"] == "$2,500,000"
    assert table_rows(players, XG_XA_PER_90)[0]["xG + xA per 90"] == "0.50"
    assert table_rows(players, GOALS_PLUS_ASSISTS)[0]["G + A"] == 6


# --- sources ---------------------------------------------------------------------------------


def test_understat_players_keep_the_current_club_of_a_mid_season_transfer(monkeypatch):
    rows = [
        {"player_name": "Mover", "team_title": "Leeds,Chelsea", "position": "F S", "time": "900", "goals": "4", "assists": "1", "xG": "3.2", "xA": "0.9"},
        {"player_name": "Stayer", "team_title": "Arsenal", "position": "M", "time": "800", "goals": "1", "assists": "5", "xG": "1.0", "xA": "4.0"},
    ]
    monkeypatch.setattr(understat_client, "_fetch_league_players", lambda slug, season: rows)
    result = fetch_top_players(EPL)
    assert [(p.name, p.team, p.goals) for p in result.players] == [("Mover", "Chelsea", 4), ("Stayer", "Arsenal", 1)]
    assert (result.source, result.season) == ("Understat", 2026)


def test_understat_falls_back_to_last_season_before_a_new_one_has_data(monkeypatch):
    seen = []

    def fetch(slug, season):
        seen.append(season)
        return [] if season == 2026 else [{"player_name": "X", "team_title": "T", "time": "90", "goals": "1", "assists": "0", "xG": "1", "xA": "0"}]

    monkeypatch.setattr(understat_client, "_fetch_league_players", fetch)
    result = fetch_top_players(EPL)
    assert seen == [2026, 2025] and result.season == 2025 and len(result.players) == 1


def test_asa_players_join_names_clubs_and_salaries_and_cope_with_traded_players(monkeypatch):
    xgoals = [
        {"player_id": "p1", "team_id": "t1", "general_position": "ST", "minutes_played": 1000, "goals": 10, "primary_assists": 3, "xgoals": 8.0, "xassists": 2.0, "points_added": 1.5},
        {"player_id": "p2", "team_id": ["t1", "t2"], "general_position": "CM", "minutes_played": 700, "goals": 1, "primary_assists": 6},
        {"player_id": "ghost", "team_id": "t1", "minutes_played": 100},  # no bio: skipped
    ]
    salaries = [{"player_id": "p1", "guaranteed_compensation": 1234567.0}, {"player_id": "p2", "guaranteed_compensation": None}]

    def cached(key, url, params):
        return salaries if "salaries" in url else xgoals

    monkeypatch.setattr(asa_client, "_get_cached_json", cached)
    monkeypatch.setattr(asa_client, "_fetch_all_players", lambda: [{"player_id": "p1", "player_name": "Striker"}, {"player_id": "p2", "player_name": "Midfielder"}])
    monkeypatch.setattr(asa_client, "_fetch_all_teams", lambda: [{"team_id": "t1", "team_name": "One FC"}, {"team_id": "t2", "team_name": "Two FC"}])

    players = fetch_top_players(MLS).players

    assert [(p.name, p.team, p.goals, p.assists, p.salary) for p in players] == [
        ("Striker", "One FC", 10, 3, 1234567.0),
        ("Midfielder", "Two FC", 1, 6, None),
    ]
    assert SALARY in available_metrics(players) and XG_XA_PER_90 in available_metrics(players)


def test_football_data_scorers_tolerate_missing_assists_and_failures(monkeypatch):
    payload = {
        "scorers": [
            {"player": {"name": "Haaland"}, "team": {"name": "Man City"}, "goals": 4, "assists": None},
            {"player": {"name": "Musiala"}, "team": {"name": "Bayern"}, "goals": 3, "assists": 2},
            {"team": {"name": "no player"}},
        ]
    }
    seen = {}
    monkeypatch.setattr(api_client, "get", lambda path, **kw: seen.update(path=path, **kw) or payload)
    result = fetch_top_players(UCL)
    assert [(p.name, p.team, p.goals, p.assists) for p in result.players] == [("Haaland", "Man City", 4, 0), ("Musiala", "Bayern", 3, 2)]
    assert seen["path"] == "/competitions/2001/scorers" and result.source.startswith("football-data.org")

    from soccer_predictor.ingest import memo  # the scorers are memoized: reset before each failure case

    memo.clear_all()

    def down(*a, **k):
        raise requests.RequestException("down")

    monkeypatch.setattr(api_client, "get", down)
    assert fetch_top_players(UCL).players == []
    memo.clear_all()
    monkeypatch.setattr(api_client, "get", lambda *a, **k: (_ for _ in ()).throw(api_client.MissingApiKey("no key")))
    assert fetch_top_players(UCL).players == []


def _leaders_payload():
    def leader(athlete_id, name, team, value):
        return {"value": value, "athlete": {"id": athlete_id, "displayName": name, "team": {"displayName": team}}}

    return {
        "stats": [
            {"name": "goalsLeaders", "leaders": [leader("1", "Kane", "England", 6.0), leader("2", "Haaland", "Norway", 5.0)]},
            {"name": "assistsLeaders", "leaders": [leader("2", "Haaland", "Norway", 3.0), leader("3", "Olise", "France", 4.0)]},
            {"name": "yellowCards", "leaders": [leader("9", "Ignored", "X", 9.0)]},
        ]
    }


def test_espn_leaders_merge_goals_and_assists_by_player(monkeypatch):
    monkeypatch.setattr(espn_client, "get", lambda *a, **k: _leaders_payload())
    leaders = {l.name: (l.team, l.goals, l.assists) for l in espn_client.fetch_stat_leaders("uefa.nations")}
    assert leaders == {"Kane": ("England", 6, 0), "Haaland": ("Norway", 5, 3), "Olise": ("France", 0, 4)}


def test_espn_leaders_empty_on_failure(monkeypatch):
    def down(*a, **k):
        raise requests.RequestException("down")

    monkeypatch.setattr(espn_client, "get", down)
    assert espn_client.fetch_stat_leaders("uefa.nations") == []


def test_a_tournament_between_editions_shows_the_last_edition_played(monkeypatch):
    asked = []

    def get(path, params=None, **kw):
        asked.append((params or {}).get("season"))
        return {"stats": []} if not params else _leaders_payload()

    monkeypatch.setattr(espn_client, "get", get)
    result = fetch_top_players(ASIA)
    assert asked == [None, 2023] and result.season == 2023 and result.source == "ESPN" and len(result.players) == 3
    result_nl = fetch_top_players(NL)  # the current Nations League edition is served straight away
    assert result_nl.season is None


# --- the page --------------------------------------------------------------------------------


def test_find_squad_entry_prefers_the_same_player_at_the_same_club(monkeypatch):
    class Squad:
        def __init__(self, name):
            self.name = name

    def search(session, league, query):
        pool = [
            PlayerSearchResult(1, "Wrong FC", Squad("Julian Alvarez")),
            PlayerSearchResult(2, "Atletico Madrid", Squad("Julián Álvarez")),
        ]
        return [r for r in pool if query.casefold() in r.player.name.casefold() or "alvarez" in query.casefold()]

    monkeypatch.setattr(players_view, "search_players_in_league", search)
    found = players_view.find_squad_entry(None, EPL, "Julian Alvarez", "Atletico Madrid")
    assert found.team_id == 2

    monkeypatch.setattr(players_view, "search_players_in_league", lambda *a: [])
    assert players_view.find_squad_entry(None, EPL, "Nobody At All", "X") is None


def test_players_page_shows_the_top_list_with_a_metric_per_source(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def app():
        from soccer_predictor.dashboard.views import players
        from soccer_predictor.ingest import top_players as tp
        from soccer_predictor.ingest.top_players import TopPlayer, TopPlayers

        tp.fetch_top_players = lambda league: TopPlayers(
            [TopPlayer("Lionel Messi", "Inter Miami CF", 25, 12, 2000, 20.0, 8.0, 28_333_332.0)], "American Soccer Analysis", 2026
        )
        players.fetch_top_players = tp.fetch_top_players
        players.render()

    at = AppTest.from_function(app, default_timeout=60)
    at.query_params["league"] = "MLS"
    at.run()
    assert not at.exception
    assert any(h.value == "Top players" for h in at.subheader)
    captions = " ".join(c.value for c in at.caption)
    assert "American Soccer Analysis" in captions and "2026" in captions
    assert "contract values aren't available" not in captions  # MLS has salaries


# --- profile extras --------------------------------------------------------------------------


def test_per_90_needs_at_least_a_game_of_minutes():
    assert top_players.per_90(5, 450) == pytest.approx(1.0)
    assert top_players.per_90(5, 60) is None and top_players.per_90(5, None) is None


def test_find_top_player_ignores_accents_and_prefers_the_same_club():
    players = [TopPlayer("Julian Alvarez", "Other FC", 1, 0), TopPlayer("Julián Álvarez", "Atletico", 5, 2)]
    assert top_players.find_top_player(players, "Julian Alvarez", "Atletico").goals == 5
    assert top_players.find_top_player(players, "Julián Álvarez").goals == 1  # no club given: first match
    assert top_players.find_top_player(players, "Nobody") is None


def test_league_rankings_share_ranks_on_ties_and_skip_metrics_the_source_lacks():
    players = [_p("A", 5, 1), _p("B", 5, 3), _p("C", 2, 9, minutes=900, xg=2.0, xa=3.0, salary=1.0)]
    standings = {label: (rank, total) for label, rank, total in top_players.league_rankings(players, players[1])}
    assert standings == {"Goals": (1, 3), "Assists": (2, 3), "Goals + assists": (2, 3)}
    # C is the only player with a per-90 rate and a salary, so those are 1 of 1.
    standings_c = {label: (rank, total) for label, rank, total in top_players.league_rankings(players, players[2])}
    assert standings_c["xG + xA per 90"] == (1, 1) and standings_c["Salary (guaranteed)"] == (1, 1)
    assert standings_c["Goals"] == (3, 3)
