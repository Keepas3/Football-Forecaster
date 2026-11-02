"""The Defenders and Goalkeepers rankings (ingest/defensive_players.py): the Fantasy Premier League
parser behind the Premier League's, and how players are ranked, formatted and found."""

from __future__ import annotations

import pytest

from soccer_predictor.config import League
from soccer_predictor.ingest import asa_client, fpl_client, memo
from soccer_predictor.ingest import defensive_players as dp
from soccer_predictor.ingest.asa_client import AsaDefender, AsaKeeper
from soccer_predictor.ingest.defensive_players import DefensivePlayer as P

EPL = League(code="EPL", name="English Premier League", seasons=["2526", "2627"], csv_code="E0")
MLS = League(code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1", season_display="single_year")
LALIGA = League(code="LALIGA", name="La Liga", seasons=["2627"], csv_code="SP1")


@pytest.fixture(autouse=True)
def _fresh_memos():
    """The fetchers keep their result for a few minutes; a test must never see another's."""
    memo.clear_all()
    yield
    memo.clear_all()


def _element(first, second, team, element_type, **stats):
    return {
        "first_name": first, "second_name": second, "web_name": second, "team": team, "element_type": element_type, **stats,
    }


FEED = {
    "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Man Utd"}],
    "elements": [
        _element("William", "Saliba", 1, 2, minutes=810, tackles=15, clearances_blocks_interceptions=50, clean_sheets=5,
                 goals_conceded=4, expected_goals_conceded="6.3", goals_scored=1, assists=0),
        _element("David", "Raya", 1, 1, minutes=810, saves=18, goals_conceded=5, expected_goals_conceded="7.2",
                 clean_sheets=6, penalties_saved=1, tackles=0, clearances_blocks_interceptions=2),
        _element("Bukayo", "Saka", 1, 3, minutes=800, goals_scored=5),  # a midfielder: not this feed's business
        _element("Lisandro", "Martinez", 2, 2, minutes=None, tackles=None, expected_goals_conceded=None),  # nothing played yet
        {"first_name": "No", "second_name": "Team", "element_type": 2, "team": 99},  # an unknown club: skipped, not fatal
        {"web_name": "Broken"},  # no role: skipped
    ],
}


# --- the Fantasy Premier League feed ---------------------------------------------------------


def test_the_feed_gives_every_defender_and_goalkeeper_with_tackles_and_the_rest_counted_as_defensive_actions():
    parsed = {p.name: p for p in fpl_client.parse_players(FEED)}
    assert set(parsed) == {"William Saliba", "David Raya", "Lisandro Martinez"}
    saliba, raya = parsed["William Saliba"], parsed["David Raya"]
    assert (saliba.role, saliba.team_name, saliba.minutes) == ("defender", "Arsenal", 810)
    assert saliba.defensive_actions == 65 and saliba.expected_goals_conceded == pytest.approx(6.3)  # "6.3" is text in the feed
    assert (saliba.clean_sheets, saliba.goals_conceded, saliba.goals, saliba.assists) == (5, 4, 1, 0)
    assert (raya.role, raya.saves, raya.penalties_saved) == ("goalkeeper", 18, 1)
    martinez = parsed["Lisandro Martinez"]
    assert (martinez.minutes, martinez.defensive_actions, martinez.expected_goals_conceded) == (0, 0, 0.0)  # null becomes 0


def test_an_unavailable_feed_gives_no_players(monkeypatch):
    monkeypatch.setattr(fpl_client, "fetch_bootstrap", lambda: None)
    assert fpl_client.fetch_players() == []


# --- ranking ---------------------------------------------------------------------------------


def _d(name, minutes, **values):
    return P(name, "Club", dp.DEFENDER, minutes, values)


def test_players_are_ranked_best_first_and_fewest_goals_conceded_first():
    players = [
        _d("Leaky", 900, **{dp.CONCEDED_P90: 1.8, dp.CLEAN_SHEETS: 1}),
        _d("Solid", 900, **{dp.CONCEDED_P90: 0.6, dp.CLEAN_SHEETS: 5}),
        _d("Mid", 900, **{dp.CONCEDED_P90: 1.0, dp.CLEAN_SHEETS: 5}),
    ]
    assert [p.name for p in dp.rank_players(players, dp.CONCEDED_P90)] == ["Solid", "Mid", "Leaky"]
    assert [p.name for p in dp.rank_players(players, dp.CLEAN_SHEETS)] == ["Mid", "Solid", "Leaky"]  # a tie: alphabetical
    assert [p.name for p in dp.rank_players(players, dp.CLEAN_SHEETS, limit=1)] == ["Mid"]


def test_a_tie_goes_to_the_player_with_more_minutes():
    players = [_d("Few", 500, **{dp.CLEAN_SHEETS: 3}), _d("Many", 900, **{dp.CLEAN_SHEETS: 3})]
    assert [p.name for p in dp.rank_players(players, dp.CLEAN_SHEETS)] == ["Many", "Few"]


def test_a_rate_needs_a_real_sample_but_a_total_does_not():
    cameo = _d("Cameo", 120, **{dp.CONCEDED_P90: 0.0, dp.CLEAN_SHEETS: 1, dp.ACTIONS_P90: 20.0})
    regular = _d("Regular", 900, **{dp.CONCEDED_P90: 1.1, dp.CLEAN_SHEETS: 0, dp.ACTIONS_P90: 8.0})
    players = [cameo, regular]
    assert [p.name for p in dp.rank_players(players, dp.CONCEDED_P90)] == ["Regular"]  # 0.0 over 120 minutes is noise
    assert [p.name for p in dp.rank_players(players, dp.ACTIONS_P90)] == ["Regular"]
    assert [p.name for p in dp.rank_players(players, dp.CLEAN_SHEETS)] == ["Cameo", "Regular"]


def test_a_player_with_no_value_for_the_metric_is_left_out():
    players = [_d("Has", 900, **{dp.CLEAN_SHEETS: 2}), _d("Hasnt", 900, **{dp.CLEAN_SHEETS: None})]
    assert [p.name for p in dp.rank_players(players, dp.CLEAN_SHEETS)] == ["Has"]


def test_the_metrics_offered_are_the_ones_the_source_has_in_the_roles_usual_order():
    players = [_d("A", 900, **{dp.CLEAN_SHEETS: 1, dp.ACTIONS_P90: 9.0, dp.GOALS_PLUS_ASSISTS: 0})]
    assert dp.available_metrics(dp.DEFENDER, players) == [dp.ACTIONS_P90, dp.CLEAN_SHEETS, dp.GOALS_PLUS_ASSISTS]
    assert dp.available_metrics(dp.DEFENDER, []) == []
    assert dp.available_metrics(dp.GOALKEEPER, players) == [dp.CLEAN_SHEETS]  # shared metric, keeper order


# --- showing numbers -------------------------------------------------------------------------


def test_numbers_are_shown_with_the_decimals_sign_and_unit_of_their_metric():
    assert dp.format_value(dp.PREVENTED, 2.24) == "+2.2" and dp.format_value(dp.PREVENTED, -1.26) == "-1.3"
    assert dp.format_value(dp.SAVE_PCT, 78.26) == "78.3%"
    assert dp.format_value(dp.CLEAN_SHEETS, 6) == "6" and dp.format_value(dp.ACTIONS_P90, 12.18) == "12.2"
    assert dp.format_value(dp.TOTAL_P90, 0.114) == "+0.11" and dp.format_value(dp.SAVES, None) == "–"
    assert dp.format_value(dp.PREVENTED, -0.04) == "+0.0" and dp.format_value(dp.PREVENTED, 0.0) == "+0.0"  # never "-0.0"


def test_the_table_has_the_ranked_metric_first_then_the_others_and_the_minutes():
    players = [
        _d("Solid", 900, **{dp.ACTIONS_P90: 9.04, dp.CLEAN_SHEETS: 5, dp.CONCEDED_P90: 0.6}),
        _d("Busy", 900, **{dp.ACTIONS_P90: 12.1, dp.CLEAN_SHEETS: 1, dp.CONCEDED_P90: 1.5}),
    ]
    rows = dp.table_rows(players, dp.DEFENDER, dp.CLEAN_SHEETS)
    assert list(rows[0]) == ["#", "Player", "Club", "Clean sheets", "Defensive actions per 90", "Goals conceded per 90", "Minutes"]
    assert [r["Player"] for r in rows] == ["Solid", "Busy"] and rows[0]["#"] == 1
    assert (rows[0]["Clean sheets"], rows[0]["Defensive actions per 90"], rows[0]["Goals conceded per 90"]) == ("5", "9.0", "0.60")
    assert rows[0]["Minutes"] == 900
    assert dp.table_rows(players, dp.DEFENDER, dp.ACTIONS_P90)[0]["Player"] == "Busy"
    assert dp.table_rows([], dp.DEFENDER, dp.CLEAN_SHEETS) == []


def test_the_leagues_standing_shares_ranks_on_ties_and_counts_fewest_conceded_as_best():
    a = _d("A", 900, **{dp.CLEAN_SHEETS: 5, dp.CONCEDED_P90: 0.6})
    b = _d("B", 900, **{dp.CLEAN_SHEETS: 5, dp.CONCEDED_P90: 1.0})
    c = _d("C", 900, **{dp.CLEAN_SHEETS: 2, dp.CONCEDED_P90: 1.4})
    cameo = _d("Cameo", 100, **{dp.CLEAN_SHEETS: 1, dp.CONCEDED_P90: 0.0})
    players = [a, b, c, cameo]
    standing = {label: (rank, total) for label, rank, total in dp.league_rankings(players, b)}
    assert standing == {"Clean sheets": (1, 4), "Goals conceded per 90": (2, 3)}  # B is level first on sheets; the cameo has no rate
    standing_c = {label: (rank, total) for label, rank, total in dp.league_rankings(players, c)}
    assert standing_c == {"Clean sheets": (3, 4), "Goals conceded per 90": (3, 3)}
    assert dp.league_rankings(players, cameo) == [("Clean sheets", 4, 4)]


def test_a_player_is_found_by_name_ignoring_accents_and_preferring_his_club():
    players = [P("Caoimhín Kelleher", "Brentford", dp.GOALKEEPER, 810), P("Caoimhin Kelleher", "Elsewhere", dp.GOALKEEPER, 90)]
    assert dp.find_defensive_player(players, "Caoimhin Kelleher", "Elsewhere").team == "Elsewhere"
    assert dp.find_defensive_player(players, "Caoimhín Kelleher", "Man Utd").team == "Brentford"  # club spelt differently: first
    assert dp.find_defensive_player(players, "Nobody") is None


# --- where each league's numbers come from ---------------------------------------------------


def test_the_premier_league_ranks_from_the_fpl_feed_with_each_roles_own_metrics(monkeypatch):
    monkeypatch.setattr(fpl_client, "fetch_players", lambda: fpl_client.parse_players(FEED))
    defenders = dp.fetch_defensive_players(EPL, dp.DEFENDER)
    assert (defenders.source, defenders.season) == ("Fantasy Premier League", 2026)
    assert [p.name for p in defenders.players] == ["William Saliba", "Lisandro Martinez"]
    saliba = defenders.players[0]
    assert saliba.values[dp.ACTIONS_P90] == pytest.approx(65 * 90 / 810)
    assert saliba.values[dp.CONCEDED_P90] == pytest.approx(0.4444, abs=1e-3)
    assert (saliba.values[dp.CLEAN_SHEETS], saliba.values[dp.GOALS_PLUS_ASSISTS]) == (5, 1)
    assert defenders.players[1].values[dp.ACTIONS_P90] is None  # no minutes: no rate, so never ranked
    assert dp.available_metrics(dp.DEFENDER, defenders.players) == [
        dp.ACTIONS_P90, dp.CLEAN_SHEETS, dp.CONCEDED_P90, dp.GOALS_PLUS_ASSISTS,
    ]

    keepers = dp.fetch_defensive_players(EPL, dp.GOALKEEPER)
    raya = keepers.players[0]
    assert [p.name for p in keepers.players] == ["David Raya"]
    assert raya.values[dp.PREVENTED] == pytest.approx(2.2)  # the team's expected goals conceded minus the goals let in
    assert raya.values[dp.SAVE_PCT] == pytest.approx(100 * 18 / 23)  # saves out of saves plus goals conceded
    assert (raya.values[dp.SAVES], raya.values[dp.PENALTIES_SAVED], raya.values[dp.CLEAN_SHEETS]) == (18, 1, 6)
    assert dp.available_metrics(dp.GOALKEEPER, keepers.players)[0] == dp.PREVENTED  # the default


def test_mls_ranks_from_american_soccer_analysis(monkeypatch):
    seen = []
    monkeypatch.setattr(
        asa_client, "fetch_league_defenders",
        lambda season: seen.append(season) or [AsaDefender("Ethan Bartlow", "Sporting Kansas City", "CB", 893, 1.25, 0.1, 1.2)],
    )
    monkeypatch.setattr(
        asa_client, "fetch_league_goalkeepers",
        lambda season: [AsaKeeper("Matt Turner", "New England Revolution", 3251, 178, 131, 47, 16.5), AsaKeeper("Rookie", "X", 90, 0, 0, 0, 0.0)],
    )
    defenders = dp.fetch_defensive_players(MLS, dp.DEFENDER)
    assert (defenders.source, defenders.season, seen) == ("American Soccer Analysis", 2026, [2026])
    assert defenders.players[0].values[dp.INTERRUPTING_P90] == pytest.approx(1.25 * 90 / 893)
    assert dp.available_metrics(dp.DEFENDER, defenders.players) == [dp.INTERRUPTING_P90, dp.TOTAL_P90]  # default: interrupting

    keepers = dp.fetch_defensive_players(MLS, dp.GOALKEEPER)
    turner, rookie = keepers.players
    assert turner.values[dp.PREVENTED] == 16.5 and turner.values[dp.SAVE_PCT] == pytest.approx(100 * 131 / 178)
    assert turner.values[dp.SHOTS_FACED] == 178 and turner.values[dp.CONCEDED_P90] == pytest.approx(47 * 90 / 3251)
    assert rookie.values[dp.SAVE_PCT] is None  # no shots faced: no percentage
    assert dp.available_metrics(dp.GOALKEEPER, keepers.players) == [
        dp.PREVENTED, dp.SAVE_PCT, dp.SAVES, dp.CONCEDED_P90, dp.SHOTS_FACED,
    ]


def test_a_league_with_no_free_source_has_nobody_to_rank():
    for role in dp.ROLES:
        result = dp.fetch_defensive_players(LALIGA, role)
        assert (result.players, result.source, result.season) == ([], "", None)
    assert set(dp.SOURCES) == {"EPL", "MLS"}
