"""ingest/lineups.py: reading the XI ESPN names, and the usual XI from season stats."""

from __future__ import annotations

import pytest

from soccer_predictor.config import League
from soccer_predictor.ingest import asa_client, espn_client, injuries, lineups
from soccer_predictor.ingest.asa_client import AsaPlayerStats
from soccer_predictor.ingest.espn_client import LineupPlayer, MatchDetail, TeamMatchInfo
from soccer_predictor.ingest.understat_client import UnderstatPlayerStats

EPL = League(code="EPL", name="Premier League", seasons=["2627"], espn_league_slug="eng.1")
MLS = League(code="MLS", name="MLS", seasons=["2026"], season_display="single_year", espn_league_slug="usa.1")
UCL = League(code="UCL", name="Champions League", seasons=["2627"], espn_league_slug="uefa.champions")
WC = League(code="WC", name="World Cup", seasons=["2026"], season_display="single_year")


def _side(name, home_away, starters=11, formation="4-3-3"):
    players = [LineupPlayer(f"{name} Starter {i}", str(i), "M", True, formation_place=i) for i in range(1, starters + 1)]
    players += [LineupPlayer(f"{name} Sub", "99", "SUB", False, formation_place=0)]
    players.reverse()  # ESPN's order is not the formation's
    return TeamMatchInfo(espn_id=name, name=name, home_away=home_away, formation=formation, players=players)


def _detail(home="Arsenal", away="Chelsea", home_starters=11, away_starters=11):
    return MatchDetail("1", "pre", "", _side(home, "home", home_starters), _side(away, "away", away_starters), None, None, None, None, None)


# --- the named XI ----------------------------------------------------------------------------------


def test_a_full_sheet_gives_both_xis_in_formation_order_without_the_bench():
    xi = lineups.confirmed_xi(_detail(), "Arsenal", "Chelsea")
    assert xi.home[0] == "Arsenal Starter 1" and xi.home[-1] == "Arsenal Starter 11" and len(xi.home) == 11
    assert "Arsenal Sub" not in xi.home
    assert xi.away[0] == "Chelsea Starter 1" and xi.home_formation == "4-3-3"


@pytest.mark.parametrize("home_starters, away_starters", [(10, 11), (11, 0), (0, 0), (12, 11)])
def test_anything_but_eleven_starters_on_both_sides_is_not_a_lineup(home_starters, away_starters):
    assert lineups.confirmed_xi(_detail(home_starters=home_starters, away_starters=away_starters), "Arsenal", "Chelsea") is None


def test_no_match_page_is_no_lineup():
    assert lineups.confirmed_xi(None, "Arsenal", "Chelsea") is None


def test_sides_are_matched_to_our_teams_by_name_when_espn_lists_them_the_other_way_round():
    xi = lineups.confirmed_xi(_detail(home="Chelsea", away="Arsenal"), "Arsenal", "Chelsea")
    assert xi.home[0].startswith("Arsenal") and xi.away[0].startswith("Chelsea")


def test_the_lineup_is_always_fetched_fresh(monkeypatch):
    seen = {}

    def fake(slug, event_id, state="post", cache_ttl_seconds=None):
        seen.update(slug=slug, event=event_id, ttl=cache_ttl_seconds)
        return _detail()

    monkeypatch.setattr(espn_client, "fetch_match_detail", fake)
    assert lineups.fetch_confirmed_xi(EPL, "42", "Arsenal", "Chelsea") is not None
    assert seen == {"slug": "eng.1", "event": "42", "ttl": 0}
    assert lineups.fetch_confirmed_xi(WC, "42", "A", "B") is None  # no ESPN slug


def test_the_stored_event_id_wins_over_a_name_search(monkeypatch):
    monkeypatch.setattr(espn_client, "find_event_by_names", lambda *a, **k: pytest.fail("should not search"))
    import datetime as dt

    assert lineups.find_event(EPL, dt.date(2026, 10, 31), "Arsenal", "Chelsea", stored_event_id="77") == "77"


def test_without_a_stored_id_the_event_is_found_by_team_names(monkeypatch):
    import datetime as dt

    monkeypatch.setattr(espn_client, "find_event_by_names", lambda slug, d, h, a, cache_ttl_seconds: ("88", "pre"))
    assert lineups.find_event(EPL, dt.date(2026, 10, 31), "Arsenal", "Chelsea") == "88"
    monkeypatch.setattr(espn_client, "find_event_by_names", lambda *a, **k: None)
    assert lineups.find_event(EPL, dt.date(2026, 10, 31), "Arsenal", "Chelsea") is None


# --- the usual XI ----------------------------------------------------------------------------------


def _understat(name, position, minutes, xg=1.0, xa=0.5):
    return UnderstatPlayerStats(name, "Arsenal", position, minutes, 1, 1, xg, xa)


def test_positions_are_bucketed_ignoring_the_substitute_marker():
    understat = lineups._UNDERSTAT_DEFENSE
    assert lineups._bucket("GK S", understat) == ("defense", True)
    assert lineups._bucket("D M S", understat) == ("defense", False)
    assert lineups._bucket("M S", understat) == ("attack", False)
    assert lineups._bucket("S", understat) == ("attack", False)
    assert lineups._bucket(None, understat) == ("attack", False)
    assert lineups._bucket("FB", lineups._ASA_DEFENSE) == ("defense", False)
    assert lineups._bucket("ST", lineups._ASA_DEFENSE) == ("attack", False)


def test_an_understat_squad_is_priced_like_an_injury_would_be(monkeypatch):
    players = [
        _understat("Keeper", "GK", 900, xg=0, xa=0),
        _understat("Defender", "D", 450, xg=0.2, xa=0.1),
        _understat("Striker", "F S", 900, xg=6.0, xa=1.0),
        _understat("Winger", "M", 800, xg=2.0, xa=3.0),
    ]
    monkeypatch.setattr(injuries, "understat_squad", lambda *a: players)
    squad = {p.name: p for p in lineups.expected_squad("Arsenal", EPL)}
    assert squad["Keeper"].is_keeper and squad["Keeper"].position == "defense" and squad["Keeper"].weight == pytest.approx(0.95)
    assert squad["Defender"].weight == pytest.approx(0.5) and not squad["Defender"].is_keeper  # half the keeper's minutes
    assert squad["Striker"].position == "attack" and squad["Striker"].weight > squad["Winger"].weight > 0.05
    assert squad["Striker"].minutes == 900


def test_each_player_carries_where_he_plays_and_what_he_has_done_for_the_team_page(monkeypatch):
    players = [
        UnderstatPlayerStats("Keeper", "Arsenal", "GK S", 900, 0, 0, 0.0, 0.0),
        UnderstatPlayerStats("Defender", "Arsenal", "D M S", 800, 1, 2, 0.2, 0.1),
        UnderstatPlayerStats("Midfielder", "Arsenal", "M", 700, 3, 4, 2.0, 3.0),
        UnderstatPlayerStats("Striker", "Arsenal", "F S", 600, 9, 1, 8.0, 1.0),
        UnderstatPlayerStats("Bench", "Arsenal", "S", 50, 0, 0, 0.0, 0.0),
        UnderstatPlayerStats("Unlisted", "Arsenal", None, 40, 0, 0, 0.0, 0.0),
    ]
    monkeypatch.setattr(injuries, "understat_squad", lambda *a: players)
    squad = {p.name: p for p in lineups.expected_squad("Arsenal", EPL)}
    assert {name: p.role for name, p in squad.items()} == {
        "Keeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Striker": "FWD", "Bench": "", "Unlisted": "",
    }
    assert (squad["Striker"].goals, squad["Striker"].assists) == (9, 1)
    assert (squad["Striker"].xg, squad["Striker"].xa) == (8.0, 1.0)  # what the chances-created rate is added up from


def test_mls_positions_are_placed_on_the_pitch_too(monkeypatch):
    def asa(name, position, goals=0, assists=0):
        return AsaPlayerStats(name, position, 900, goals, assists, 1.0, 1.0, 0, 0, 0.0)

    positions = {"Keeper": "GK", "Centre": "CB", "Full": "FB", "Holding": "DM", "Central": "CM", "Playmaker": "AM", "Wing": "W", "Striker": "ST"}
    monkeypatch.setattr(asa_client, "fetch_team_season", lambda name, season: [asa(n, p, 4, 3) for n, p in positions.items()])
    squad = {p.name: p for p in lineups.expected_squad("LAFC", MLS)}
    assert {name: p.role for name, p in squad.items()} == {
        "Keeper": "GK", "Centre": "DEF", "Full": "DEF", "Holding": "MID", "Central": "MID", "Playmaker": "MID", "Wing": "FWD", "Striker": "FWD",
    }
    assert (squad["Wing"].goals, squad["Wing"].assists) == (4, 3)


def test_a_champions_league_club_borrows_its_domestic_stats(monkeypatch):
    calls = []
    monkeypatch.setattr(injuries, "understat_squad", lambda *a: calls.append(a) or [_understat("Striker", "F", 900)])
    assert [p.name for p in lineups.expected_squad("Arsenal FC", UCL)] == ["Striker"]
    assert calls[0][2].code == "UCL"


def test_mls_players_come_from_american_soccer_analysis(monkeypatch):
    players = [AsaPlayerStats("Keeper", "GK", 900, 0, 0, 0.0, 0.0, 0, 0, 0.0), AsaPlayerStats("Striker", "ST", 800, 8, 2, 7.0, 1.0, 40, 20, 1.0)]
    monkeypatch.setattr(asa_client, "fetch_team_season", lambda name, season: players)
    squad = {p.name: p for p in lineups.expected_squad("LAFC", MLS)}
    assert squad["Keeper"].is_keeper and squad["Striker"].position == "attack"


def test_a_competition_without_player_stats_has_no_usual_xi():
    assert lineups.expected_squad("Brazil", WC) == []
