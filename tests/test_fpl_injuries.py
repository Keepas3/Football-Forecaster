"""Premier League injuries from the Fantasy Premier League feed."""

from __future__ import annotations

import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import fpl_client, injuries, player_importance, understat_client
from soccer_predictor.ingest.player_importance import UnderstatTeamTotals
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, injuries_for_team, replace_injuries

EPL = League(code="EPL", name="English Premier League", seasons=["2627"], csv_code="E0")


def _element(web, first, second, team, status, element_type=3, chance=None, news=""):
    return {
        "web_name": web, "first_name": first, "second_name": second, "team": team, "status": status,
        "element_type": element_type, "chance_of_playing_next_round": chance, "news": news,
    }


FEED = {
    "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Spurs"}, {"id": 3, "name": "Man Utd"}],
    "elements": [
        _element("Saliba", "William", "Saliba", 1, "i", 2, 0, "Back injury - Unknown return date"),
        _element("Saka", "Bukayo", "Saka", 1, "d", 3, 75, "Knock - 75% chance of playing"),
        _element("Romero", "Cristian", "Romero", 2, "s", 2, 0, "Suspended for one match"),
        _element("Onana", "Andre", "Onana", 3, "u", 1, 0, "Has joined another club"),
        _element("Healthy", "Ben", "Fit", 3, "a", 4, 100, ""),
        {"web_name": "Broken", "status": "i"},  # incomplete row: skipped, not fatal
    ],
}


def test_parse_keeps_injured_doubtful_and_suspended_players_with_their_position_bucket():
    parsed = {i.player_name: i for i in fpl_client.parse_injuries(FEED)}
    assert set(parsed) == {"William Saliba", "Bukayo Saka", "Cristian Romero"}  # not "u", not available
    assert parsed["William Saliba"].position_bucket == "defense" and parsed["Bukayo Saka"].position_bucket == "attack"
    assert (parsed["Bukayo Saka"].status, parsed["Bukayo Saka"].chance_of_playing) == ("d", 75)
    assert parsed["Cristian Romero"].chance_of_playing == 0 and parsed["Cristian Romero"].team_name == "Spurs"


def test_a_doubtful_player_with_no_stated_chance_is_a_coin_flip():
    feed = {"teams": [{"id": 1, "name": "Arsenal"}], "elements": [_element("X", "A", "B", 1, "d", 3, None)]}
    assert fpl_client.parse_injuries(feed)[0].chance_of_playing == fpl_client.DEFAULT_CHANCE_OF_PLAYING


def test_fpl_club_names_map_to_ours_without_mixing_up_lookalikes():
    ours = ["Arsenal", "Man City", "Man United", "Tottenham", "Nott'm Forest", "Wolves"]
    assert fpl_client.resolve_team_name("Spurs", ours) == "Tottenham"
    assert fpl_client.resolve_team_name("Man Utd", ours) == "Man United"
    assert fpl_client.resolve_team_name("Man City", ours) == "Man City"
    assert fpl_client.resolve_team_name("Nott'm Forest", ours) == "Nott'm Forest"
    assert fpl_client.resolve_team_name("Atlantis FC", ours) is None


def test_the_feed_is_cached_and_a_failure_is_unknown_not_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(fpl_client, "CACHE_FILE", tmp_path / "fpl.json")
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return FEED

    monkeypatch.setattr(fpl_client.http_session, "get", lambda *a, **k: calls.append(1) or Response())
    assert fpl_client.fetch_injuries_by_club()["Arsenal"][0].player_name == "William Saliba"
    assert fpl_client.fetch_injuries_by_club()["Man Utd"] == []  # a club with none is listed with none
    assert len(calls) == 1  # the second read came from the cache

    (tmp_path / "fpl.json").unlink()

    def down(*a, **k):
        raise requests.RequestException("down")

    monkeypatch.setattr(fpl_client.http_session, "get", down)
    assert fpl_client.fetch_injuries_by_club() is None


# --- syncing into the database ---------------------------------------------------------------


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def _stub_pricing(monkeypatch, weight=0.6):
    monkeypatch.setattr(understat_client, "fetch_team_season", lambda *a, **k: [])
    monkeypatch.setattr(player_importance, "aggregate_understat_team_output", lambda players: UnderstatTeamTotals(0.0, 0))
    monkeypatch.setattr(player_importance, "resolve_api_importance_weight", lambda *a, **k: weight)
    monkeypatch.setattr(player_importance, "find_understat_player", lambda name, players: object())  # matched


def test_sync_stores_priced_injuries_per_club_scaling_doubts_and_clearing_healthy_clubs(monkeypatch):
    _stub_pricing(monkeypatch, weight=0.6)
    monkeypatch.setattr(fpl_client, "fetch_bootstrap", lambda: FEED)
    with _session() as s:
        arsenal, spurs = get_or_create_team(s, "Arsenal", "EPL"), get_or_create_team(s, "Tottenham", "EPL")
        united = get_or_create_team(s, "Man United", "EPL")
        replace_injuries(s, united.id, "api", [{"player_name": "Stale", "position": "attack", "importance_weight": 0.5}])
        replace_injuries(s, arsenal.id, "manual", [{"player_name": "Keep Me", "position": "attack", "importance_weight": 0.5}])
        s.commit()

        synced, skipped = injuries.sync_injuries_to_db(s, EPL, 2026)
        s.commit()

        assert (synced, skipped) == (3, 0)
        by_name = {i.player_name: i for i in injuries_for_team(s, arsenal.id) if i.source == "api"}
        assert by_name["William Saliba"].importance_weight == 0.6  # out: full weight
        assert abs(by_name["Bukayo Saka"].importance_weight - 0.15) < 1e-9  # 75% likely to play: a quarter
        assert by_name["Bukayo Saka"].note == "FPL: Knock - 75% chance of playing"  # not repeated
        assert [i.player_name for i in injuries_for_team(s, spurs.id)] == ["Cristian Romero"]
        assert [i for i in injuries_for_team(s, united.id) if i.source == "api"] == []  # no longer injured
        assert any(i.source == "manual" and i.player_name == "Keep Me" for i in injuries_for_team(s, arsenal.id))


def test_an_unreachable_feed_leaves_the_stored_injuries_alone(monkeypatch):
    monkeypatch.setattr(fpl_client, "fetch_bootstrap", lambda: None)
    with _session() as s:
        arsenal = get_or_create_team(s, "Arsenal", "EPL")
        replace_injuries(s, arsenal.id, "api", [{"player_name": "Yesterday", "position": "attack", "importance_weight": 0.5}])
        s.commit()

        assert injuries.sync_injuries_to_db(s, EPL, 2026) == (0, 0)
        assert [i.player_name for i in injuries_for_team(s, arsenal.id)] == ["Yesterday"]


def test_doubts_too_unlikely_to_matter_are_dropped(monkeypatch):
    _stub_pricing(monkeypatch, weight=0.05)
    feed = {"teams": [{"id": 1, "name": "Arsenal"}], "elements": [_element("X", "A", "B", 1, "d", 3, 75)]}  # 0.05 * 0.25
    monkeypatch.setattr(fpl_client, "fetch_bootstrap", lambda: feed)
    with _session() as s:
        arsenal = get_or_create_team(s, "Arsenal", "EPL")
        s.commit()
        injuries.sync_injuries_to_db(s, EPL, 2026)
        assert injuries_for_team(s, arsenal.id) == []


def test_an_unmatched_reserve_gets_a_small_weight_but_an_unmatched_regular_keeps_the_default(monkeypatch):
    _stub_pricing(monkeypatch, weight=0.5)
    monkeypatch.setattr(player_importance, "find_understat_player", lambda name, players: None)  # Understat has neither
    feed = {
        "teams": [{"id": 1, "name": "Arsenal"}],
        "elements": [
            {**_element("Reserve", "Res", "Erve", 1, "i", 1, 0, "Knee"), "minutes": 0},
            {**_element("Regular", "Reg", "Ular", 1, "i", 3, 0, "Ankle"), "minutes": 1800},
        ],
    }
    monkeypatch.setattr(fpl_client, "fetch_bootstrap", lambda: feed)
    with _session() as s:
        arsenal = get_or_create_team(s, "Arsenal", "EPL")
        s.commit()
        injuries.sync_injuries_to_db(s, EPL, 2026)
        weights = {i.player_name: i.importance_weight for i in injuries_for_team(s, arsenal.id)}
        assert weights == {"Res Erve": injuries.UNPLAYED_WEIGHT, "Reg Ular": 0.5}
