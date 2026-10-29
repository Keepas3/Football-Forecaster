"""La Liga and Champions League injuries from Biwenger's fantasy feed."""

from __future__ import annotations

import pytest
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import biwenger_client, injuries, player_importance, understat_client
from soccer_predictor.ingest.player_importance import UnderstatTeamTotals
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, injuries_for_team, replace_injuries

LALIGA = League(code="LALIGA", name="La Liga", seasons=["2627"], csv_code="SP1")
UCL = League(code="UCL", name="UEFA Champions League", seasons=["2627"])


def _player(pid, name, team, status, position=3, info=None, price=1_000_000, played=5):
    return {
        "id": pid, "name": name, "teamID": team, "status": status, "position": position, "statusInfo": info,
        "price": price, "playedHome": played, "playedAway": 0,
    }


def _feed(*players):
    return {
        "teams": {"1": {"id": 1, "name": "Athletic"}, "2": {"id": 2, "name": "Betis"}, "3": {"id": 3, "name": "Barcelona"}},
        "players": {str(n): p for n, p in enumerate(players)},
    }


FEED = _feed(
    _player(1, "Williams", 1, "injured", 4, "Hamstring injury.. Estimated return: Nov 25, 2026", price=8_000_000),
    _player(2, "Sancet", 1, "doubt", 3, "Stroke.", price=4_000_000),
    _player(3, "Isco", 2, "sanctioned", 3, "Direct red card. Suspended through Matchday 4."),
    _player(4, "Remiro", 1, "ok", 1, None, price=2_000_000),
    _player(5, "Cucho", 2, "warned", 4),
    _player(6, "Pedri", 3, "unknown", 3),
    {"name": "Broken", "status": "injured"},  # incomplete row: skipped, not fatal
)


@pytest.mark.parametrize(
    ("status", "info", "expected"),
    [
        ("injured", "Ankle injury.. Estimated return: Nov 25, 2026", "Ankle injury - Expected back 25 Nov 2026"),
        ("injured", "Meniscus injury. Estimated return: Mid-November.", "Meniscus injury - Expected back mid-November"),
        ("injured", "Knee injury. Estimated return: Undetermined.", "Knee injury - Unknown return date"),
        ("injured", "Muscle injury. Estimated return: Indefinite return.", "Muscle injury - Unknown return date"),
        ("injured", "Estimated return: Jan 5, 2027", "Injury - Expected back 5 Jan 2027"),
        ("injured", None, "Injury"),
        ("doubt", "Muscle discomfort.", "Muscle discomfort"),
        ("doubt", "Stroke.", "Knock"),  # a machine translation of "golpe"
        ("doubt", None, ""),
        ("sanctioned", "Direct red card. Suspended through Matchday 4.", "Direct red card - Suspended through Matchday 4"),
        ("sanctioned", "Two yellow cards", "Two yellow cards - Suspended"),
        ("sanctioned", None, "Suspended"),
    ],
)
def test_the_free_form_explanation_becomes_problem_and_return(status, info, expected):
    assert biwenger_client.clean_news(status, info) == expected


def test_parse_keeps_injured_doubtful_and_suspended_players_with_position_and_price_share():
    parsed = {i.player_name: i for i in biwenger_client.parse_injuries(FEED)}
    assert set(parsed) == {"Williams", "Sancet", "Isco"}  # not fit, not merely warned, not unknown
    williams, sancet, isco = parsed["Williams"], parsed["Sancet"], parsed["Isco"]
    assert (williams.team_name, williams.position_bucket, williams.chance_of_playing) == ("Athletic", "attack", 0)
    assert (sancet.status, sancet.chance_of_playing) == ("doubt", biwenger_client.DEFAULT_CHANCE_OF_PLAYING)
    assert (isco.status, isco.chance_of_playing) == ("sanctioned", 0)
    assert williams.price_share == 1.0 and sancet.price_share == 0.5  # against Athletic's dearest, Williams
    assert williams.games_played == 5


def test_goalkeepers_and_defenders_are_defence():
    feed = _feed(_player(1, "Keeper", 1, "injured", 1), _player(2, "Back", 1, "injured", 2), _player(3, "Fwd", 1, "injured", 4))
    buckets = {i.player_name: i.position_bucket for i in biwenger_client.parse_injuries(feed)}
    assert buckets == {"Keeper": "defense", "Back": "defense", "Fwd": "attack"}


LALIGA_NAMES = ["Athletic Bilbao", "Atletico Madrid", "Barcelona", "Real Betis", "Racing Santander", "Celta Vigo"]
UCL_NAMES = [
    "Arsenal FC", "FC Bayern München", "Paris Saint-Germain FC", "FC Internazionale Milano", "AC Milan",
    "Manchester City FC", "Manchester United FC", "Club Atlético de Madrid", "Sporting Clube de Portugal",
    "Sporting Clube de Braga", "SK Slavia Praha", "Racing Club de Lens", "PAE AEK",
]


@pytest.mark.parametrize(
    ("theirs", "ours", "expected"),
    [
        ("Athletic", LALIGA_NAMES, "Athletic Bilbao"),
        ("Atlético", LALIGA_NAMES, "Atletico Madrid"),
        ("Betis", LALIGA_NAMES, "Real Betis"),
        ("Racing", LALIGA_NAMES, "Racing Santander"),
        ("Barcelona", LALIGA_NAMES, "Barcelona"),
        ("Bayern Munich", UCL_NAMES, "FC Bayern München"),
        ("PSG", UCL_NAMES, "Paris Saint-Germain FC"),
        ("Inter", UCL_NAMES, "FC Internazionale Milano"),
        ("Atlético de Madrid", UCL_NAMES, "Club Atlético de Madrid"),
        ("Sporting Portugal", UCL_NAMES, "Sporting Clube de Portugal"),  # not Braga
        ("Slavia Prag", UCL_NAMES, "SK Slavia Praha"),
        ("Lens", UCL_NAMES, "Racing Club de Lens"),
        ("AEK Athen", UCL_NAMES, "PAE AEK"),
        ("Manchester City", UCL_NAMES, "Manchester City FC"),
        ("Manchester United", UCL_NAMES, "Manchester United FC"),
    ],
)
def test_biwenger_club_names_map_to_ours_without_mixing_up_lookalikes(theirs, ours, expected):
    assert biwenger_client.resolve_team_name(theirs, ours) == expected


def test_an_unknown_or_ambiguous_club_is_left_unmatched():
    assert biwenger_client.resolve_team_name("Atlantis", UCL_NAMES) is None
    assert biwenger_client.resolve_team_name("Manchester", UCL_NAMES) is None  # City or United: no guess


def test_the_feed_is_cached_per_competition_and_a_failure_is_unknown_not_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(biwenger_client, "CACHE_DIR", tmp_path)
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"status": 200, "data": FEED}

    def get(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(biwenger_client.http_session, "get", get)
    assert [i.player_name for i in biwenger_client.fetch_injuries_by_club("LALIGA")["Athletic"]] == ["Williams", "Sancet"]
    assert biwenger_client.fetch_injuries_by_club("LALIGA")["Barcelona"] == []  # a club with none is listed with none
    assert len(calls) == 1 and "la-liga" in calls[0]  # the second read came from the cache
    biwenger_client.fetch_injuries_by_club("UCL")
    assert len(calls) == 2 and "champions-league" in calls[1]

    assert biwenger_client.fetch_injuries_by_club("EPL") is None  # not a Biwenger competition

    for cached in tmp_path.iterdir():
        cached.unlink()

    def down(*args, **kwargs):
        raise requests.RequestException("down")

    monkeypatch.setattr(biwenger_client.http_session, "get", down)
    assert biwenger_client.fetch_injuries_by_club("LALIGA") is None


def test_a_payload_without_players_is_not_cached_as_a_healthy_league(monkeypatch, tmp_path):
    monkeypatch.setattr(biwenger_client, "CACHE_DIR", tmp_path)

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"players": {}, "teams": {}}}

    monkeypatch.setattr(biwenger_client.http_session, "get", lambda *a, **k: Response())
    assert biwenger_client.fetch_injuries_by_club("LALIGA") is None
    assert list(tmp_path.iterdir()) == []


# --- syncing into the database ---------------------------------------------------------------


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def _stub_pricing(monkeypatch, weight=0.6, matched=True):
    monkeypatch.setattr(understat_client, "fetch_team_season", lambda *a, **k: [object()] if matched else [])
    monkeypatch.setattr(understat_client, "fetch_all_teams_totals", lambda *a, **k: {})
    monkeypatch.setattr(player_importance, "aggregate_understat_team_output", lambda players: UnderstatTeamTotals(0.0, 0))
    monkeypatch.setattr(player_importance, "resolve_api_importance_weight", lambda *a, **k: weight)
    monkeypatch.setattr(
        player_importance, "find_understat_player", lambda name, players: object() if players else None
    )


def test_sync_stores_priced_injuries_with_notes_scaling_doubts_and_clearing_healthy_clubs(monkeypatch):
    _stub_pricing(monkeypatch, weight=0.6)
    monkeypatch.setattr(biwenger_client, "fetch_competition", lambda slug: FEED)
    with _session() as s:
        athletic = get_or_create_team(s, "Athletic Bilbao", "LALIGA")
        betis = get_or_create_team(s, "Real Betis", "LALIGA")
        barca = get_or_create_team(s, "Barcelona", "LALIGA")
        replace_injuries(s, barca.id, "api", [{"player_name": "Stale", "position": "attack", "importance_weight": 0.5}])
        replace_injuries(s, athletic.id, "manual", [{"player_name": "Keep Me", "position": "attack", "importance_weight": 0.5}])
        s.commit()

        synced, skipped = injuries.sync_injuries_to_db(s, LALIGA, 2026)
        s.commit()

        assert (synced, skipped) == (3, 0)
        by_name = {i.player_name: i for i in injuries_for_team(s, athletic.id) if i.source == "api"}
        assert by_name["Williams"].importance_weight == 0.6  # out: full weight
        assert by_name["Williams"].note == "Biwenger: Hamstring injury - Expected back 25 Nov 2026"
        assert abs(by_name["Sancet"].importance_weight - 0.3) < 1e-9  # a coin flip: half
        assert by_name["Sancet"].note == "Biwenger: Knock - 50% chance of playing"
        assert [i.note for i in injuries_for_team(s, betis.id)] == ["Biwenger: Direct red card - Suspended through Matchday 4"]
        assert [i for i in injuries_for_team(s, barca.id) if i.source == "api"] == []  # no longer injured
        assert any(i.source == "manual" and i.player_name == "Keep Me" for i in injuries_for_team(s, athletic.id))


def test_an_unreachable_feed_leaves_the_stored_injuries_alone(monkeypatch):
    monkeypatch.setattr(biwenger_client, "fetch_competition", lambda slug: None)
    with _session() as s:
        athletic = get_or_create_team(s, "Athletic Bilbao", "LALIGA")
        replace_injuries(s, athletic.id, "api", [{"player_name": "Yesterday", "position": "attack", "importance_weight": 0.5}])
        s.commit()

        assert injuries.sync_injuries_to_db(s, LALIGA, 2026) == (0, 0)
        assert [i.player_name for i in injuries_for_team(s, athletic.id)] == ["Yesterday"]


def test_a_club_that_matches_none_of_ours_is_skipped_and_two_cannot_claim_one(monkeypatch):
    _stub_pricing(monkeypatch)
    feed = {
        "teams": {"1": {"id": 1, "name": "Betis"}, "2": {"id": 2, "name": "Real Betis"}, "3": {"id": 3, "name": "Atlantis"}},
        "players": {
            "1": _player(1, "A", 1, "injured"),
            "2": _player(2, "B", 2, "injured"),
            "3": _player(3, "C", 3, "injured"),
        },
    }
    monkeypatch.setattr(biwenger_client, "fetch_competition", lambda slug: feed)
    with _session() as s:
        get_or_create_team(s, "Real Betis", "LALIGA")
        s.commit()
        assert injuries.sync_injuries_to_db(s, LALIGA, 2026) == (1, 2)


def test_doubts_too_unlikely_to_matter_are_dropped(monkeypatch):
    _stub_pricing(monkeypatch, weight=0.03)  # 0.03 * 0.5 = 0.015, under the floor
    monkeypatch.setattr(biwenger_client, "fetch_competition", lambda slug: _feed(_player(1, "X", 1, "doubt")))
    with _session() as s:
        athletic = get_or_create_team(s, "Athletic Bilbao", "LALIGA")
        s.commit()
        injuries.sync_injuries_to_db(s, LALIGA, 2026)
        assert injuries_for_team(s, athletic.id) == []


def test_a_player_understat_lacks_is_priced_from_the_feed_itself(monkeypatch):
    _stub_pricing(monkeypatch, weight=0.9, matched=False)  # Understat has nobody for the club
    feed = _feed(
        _player(1, "Star", 1, "injured", price=10_000_000),
        _player(2, "Squad", 1, "injured", price=2_000_000),
        _player(3, "Bench", 1, "injured", price=1_000_000, played=0),
    )
    monkeypatch.setattr(biwenger_client, "fetch_competition", lambda slug: feed)
    with _session() as s:
        athletic = get_or_create_team(s, "Athletic Bilbao", "LALIGA")
        s.commit()
        injuries.sync_injuries_to_db(s, LALIGA, 2026)
        weights = {i.player_name: i.importance_weight for i in injuries_for_team(s, athletic.id)}
        assert weights["Star"] == player_importance.DEFAULT_API_IMPORTANCE_WEIGHT  # the club's dearest player
        assert weights["Squad"] == injuries.UNPLAYED_WEIGHT  # 0.5 * 0.2 = 0.1: the floor
        assert weights["Bench"] == injuries.UNPLAYED_WEIGHT  # never played


def test_champions_league_clubs_are_priced_from_the_domestic_league_understat_tracks(monkeypatch):
    seen = []
    squads = {"Arsenal": [object()], "Chelsea": [object()]}
    monkeypatch.setattr(understat_client, "fetch_all_teams_totals", lambda code, season: squads if code == "EPL" else {})
    monkeypatch.setattr(understat_client, "fetch_team_season", lambda *a, **k: seen.append("per-league") or [])

    arsenal = injuries._understat_squad("Arsenal", "Arsenal FC", UCL, 2026)
    assert arsenal is squads["Arsenal"] and seen == []  # not through the La Liga-style per-league lookup
    assert injuries._understat_squad("Slavia Prag", "SK Slavia Praha", UCL, 2026) == []  # Understat has no such club
    assert injuries._understat_squad("Arsenal", "Arsenal FC", LALIGA, 2026) == [] and seen == ["per-league"]


def test_only_the_single_request_feeds_run_in_the_quick_refresh():
    assert injuries.SINGLE_REQUEST_LEAGUES == {"EPL", "LALIGA", "UCL"}
    assert injuries.has_automatic_feed(UCL) and injuries.has_automatic_feed(LALIGA)
    assert not injuries.has_automatic_feed(League(code="SERIEA", name="Serie A", seasons=["2627"], csv_code="I1"))
