"""MLS, Bundesliga, Serie A and Ligue 1 injuries read from starting11.com's league pages."""

from __future__ import annotations

import pytest
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import asa_client, injuries, player_importance, starting11_client, understat_client
from soccer_predictor.ingest.player_importance import AsaTeamTotals, UnderstatTeamTotals
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, injuries_for_team, replace_injuries

MLS = League(code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1")
BUNDESLIGA = League(code="BUNDESLIGA", name="Bundesliga", seasons=["2627"], csv_code="D1")


def _row(shirt, name, position, problem, status):
    return (
        f'<li><span class="x"></span><span>{shirt}</span><div><div><a href="/player/{name.lower()}">{name}</a>'
        f'<span>{position}</span></div><p>{problem}</p></div><div><span><span><svg><path d="M0"/></svg></span>{status}'
        "</span></div></li>"
    )


def _card(slug, club, *rows):
    return (
        f'<article id="club-{slug}"><a href="/{slug}"><span><span></span><h3>{club}</h3></span><span>x</span></a>'
        f'<ul>{"".join(rows)}</ul><a href="/{slug}/injuries">Full <!-- -->{club}<!-- --> injury list</a></article>'
    )


def _page(*cards, totals="All injuries 5 Out 2 Doubtful 1 Suspended"):
    return (
        "<html><body><section><h1>Injuries</h1><div>" + totals.replace(" Out", "</span> Out") + "</div></section>"
        '<section><h2 id="injuries-by-club">By club</h2><div>' + "".join(cards) + "</div></section></body></html>"
    )


# Five out (one of them "off the roster"), two doubtful, one suspended: the page's own totals.
CARDS = (
    _card(
        "columbus-crew",
        "Columbus Crew",
        _row(4, "Mihaj", "DEF", "Hip Injury", "Out"),
        _row(28, "Reilly", "MID", "Ankle Injury", "Out"),
        _row(1, "Keeper", "GK", "Knee Injury", "Out"),
        _row(9, "Gone", "FWD", "Off the roster", "Out"),
    ),
    _card(
        "austin",
        "Austin",
        _row(38, "Torres", "MID", "Injury", "Doubtful"),
        _row(22, "Bell", "DEF", "Muscle Injury", "Doubtful"),
        _row(33, "Thompson", "DEF", "Red Card", "Suspended"),
        _row(10, "Anyone", "FWD", "Thigh Injury", "Out"),
    ),
)
PAGE = _page(*CARDS)


def test_the_page_is_read_into_clubs_and_players_with_status_position_and_problem():
    parsed = starting11_client.parse_page(PAGE)
    assert list(parsed) == ["Columbus Crew", "Austin"]
    crew = {p.player_name: p for p in parsed["Columbus Crew"]}
    assert set(crew) == {"Mihaj", "Reilly", "Keeper"}  # "Off the roster" is not an injury
    assert (crew["Mihaj"].position_bucket, crew["Keeper"].position_bucket, crew["Reilly"].position_bucket) == (
        "defense", "defense", "attack",
    )
    assert (crew["Mihaj"].status, crew["Mihaj"].problem, crew["Mihaj"].chance_of_playing) == ("Out", "Hip Injury", 0)
    austin = {p.player_name: p for p in parsed["Austin"]}
    assert (austin["Torres"].status, austin["Torres"].chance_of_playing) == ("Doubtful", 75)
    assert (austin["Thompson"].status, austin["Thompson"].problem) == ("Suspended", "Red Card")


def test_a_club_left_with_nobody_after_the_non_injuries_are_dropped_is_still_listed():
    page = _page(
        _card("austin", "Austin", _row(9, "Gone", "FWD", "Loan agreement", "Out")),
        totals="All injuries 1 Out 0 Doubtful 0 Suspended",
    )
    assert starting11_client.parse_page(page) == {"Austin": []}


def test_a_page_whose_rows_disagree_with_its_own_totals_is_rejected_not_half_trusted():
    assert starting11_client.parse_page(_page(*CARDS, totals="All injuries 9 Out 2 Doubtful 1 Suspended")) is None


def test_a_page_with_no_club_cards_or_no_players_is_not_a_page_we_can_read():
    assert starting11_client.parse_page("<html><body><h1>Maintenance</h1></body></html>") is None
    assert starting11_client.parse_page(_page(_card("austin", "Austin"), totals="All injuries 0 Out 0 Doubtful 0 Suspended")) is None


def test_the_order_of_the_texts_within_a_row_does_not_matter():
    shuffled = (
        '<article id="club-austin"><h3>Austin</h3><ul><li><div><p>Hip Injury</p></div><span>Doubtful</span>'
        '<span>DEF</span><a href="/player/bell">Bell</a><span>5</span></li></ul></article>'
    )
    parsed = starting11_client.parse_page(shuffled)
    assert [(p.player_name, p.status, p.position_bucket, p.problem) for p in parsed["Austin"]] == [
        ("Bell", "Doubtful", "defense", "Hip Injury")
    ]


OURS = ["Columbus Crew", "Austin FC", "LAFC", "LA Galaxy", "D.C. United", "Inter Miami CF"]
GERMAN = ["Bayern Munich", "Dortmund", "Ein Frankfurt", "M'gladbach", "Hamburg", "St Pauli", "FC Koln"]


@pytest.mark.parametrize(
    ("theirs", "ours", "expected"),
    [
        ("Austin", OURS, "Austin FC"),
        ("Los Angeles FC", OURS, "LAFC"),
        ("Los Angeles Galaxy", OURS, "LA Galaxy"),
        ("DC United", OURS, "D.C. United"),
        ("Bayern München", GERMAN, "Bayern Munich"),
        ("Borussia Mönchengladbach", GERMAN, "M'gladbach"),
        ("Eintracht Frankfurt", GERMAN, "Ein Frankfurt"),
        ("Hamburger SV", GERMAN, "Hamburg"),
        ("Borussia Dortmund", GERMAN, "Dortmund"),
        ("FC St. Pauli", GERMAN, "St Pauli"),
    ],
)
def test_starting11_club_names_map_to_ours(theirs, ours, expected):
    assert starting11_client.resolve_team_name(theirs, ours) == expected


def test_the_page_is_cached_per_league_and_a_failure_or_unreadable_page_is_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(starting11_client, "CACHE_DIR", tmp_path)
    calls = []

    class Response:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    def get(url, **kwargs):
        calls.append((url, kwargs["headers"]["User-Agent"]))
        return Response(PAGE)

    monkeypatch.setattr(starting11_client.http_session, "get", get)
    first = starting11_client.fetch_injuries_by_club("MLS")
    assert [p.player_name for p in first["Austin"]] == ["Torres", "Bell", "Thompson", "Anyone"]
    assert starting11_client.fetch_injuries_by_club("MLS") == first  # the second read came from the cache
    assert len(calls) == 1 and calls[0][0].endswith("/injuries/mls") and "Football-Forecaster" in calls[0][1]
    starting11_client.fetch_injuries_by_club("BUNDESLIGA")
    assert len(calls) == 2 and calls[1][0].endswith("/injuries/bundesliga")

    assert starting11_client.fetch_injuries_by_club("EPL") is None  # not a Starting 11 league

    for cached in tmp_path.iterdir():
        cached.unlink()

    def down(*args, **kwargs):
        raise requests.RequestException("down")

    monkeypatch.setattr(starting11_client.http_session, "get", down)
    assert starting11_client.fetch_injuries_by_club("MLS") is None

    monkeypatch.setattr(starting11_client.http_session, "get", lambda *a, **k: Response("<html>redesigned</html>"))
    assert starting11_client.fetch_injuries_by_club("MLS") is None
    assert list(tmp_path.iterdir()) == []  # an unreadable page is not cached


# --- syncing into the database ---------------------------------------------------------------


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def _stub_stats(monkeypatch, weight=0.6, known=True):
    squad = [object()] if known else []
    monkeypatch.setattr(asa_client, "fetch_team_season", lambda *a, **k: squad)
    monkeypatch.setattr(understat_client, "fetch_team_season", lambda *a, **k: squad)
    monkeypatch.setattr(player_importance, "aggregate_asa_team_output", lambda players: AsaTeamTotals(0.0, 0))
    monkeypatch.setattr(player_importance, "aggregate_understat_team_output", lambda players: UnderstatTeamTotals(0.0, 0))
    monkeypatch.setattr(player_importance, "resolve_api_importance_weight", lambda *a, **k: weight)
    monkeypatch.setattr(player_importance, "find_asa_player", lambda name, players: object() if players else None)
    monkeypatch.setattr(player_importance, "find_understat_player", lambda name, players: object() if players else None)


def test_mls_goes_to_starting11_not_the_empty_espn_roster_injuries(monkeypatch):
    _stub_stats(monkeypatch, weight=0.6)
    monkeypatch.setattr(starting11_client, "fetch_injuries_by_club", lambda code: starting11_client.parse_page(PAGE))
    monkeypatch.setattr(
        injuries.espn_client, "fetch_team_roster", lambda *a, **k: pytest.fail("ESPN roster injuries should not run for MLS")
    )
    with _session() as s:
        crew = get_or_create_team(s, "Columbus Crew", "MLS")
        austin = get_or_create_team(s, "Austin FC", "MLS")
        other = get_or_create_team(s, "Toronto FC", "MLS")
        replace_injuries(s, other.id, "api", [{"player_name": "Untouched", "position": "attack", "importance_weight": 0.5}])
        replace_injuries(s, crew.id, "manual", [{"player_name": "Keep Me", "position": "attack", "importance_weight": 0.5}])
        s.commit()

        synced, skipped = injuries.sync_injuries_to_db(s, MLS, 2026)
        s.commit()

        assert (synced, skipped) == (2, 0)
        by_name = {i.player_name: i for i in injuries_for_team(s, crew.id) if i.source == "api"}
        assert set(by_name) == {"Mihaj", "Reilly", "Keeper"}
        assert by_name["Mihaj"].importance_weight == 0.6 and by_name["Mihaj"].position == "defense"
        assert by_name["Mihaj"].note == "Starting XI: Hip Injury"
        austin_rows = {i.player_name: i for i in injuries_for_team(s, austin.id)}
        assert abs(austin_rows["Torres"].importance_weight - 0.15) < 1e-9  # 75% likely to play: a quarter
        assert austin_rows["Torres"].note == "Starting XI: Injury - 75% chance of playing"
        assert austin_rows["Thompson"].note == "Starting XI: Red Card - Suspended"
        assert any(i.source == "manual" and i.player_name == "Keep Me" for i in injuries_for_team(s, crew.id))
        assert [i.player_name for i in injuries_for_team(s, other.id)] == ["Untouched"]  # not on the page: left alone


def test_a_european_league_is_priced_from_understat(monkeypatch):
    _stub_stats(monkeypatch, weight=0.4)
    seen = []
    monkeypatch.setattr(understat_client, "fetch_team_season", lambda name, code, season: seen.append((name, code, season)) or [object()])
    page = _page(_card("bayern", "Bayern München", _row(9, "Kane", "FWD", "Calf Injury", "Out")), totals="All injuries 1 Out 0 Doubtful 0 Suspended")
    monkeypatch.setattr(starting11_client, "fetch_injuries_by_club", lambda code: starting11_client.parse_page(page))
    with _session() as s:
        bayern = get_or_create_team(s, "Bayern Munich", "BUNDESLIGA")
        s.commit()
        assert injuries.sync_injuries_to_db(s, BUNDESLIGA, 2026) == (1, 0)
        assert [(i.player_name, i.importance_weight) for i in injuries_for_team(s, bayern.id)] == [("Kane", 0.4)]
    assert seen == [("Bayern Munich", "BUNDESLIGA", 2026)]


def test_a_player_the_stats_do_not_know_gets_the_small_weight_not_the_default(monkeypatch):
    _stub_stats(monkeypatch, weight=0.9, known=False)
    monkeypatch.setattr(starting11_client, "fetch_injuries_by_club", lambda code: starting11_client.parse_page(PAGE))
    with _session() as s:
        crew = get_or_create_team(s, "Columbus Crew", "MLS")
        get_or_create_team(s, "Austin FC", "MLS")
        s.commit()
        injuries.sync_injuries_to_db(s, MLS, 2026)
        assert {i.importance_weight for i in injuries_for_team(s, crew.id)} == {injuries.UNPLAYED_WEIGHT}


def test_an_unreachable_page_leaves_the_stored_injuries_alone(monkeypatch):
    monkeypatch.setattr(starting11_client, "fetch_injuries_by_club", lambda code: None)
    with _session() as s:
        crew = get_or_create_team(s, "Columbus Crew", "MLS")
        replace_injuries(s, crew.id, "api", [{"player_name": "Yesterday", "position": "attack", "importance_weight": 0.5}])
        s.commit()
        assert injuries.sync_injuries_to_db(s, MLS, 2026) == (0, 0)
        assert [i.player_name for i in injuries_for_team(s, crew.id)] == ["Yesterday"]


def test_a_club_that_matches_none_of_ours_is_skipped_and_a_fit_club_is_cleared(monkeypatch):
    _stub_stats(monkeypatch)
    page = _page(
        _card("atlantis", "Atlantis", _row(1, "Nobody", "DEF", "Knee Injury", "Out")),
        _card("austin", "Austin", _row(2, "Gone", "FWD", "Off the roster", "Out")),
        totals="All injuries 2 Out 0 Doubtful 0 Suspended",
    )
    monkeypatch.setattr(starting11_client, "fetch_injuries_by_club", lambda code: starting11_client.parse_page(page))
    with _session() as s:
        austin = get_or_create_team(s, "Austin FC", "MLS")
        replace_injuries(s, austin.id, "api", [{"player_name": "Stale", "position": "attack", "importance_weight": 0.5}])
        s.commit()
        assert injuries.sync_injuries_to_db(s, MLS, 2026) == (1, 1)
        assert injuries_for_team(s, austin.id) == []


def test_these_leagues_run_in_the_daily_refresh_only():
    assert injuries.STARTING11_LEAGUES == {"MLS", "BUNDESLIGA", "SERIEA", "LIGUE1"}
    assert not injuries.STARTING11_LEAGUES & injuries.SINGLE_REQUEST_LEAGUES  # so not in the quick refresh
    assert all(injuries.has_automatic_feed(League(code=c, name=c, seasons=["2627"])) for c in injuries.STARTING11_LEAGUES)


def test_a_name_with_accents_is_matched_to_a_stats_site_that_drops_them(monkeypatch):
    _stub_stats(monkeypatch, weight=0.5)
    looked_up = []

    def find(name, players):
        looked_up.append(name)
        return object() if name == "Pongracic" else None  # the stats site's own spelling

    monkeypatch.setattr(player_importance, "find_understat_player", find)
    page = _page(
        _card("fiorentina", "Fiorentina", _row(5, "Pongračić", "DEF", "Knee Injury", "Out")),
        totals="All injuries 1 Out 0 Doubtful 0 Suspended",
    )
    monkeypatch.setattr(starting11_client, "fetch_injuries_by_club", lambda code: starting11_client.parse_page(page))
    serie_a = League(code="SERIEA", name="Serie A", seasons=["2627"], csv_code="I1")
    with _session() as s:
        fiorentina = get_or_create_team(s, "Fiorentina", "SERIEA")
        s.commit()
        injuries.sync_injuries_to_db(s, serie_a, 2026)
        assert [(i.player_name, i.importance_weight) for i in injuries_for_team(s, fiorentina.id)] == [("Pongračić", 0.5)]
    assert looked_up == ["Pongračić", "Pongracic"]  # his own spelling first, then without the marks
