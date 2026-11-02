from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import api_client, squad
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, upsert_team_alias

LEAGUE = League(
    code="EPL", name="English Premier League", csv_code="E0", api_competition_id=2021, seasons=["2425"]
)

RAW_RESPONSE = {
    "teams": [
        {
            "name": "Arsenal FC",
            "squad": [
                {"name": "Bukayo Saka", "position": "Attacker", "nationality": "England", "dateOfBirth": "2001-09-05"},
                {"name": "Kepa Arrizabalaga", "position": "Goalkeeper", "nationality": "Spain", "dateOfBirth": "1994-10-03"},
            ],
        },
        {
            "name": "Chelsea FC",
            "squad": [
                {"name": "Robert Sanchez", "position": "Goalkeeper", "nationality": "Spain", "dateOfBirth": "1997-11-18"},
            ],
        },
    ]
}


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        arsenal = get_or_create_team(s, "Arsenal", "EPL")
        upsert_team_alias(s, arsenal, "Arsenal FC", source="api")
        chelsea = get_or_create_team(s, "Chelsea", "EPL")
        upsert_team_alias(s, chelsea, "Chelsea FC", source="api")
        s.commit()
        yield s


def test_finds_player_across_teams_case_insensitively(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: RAW_RESPONSE)

    results = squad.search_players_in_league(session, LEAGUE, "saka")

    assert len(results) == 1
    assert results[0].player.name == "Bukayo Saka"
    assert results[0].team_name == "Arsenal"


def test_matches_multiple_teams_for_a_shared_substring(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: RAW_RESPONSE)

    results = squad.search_players_in_league(session, LEAGUE, "Goalkeeper")
    # Substring matches player NAME only, not position -- "Goalkeeper" isn't
    # in any of these names, so this should find nothing.
    assert results == []

    results = squad.search_players_in_league(session, LEAGUE, "sanchez")
    assert len(results) == 1
    assert results[0].team_name == "Chelsea"


def test_returns_empty_list_for_no_match(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: RAW_RESPONSE)

    assert squad.search_players_in_league(session, LEAGUE, "Zzzznonexistent") == []


def test_returns_empty_list_when_squad_fetch_fails(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: {"teams": []})

    assert squad.search_players_in_league(session, LEAGUE, "Saka") == []


# --- searching a club's name too -------------------------------------------------------------

CLUB_RESPONSE = {
    "teams": [
        {
            "name": "Arsenal FC",
            "squad": [
                {"name": "Bukayo Saka", "position": "Attacker", "nationality": "England", "dateOfBirth": "2001-09-05"},
                {"name": "Kepa Arrizabalaga", "position": "Goalkeeper", "nationality": "Spain", "dateOfBirth": "1994-10-03"},
            ],
        },
        {
            "name": "Chelsea FC",
            "squad": [
                {"name": "Robert Sanchez", "position": "Goalkeeper", "nationality": "Spain", "dateOfBirth": "1997-11-18"},
                {"name": "Arsenal Fan", "position": "Midfield", "nationality": "England", "dateOfBirth": "2000-01-01"},
            ],
        },
        {
            "name": "Atlético Madrid",
            "squad": [{"name": "Julián Álvarez", "position": "Attacker", "nationality": "Argentina", "dateOfBirth": "2000-01-31"}],
        },
    ]
}


@pytest.fixture
def club_session(session):
    atletico = get_or_create_team(session, "Ath Madrid", "EPL")
    upsert_team_alias(session, atletico, "Atlético Madrid", source="api")
    session.commit()
    return session


def _names(results):
    return [(r.team_name, r.player.name) for r in results]


def test_a_club_name_brings_the_whole_squad_when_asked_to(club_session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: CLUB_RESPONSE)

    results = squad.search_players_in_league(club_session, LEAGUE, "chelsea", include_club=True)

    assert _names(results) == [("Chelsea", "Robert Sanchez"), ("Chelsea", "Arsenal Fan")]
    # a player who happens to have the club's name in theirs is found too, and nobody is listed twice
    results = squad.search_players_in_league(club_session, LEAGUE, "arsenal", include_club=True)
    assert _names(results) == [("Arsenal", "Bukayo Saka"), ("Arsenal", "Kepa Arrizabalaga"), ("Chelsea", "Arsenal Fan")]


def test_a_club_is_found_by_any_spelling_on_record_ignoring_case_and_accents(club_session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: CLUB_RESPONSE)

    for query in ("Arsenal FC", "ATLETICO MADRID", "atlético", "ath madrid", "madrid"):
        assert [r.team_name for r in squad.search_players_in_league(club_session, LEAGUE, query, include_club=True)] != []
    results = squad.search_players_in_league(club_session, LEAGUE, "atletico", include_club=True)
    assert _names(results) == [("Ath Madrid", "Julián Álvarez")]  # the team's own name, not the API's spelling
    assert _names(squad.search_players_in_league(club_session, LEAGUE, "Arsenal FC", include_club=True))[:2] == [
        ("Arsenal", "Bukayo Saka"), ("Arsenal", "Kepa Arrizabalaga"),
    ]


def test_a_search_by_name_alone_does_not_pull_in_clubs_unless_asked(club_session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: CLUB_RESPONSE)

    assert _names(squad.search_players_in_league(club_session, LEAGUE, "chelsea")) == []  # the default: names only
    assert _names(squad.search_players_in_league(club_session, LEAGUE, "arsenal")) == [("Chelsea", "Arsenal Fan")]


def test_a_blank_or_unmatched_club_query_adds_nothing(club_session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: CLUB_RESPONSE)

    assert squad.search_players_in_league(club_session, LEAGUE, "Zzzznonexistent", include_club=True) == []
    # an empty query is not "every club": the page never searches for one, but the function stays safe
    assert squad._clubs_named(club_session, LEAGUE, {1: "Arsenal"}, "   ") == set()


# --- how much work a search is -------------------------------------------------------------------


def _count_statements(session, run) -> int:
    seen: list[str] = []

    def record(conn, cursor, statement, *args):
        seen.append(statement)

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", record)
    try:
        run()
    finally:
        event.remove(engine, "before_cursor_execute", record)
    return len(seen)


def test_a_search_costs_the_same_number_of_queries_however_many_teams_the_league_has(session, monkeypatch):
    """The Players page searches every team of the league each time a player is opened. Matching the
    API's names to our teams used to cost two queries per team per entry, which took over a second
    for the Premier League (relegated clubs stay in the table)."""
    monkeypatch.setattr(api_client, "get", lambda *a, **k: RAW_RESPONSE)

    def search():
        assert [r.player.name for r in squad.search_players_in_league(session, LEAGUE, "saka", include_club=True)] == ["Bukayo Saka"]

    few = _count_statements(session, search)
    for number in range(30):  # clubs the API's response has no entry for
        get_or_create_team(session, f"Old Club {number}", "EPL")
    session.commit()
    many = _count_statements(session, search)

    assert many - few <= 1  # at most the one extra row-fetching query, never one per team


def test_a_team_the_api_has_no_entry_for_has_no_squad(session, monkeypatch):
    monkeypatch.setattr(api_client, "get", lambda *a, **k: RAW_RESPONSE)
    old = get_or_create_team(session, "Old Club", "EPL")
    arsenal = get_or_create_team(session, "Arsenal", "EPL")
    session.commit()

    assert squad.fetch_squad_for_team(session, LEAGUE, old.id) is None
    assert [p.name for p in squad.fetch_squad_for_team(session, LEAGUE, arsenal.id)] == ["Bukayo Saka", "Kepa Arrizabalaga"]
    # a caller that already worked the entries out hands them in, and nothing is looked up again
    entries = squad._raw_entries_by_team_id(session, LEAGUE)
    assert set(entries) == {arsenal.id, get_or_create_team(session, "Chelsea", "EPL").id}
    assert squad.fetch_squad_for_team(session, LEAGUE, arsenal.id, raw_entries={}) is None
    assert squad.fetch_squad_for_team(session, LEAGUE, arsenal.id, raw_entries=entries) is not None
