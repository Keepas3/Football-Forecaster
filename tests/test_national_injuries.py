"""A club player's injury shown on his national team too (ingest/national_injuries.py)."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.dashboard import injury_table as it
from soccer_predictor.dashboard.injury_tables import injury_players_table_html
from soccer_predictor.ingest import national_injuries as ni
from soccer_predictor.ingest import refresh
from soccer_predictor.ingest.squad import SquadPlayer
from soccer_predictor.prediction.service import get_injuries_for_team
from soccer_predictor.storage.models import Base, Injury, Team
from soccer_predictor.storage.repository import replace_injuries

EPL = League(code="EPL", name="Premier League", csv_code="E0", seasons=["2627"], api_competition_id=2021)
UCL = League(code="UCL", name="Champions League", seasons=["2627"], api_competition_id=2001)
MLS = League(code="MLS", name="MLS", seasons=["2026"], data_source="espn", espn_league_slug="usa.1")
NL = League(code="NL", name="Nations League", seasons=["2627"], team_type="national", data_source="espn")
WC = League(code="WC", name="World Cup", seasons=["2026"], team_type="national", api_competition_id=2000)

SAKA = SquadPlayer("Bukayo Saka", "Offence", "England", "2001-09-05", player_id=99813)
RICE = SquadPlayer("Declan Rice", "Midfield", "England", "1999-01-14", player_id=1)
RAMSDALE = SquadPlayer("Aaron Ramsdale", "Goalkeeper", "England", "1998-05-14", player_id=2)
ODEGAARD = SquadPlayer("Martin Ødegaard", "Midfield", "Norway", "1998-12-17", player_id=3)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _team(session, league, name):
    team = Team(canonical_name=name, league_code=league)
    session.add(team)
    session.flush()
    return team


def _world(session, monkeypatch, squads=None, club_squads=None):
    """Arsenal (an EPL and a UCL row) with injuries; England, Norway and Spain in the Nations League."""
    arsenal = _team(session, "EPL", "Arsenal")
    arsenal_ucl = _team(session, "UCL", "Arsenal FC")
    england, norway, spain = (_team(session, "NL", n) for n in ("England", "Norway", "Spain"))
    leagues = {"EPL": EPL, "UCL": UCL, "MLS": MLS, "NL": NL, "WC": WC}
    monkeypatch.setattr(ni, "load_leagues", lambda: leagues)
    national = squads if squads is not None else {"england": {99813, 1}, "norway": {3}}
    monkeypatch.setattr(ni, "fetch_national_squads", lambda _: national)
    by_team = club_squads or {arsenal.id: [SAKA, RICE, RAMSDALE, ODEGAARD], arsenal_ucl.id: [SAKA, RICE, ODEGAARD]}
    monkeypatch.setattr(ni.squad_module, "_raw_entries_by_team_id", lambda *a: {})
    monkeypatch.setattr(ni.squad_module, "fetch_squad_for_team", lambda s, lg, team_id, raw=None: by_team.get(team_id))
    return SimpleNamespace(arsenal=arsenal, arsenal_ucl=arsenal_ucl, england=england, norway=norway, spain=spain)


def _add_api(session, team, name, note, position="attack", role="Midfield"):
    """Adds one automatic-feed injury to the team's list."""
    existing = [
        {
            "player_name": r.player_name,
            "position": r.position,
            "role": r.role,
            "importance_weight": r.importance_weight,
            "note": r.note,
        }
        for r in session.scalars(select(Injury).where(Injury.team_id == team.id, Injury.source == "api")).all()
    ]
    entry = {"player_name": name, "position": position, "role": role, "importance_weight": 0.4, "note": note}
    replace_injuries(session, team.id, "api", [*existing, entry])


def _club_rows(session, team):
    return session.scalars(select(Injury).where(Injury.team_id == team.id, Injury.source == "club")).all()


# --- finding the player ----------------------------------------------------------------------------


def test_a_feed_spelling_is_matched_to_the_squad_player():
    squad = [SAKA, RICE, ODEGAARD]
    assert ni.find_squad_player("Bukayo Saka", squad) is SAKA  # the whole name
    assert ni.find_squad_player("Saka", squad) is SAKA  # a surname, as Starting 11 gives it
    assert ni.find_squad_player("Odegaard", squad) is ODEGAARD  # without the accent
    assert ni.find_squad_player("Nobody", squad) is None


def test_two_players_who_could_be_meant_match_neither():
    squad = [
        SquadPlayer("Gabriel Martinelli", "Offence", "Brazil", None, 5),
        SquadPlayer("Gabriel Magalhaes", "Defence", "Brazil", None, 6),
    ]
    assert ni.find_squad_player("Gabriel", squad) is None


@pytest.mark.parametrize(
    "a, b",
    [("Czech Republic", "Czechia"), ("Turkey", "Türkiye"), ("Republic of Ireland", "Ireland")],
)
def test_a_country_has_one_key_whatever_a_source_calls_it(a, b):
    assert ni.nation_key(a) == ni.nation_key(b)


def test_the_note_keeps_the_problem_and_return_date_the_table_reads_and_names_the_club():
    note = ni.club_note("FPL: Ankle injury - Expected back 24 Oct", "Arsenal")
    assert note == "Club: Ankle injury (Arsenal) - Expected back 24 Oct"
    parsed = it.parse_note(note, dt.date(2026, 11, 13))
    assert parsed.problem == "Ankle injury (Arsenal)"
    assert parsed.back is not None and (parsed.back.month, parsed.back.day) == (10, 24)
    assert ni.club_note("Biwenger: Out", "Chelsea") == "Club: Out (Chelsea)"


# --- the sync --------------------------------------------------------------------------------------


def test_a_club_injury_is_listed_under_the_players_country(session, monkeypatch):
    w = _world(session, monkeypatch)
    _add_api(session, w.arsenal, "Bukayo Saka", "FPL: Thigh injury - 75% chance of playing")

    assert ni.sync_club_injuries_to_national_teams(session, NL) == (1, 2)  # England listed; Norway, Spain cleared

    (row,) = _club_rows(session, w.england)
    assert (row.player_name, row.source, row.importance_weight, row.position, row.role) == (
        "Bukayo Saka",
        "club",
        0.0,
        "attack",
        "Midfield",
    )
    assert row.note == "Club: Thigh injury (Arsenal) - 75% chance of playing"
    assert _club_rows(session, w.norway) == [] and _club_rows(session, w.spain) == []


def test_it_is_listed_in_each_national_competition_the_country_is_in(session, monkeypatch):
    w = _world(session, monkeypatch)
    wc_england = _team(session, "WC", "England")
    _add_api(session, w.arsenal, "Saka", "FPL: Hamstring")

    ni.sync_club_injuries_to_national_teams(session, WC)

    assert [r.player_name for r in _club_rows(session, wc_england)] == ["Bukayo Saka"]
    assert _club_rows(session, w.england) == []  # only the competition asked for


def test_the_same_player_listed_under_his_league_and_the_champions_league_is_listed_once(session, monkeypatch):
    w = _world(session, monkeypatch)
    _add_api(session, w.arsenal, "Bukayo Saka", "FPL: Thigh injury")
    _add_api(session, w.arsenal_ucl, "Bukayo Saka", "Biwenger: Thigh injury")

    ni.sync_club_injuries_to_national_teams(session, NL)

    assert len(_club_rows(session, w.england)) == 1


def test_a_player_who_is_not_in_a_national_squad_is_not_listed(session, monkeypatch):
    w = _world(session, monkeypatch)
    _add_api(session, w.arsenal, "Aaron Ramsdale", "FPL: Knock")  # in the club squad, in no tournament squad

    ni.sync_club_injuries_to_national_teams(session, NL)

    assert _club_rows(session, w.england) == []


def test_a_club_suspension_does_not_carry_over_to_the_country(session, monkeypatch):
    w = _world(session, monkeypatch)
    _add_api(session, w.arsenal, "Declan Rice", "FPL: Suspended until 29 Nov")
    _add_api(session, w.arsenal, "Bukayo Saka", "Starting XI: Red Card - Suspended")

    ni.sync_club_injuries_to_national_teams(session, NL)

    assert _club_rows(session, w.england) == []


def test_a_recovered_player_is_removed_on_the_next_sync(session, monkeypatch):
    w = _world(session, monkeypatch)
    _add_api(session, w.arsenal, "Bukayo Saka", "FPL: Thigh injury")
    ni.sync_club_injuries_to_national_teams(session, NL)
    assert len(_club_rows(session, w.england)) == 1

    replace_injuries(session, w.arsenal.id, "api", [])  # the club feed no longer lists him
    ni.sync_club_injuries_to_national_teams(session, NL)

    assert _club_rows(session, w.england) == []


def test_nothing_changes_when_no_national_squad_could_be_read(session, monkeypatch):
    w = _world(session, monkeypatch, squads={})
    entry = {"player_name": "Bukayo Saka", "position": "attack", "role": None, "importance_weight": 0.0, "note": "Club: x"}
    replace_injuries(session, w.england.id, "club", [entry])

    assert ni.sync_club_injuries_to_national_teams(session, NL) == (0, 0)

    assert len(_club_rows(session, w.england)) == 1  # yesterday's list beats wiping it


def test_a_club_league_is_not_given_club_rows(session, monkeypatch):
    _world(session, monkeypatch)
    assert ni.sync_club_injuries_to_national_teams(session, EPL) == (0, 0)


def test_mls_clubs_are_not_read(session, monkeypatch):
    # ESPN squads carry no football-data.org id, so the MLS is skipped rather than guessed at
    _world(session, monkeypatch)
    mls_team = _team(session, "MLS", "LAFC")
    _add_api(session, mls_team, "Bukayo Saka", "Starting XI: Knock")
    assert ni.collect_club_injured(session, {"MLS": MLS}) == []


def test_the_chat_notes_of_a_club_are_not_copied(session, monkeypatch):
    w = _world(session, monkeypatch)
    note = {"player_name": "Bukayo Saka", "position": "attack", "role": None, "importance_weight": 0.3, "note": "Chat"}
    replace_injuries(session, w.arsenal.id, "chat", [note])

    ni.sync_club_injuries_to_national_teams(session, NL)

    assert _club_rows(session, w.england) == []


# --- display only ----------------------------------------------------------------------------------


def test_club_injuries_never_reach_a_forecast_but_are_there_for_the_pages_that_list_them(session, monkeypatch):
    w = _world(session, monkeypatch)
    _add_api(session, w.arsenal, "Bukayo Saka", "FPL: Thigh injury")
    ni.sync_club_injuries_to_national_teams(session, NL)
    today = dt.date(2026, 11, 13)

    assert get_injuries_for_team(session, w.england.id, "England", today).entries == []  # what predict_fixture asks
    listed = get_injuries_for_team(session, w.england.id, "England", today, include_club=True)
    assert [(e.player_name, e.importance_weight) for e in listed.entries] == [("Bukayo Saka", 0.0)]
    assert listed.sources == ["club"]


def test_the_table_labels_them_and_draws_no_impact_bar(session, monkeypatch):
    w = _world(session, monkeypatch)
    _add_api(session, w.arsenal, "Bukayo Saka", "FPL: Thigh injury")
    ni.sync_club_injuries_to_national_teams(session, NL)
    today = dt.date(2026, 11, 13)
    listed = get_injuries_for_team(session, w.england.id, "England", today, include_club=True)

    frame = it.players_frame(it.player_rows("England", listed, today))
    assert frame.loc[0, "Source"] == it.CLUB_INJURY_LABEL
    html = injury_players_table_html(frame, {"England": w.england.id}, "NL")
    assert "Club injury" in html and 'class="impact"' not in html


# --- the refresh -----------------------------------------------------------------------------------


def _patch_refresh(monkeypatch, calls):
    monkeypatch.setattr(refresh.injuries_module, "sync_injuries_to_db", lambda *a, **k: (0, 0))
    monkeypatch.setattr(refresh, "sync_fixtures_to_db", lambda *a, **k: (0, 0))
    monkeypatch.setattr(refresh, "sync_results_to_db", lambda *a, **k: (0, 0))
    monkeypatch.setattr(refresh, "seed_teams_from_api", lambda *a, **k: 0)
    monkeypatch.setattr(refresh, "load_latest_params", lambda *a, **k: None)
    monkeypatch.setattr(
        refresh.national_injuries,
        "sync_club_injuries_to_national_teams",
        lambda s, league: calls.append(league.code) or (7, 3),
    )

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(refresh, "session_scope", lambda: _Session())


def test_the_daily_refresh_of_a_national_competition_lists_the_club_injuries(monkeypatch):
    calls = []
    _patch_refresh(monkeypatch, calls)

    result = refresh.refresh_league(WC)

    assert calls == ["WC"] and result.club_injuries_listed == 7


def test_the_quick_refresh_and_club_leagues_leave_them_alone(monkeypatch):
    calls = []
    _patch_refresh(monkeypatch, calls)

    refresh.refresh_league(WC, quick=True)
    refresh.refresh_league(EPL)

    assert calls == []
