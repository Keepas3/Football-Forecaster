"""Where an injured player plays (Goalkeeper, Defence, Midfield, Offence) -- not just the attack/defence
side of the rating -- from storage to the Injuries tab. The feeds' own parsing is tested with each feed."""

from __future__ import annotations

import datetime as dt
import re

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from soccer_predictor import config
from soccer_predictor.dashboard import injury_table as it
from soccer_predictor.dashboard.injury_tables import injury_players_table_html
from soccer_predictor.ingest import positions
from soccer_predictor.model.injury_adjustment import InjuryEntry
from soccer_predictor.prediction import service
from soccer_predictor.prediction.service import TeamInjuries
from soccer_predictor.storage import db
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, injuries_for_team, replace_injuries

TODAY = dt.date(2026, 10, 29)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


# --- the words -------------------------------------------------------------------------------


def test_goalkeepers_and_defenders_count_as_defence_everyone_else_as_attack():
    assert [positions.bucket(p) for p in (positions.GOALKEEPER, positions.DEFENCE)] == ["defense", "defense"]
    assert [positions.bucket(p) for p in (positions.MIDFIELD, positions.OFFENCE, None, "")] == ["attack"] * 4


def test_the_fantasy_feeds_number_positions_goalkeeper_to_forward():
    assert [positions.FANTASY_POSITIONS[n] for n in (1, 2, 3, 4)] == list(positions.POSITION_ORDER)


@pytest.mark.parametrize(
    ("typed", "word"),
    [
        ("striker", "Striker"),
        ("Winger", "Winger"),
        ("cdm", "Defensive mid"),
        ("CAM", "Attacking mid"),
        ("centerback", "Centre-back"),
        ("Centre-back", "Centre-back"),
        ("leftback", "Left-back"),
        ("Left back", "Left-back"),
        ("RB", "Right-back"),
        ("  midfielder ", "Midfield"),
        ("Keeper", "Goalkeeper"),
        ("libero", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_a_typed_position_becomes_one_of_the_sites_words(typed, word):
    assert positions.normalize_role(typed) == word


def test_each_finer_word_sits_in_its_broad_group_and_every_word_has_a_place_in_the_order():
    assert set(positions.GROUP.values()) == set(positions.POSITION_ORDER)
    groups = [positions.GROUP[word] for word in positions.ROLE_ORDER]
    assert groups == sorted(groups, key=positions.POSITION_ORDER.get)  # a group's words sit together, goalkeepers first
    assert all(positions.ROLE_ORDER[word] >= positions.ROLE_ORDER[group] for word, group in positions.GROUP.items())


def test_a_finer_word_counts_for_the_same_side_of_the_rating_as_its_group():
    for word, group in positions.GROUP.items():
        assert positions.bucket(word) == positions.bucket(group), word
    assert positions.bucket(positions.CENTRE_BACK) == "defense" and positions.bucket(positions.STRIKER) == "attack"


def test_every_american_soccer_analysis_position_is_one_of_our_words():
    assert set(positions.ASA_POSITIONS.values()) <= set(positions.GROUP)


# --- storage ---------------------------------------------------------------------------------


def test_the_role_is_stored_with_the_injury_and_is_none_when_the_source_gave_none(session):
    team = get_or_create_team(session, "Arsenal", "EPL")
    replace_injuries(
        session,
        team.id,
        "api",
        [
            {"player_name": "Raya", "position": "defense", "importance_weight": 0.5, "role": "Goalkeeper"},
            {"player_name": "Odd", "position": "attack", "importance_weight": 0.2, "role": ""},
            {"player_name": "Old", "position": "attack", "importance_weight": 0.2},
        ],
    )
    session.commit()

    assert {i.player_name: i.role for i in injuries_for_team(session, team.id)} == {
        "Raya": "Goalkeeper", "Odd": None, "Old": None,
    }


def test_an_old_database_gets_the_role_column_and_keeps_its_rows():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE injuries (id INTEGER PRIMARY KEY, player_name VARCHAR)"))
        conn.execute(text("INSERT INTO injuries (player_name) VALUES ('Saka')"))

    assert db.ensure_injury_role_column(engine) is True
    assert "role" in {c["name"] for c in inspect(engine).get_columns("injuries")}
    assert db.ensure_injury_role_column(engine) is False  # already there
    with engine.connect() as conn:
        assert conn.execute(text("SELECT player_name, role FROM injuries")).all() == [("Saka", None)]


# --- from storage to the prediction service ----------------------------------------------------


def test_stored_and_manual_injuries_reach_the_service_with_their_role(session, monkeypatch):
    team = get_or_create_team(session, "Arsenal", "EPL")
    replace_injuries(
        session,
        team.id,
        "api",
        [
            {"player_name": "Rice", "position": "attack", "importance_weight": 0.4, "role": "Midfield"},
            {"player_name": "Old", "position": "defense", "importance_weight": 0.1},  # synced before roles
        ],
    )
    session.commit()
    monkeypatch.setattr(
        service,
        "load_manual_injuries",
        lambda: [
            config.ManualInjury("Arsenal", "Saka", "attack", 0.8, role="right winger"),  # typed loosely
            config.ManualInjury("Arsenal", "Plain", "attack", 0.2),
            config.ManualInjury("Arsenal", "Typo", "attack", 0.1, role="libero"),  # not a position we know
        ],
    )

    entries = service.get_injuries_for_team(session, team.id, "Arsenal", TODAY).entries

    assert {e.player_name: e.role for e in entries} == {
        "Saka": "Right winger", "Plain": "", "Typo": "", "Rice": "Midfield", "Old": "",
    }


def test_the_manual_list_can_name_a_role_and_a_blank_one_means_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    (tmp_path / "injuries.yaml").write_text(
        "injuries:\n"
        "  - {team: Arsenal, player: Saka, position: attack, importance_weight: 0.8, role: Offence}\n"
        "  - {team: Arsenal, player: Rice, position: attack, importance_weight: 0.3, role: }\n"
        "  - {team: Arsenal, player: Raya, position: defense, importance_weight: 0.5}\n",
        encoding="utf-8",
    )

    assert [m.role for m in config.load_manual_injuries()] == ["Offence", "", ""]


# --- what the Injuries tab shows ---------------------------------------------------------------


def _team(*entries):
    return TeamInjuries(
        entries=list(entries),
        sources=["api"] * len(entries),
        expected_return_dates=[None] * len(entries),
        notes=["FPL: Knock"] * len(entries),
    )


def _frame():
    team = _team(
        InjuryEntry("Raya", "defense", 0.5, "Goalkeeper"),
        InjuryEntry("Saliba", "defense", 0.4, "Centre-back"),
        InjuryEntry("Timber", "defense", 0.35, "Left-back"),
        InjuryEntry("Rice", "attack", 0.3, "Defensive mid"),
        InjuryEntry("Saka", "attack", 0.2, "Winger"),
        InjuryEntry("Havertz", "attack", 0.15, "Striker"),
        InjuryEntry("Note Defender", "defense", 0.1),  # a chat note or an older row: no role
        InjuryEntry("Note Attacker", "attack", 0.05),
        InjuryEntry("Typo", "attack", 0.01, "Libero"),  # not one of our words: not shown as if it were
    )
    return it.players_frame(it.player_rows("Arsenal", team, TODAY))


def test_a_player_shows_where_he_plays_and_falls_back_to_the_side_he_counts_for():
    frame = _frame()

    assert dict(zip(frame["Player"], frame["Position"], strict=True)) == {
        "Raya": "Goalkeeper", "Saliba": "Centre-back", "Timber": "Left-back", "Rice": "Defensive mid",
        "Saka": "Winger", "Havertz": "Striker", "Note Defender": "Defence", "Note Attacker": "Attack", "Typo": "Attack",
    }


def test_the_position_filter_lists_goalkeepers_first_and_filters_on_them():
    frame = _frame()

    assert it.position_choices(frame) == [
        "Goalkeeper", "Defence", "Centre-back", "Left-back", "Defensive mid", "Winger", "Striker", "Attack",
    ]
    assert list(it.filter_players(frame, positions=["Defensive mid", "Goalkeeper"])["Player"]) == ["Raya", "Rice"]
    assert it.position_choices(it.players_frame([])) == []


def test_the_position_column_sorts_goalkeepers_first_and_unnamed_positions_last():
    markup = injury_players_table_html(_frame(), {"Arsenal": 1}, "EPL")
    values = dict(re.findall(r'<td class="left" data-v="(\d+)">([A-Za-z -]+)</td>', markup))
    order = positions.ROLE_ORDER

    assert {int(key): word for key, word in values.items()} == {
        order["Goalkeeper"]: "Goalkeeper", order["Defence"]: "Defence", order["Centre-back"]: "Centre-back",
        order["Left-back"]: "Left-back", order["Defensive mid"]: "Defensive mid", order["Winger"]: "Winger",
        order["Striker"]: "Striker", len(order): "Attack",  # a position the source did not give: after them all
    }
    assert "goalkeepers first" in markup
