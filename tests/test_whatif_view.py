"""The what-if tab's pure helpers (dashboard/whatif.py) and the tab itself
(the Predictor page's "AI what-if"), including that it never saves anything."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from soccer_predictor.ai.note_parser import ParsedFormNote, ParsedInjuryNote
from soccer_predictor.ai.whatif_parser import WhatIfEntries
from soccer_predictor.dashboard import whatif
from soccer_predictor.model.injury_adjustment import ATTACK_IMPACT_FACTOR
from soccer_predictor.model.markets import MatchPrediction, PredictionBreakdown
from soccer_predictor.storage.models import Base, Injury, PredictionRecord, TeamFormNote
from soccer_predictor.storage.repository import get_or_create_team

TODAY = dt.date(2026, 10, 27)


def _injury(team_id, weight=0.4, position="attack", name="Saka"):
    return ParsedInjuryNote(name, "x", team_id, position, weight, None, "why")


def _form(team_id, magnitude=-0.4, affects="both"):
    return ParsedFormNote("x", team_id, magnitude, affects, TODAY, "flat", "why")


def test_entries_are_split_by_side_and_scaled_by_the_sliders():
    entries = WhatIfEntries(
        injuries=[_injury(1, 0.4), _injury(2, 0.2, "defense", "Keeper")],
        form_notes=[_form(2, -0.5, "attack")],
    )
    full = whatif.build_adjustments(entries, home_id=1)
    assert [(i.player_name, i.importance_weight) for i in full.home_injuries] == [("Saka", 0.4)]
    assert [(i.player_name, i.position) for i in full.away_injuries] == [("Keeper", "defense")]
    assert [(f.magnitude, f.affects) for f in full.away_form] == [(-0.5, "attack")] and not full.home_form

    half = whatif.build_adjustments(entries, 1, injury_scale=[0.5, 1.0], form_scale=[0.0])
    assert half.home_injuries[0].importance_weight == pytest.approx(0.2)
    assert half.away_injuries[0].importance_weight == pytest.approx(0.2)
    assert half.away_form[0].magnitude == 0


def _prediction(home_win, draw, away_win, score=(1, 1), lam=(1.4, 1.1)):
    return MatchPrediction(
        home_win=home_win, draw=draw, away_win=away_win, over_2_5=0.5, under_2_5=0.5,
        both_teams_to_score=0.55, top_scorelines=[(*score, 0.12)],
        breakdown=PredictionBreakdown(
            1, 1, 1, 1, 1, 1, 1, 1, 1.3, -0.1, lam[0], lam[1], n_matches=10, fitted_at="x", xi=0.001
        ),
    )


def test_the_comparison_shows_both_columns_and_the_change():
    rows = whatif.comparison_rows(
        _prediction(0.50, 0.25, 0.25), _prediction(0.40, 0.28, 0.32, score=(1, 0), lam=(1.2, 1.1))
    )
    by_label = {r[""]: r for r in rows}
    assert by_label["Home win"] == {"": "Home win", "Model only": "50.0%", "With this note": "40.0%", "Change": "−10.0 pts"}
    assert by_label["Away win"]["Change"] == "+7.0 pts"
    assert by_label["Most likely score"]["Change"] == "changed"
    assert by_label["Expected home goals"]["Change"] == "−0.20"
    assert by_label["Expected away goals"]["Change"] == "no change"
    same = whatif.comparison_rows(_prediction(0.5, 0.25, 0.25), _prediction(0.5, 0.25, 0.25))
    assert {r["Change"] for r in same} == {"no change"}


def test_the_wording_uses_the_models_own_numbers():
    headline, maths = whatif.describe_injury_entry("Saka", "Arsenal", "attack", 0.4)
    assert "Saka" in headline and "Arsenal" in headline
    assert f"0.40 × {ATTACK_IMPACT_FACTOR:g}" in maths and "−12.0%" in maths
    _, defence = whatif.describe_injury_entry("Keeper", "Chelsea", "defense", 0.5)
    assert "5.0% more goals" in defence
    _, capped = whatif.describe_injury_entry("Everyone", "Arsenal", "attack", 2.0)
    assert "−30.0%" in capped  # the model's own floor
    headline, form = whatif.describe_form_entry("Struggling to score.", "Chelsea", -0.4, "both")
    assert "bad form" in headline and "Struggling to score" in headline and "Struggling to score." not in headline
    assert "attack −6.0%" in form and "goals conceded +6.0%" in form


# --- the tab -----------------------------------------------------------------------------------------


@pytest.fixture
def db(monkeypatch):
    """An in-memory database with two teams, shared with the dashboard code."""
    from contextlib import contextmanager

    from soccer_predictor.dashboard.views import predictor

    engine = create_engine(
        "sqlite://", future=True, connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    home = get_or_create_team(session, "Arsenal", "EPL")
    away = get_or_create_team(session, "Chelsea", "EPL")
    session.commit()

    @contextmanager
    def scope():
        yield session
        session.flush()

    monkeypatch.setattr(predictor, "session_scope", scope)
    monkeypatch.setattr(
        "soccer_predictor.prediction.service.resolve_current_attack_strength", lambda *a, **k: None
    )
    yield session, home.id, away.id
    session.close()


def _tab_app():
    from soccer_predictor.config import League
    from soccer_predictor.dashboard.views import predictor
    from soccer_predictor.model.dixon_coles import DixonColesParams
    from soccer_predictor.storage.repository import teams_for_league

    with predictor.session_scope() as session:
        names = teams_for_league(session, "EPL")
    home_id, away_id = sorted(names)
    params = DixonColesParams(
        league_code="EPL",
        attack={home_id: 1.4, away_id: 1.0},
        defense={home_id: 0.9, away_id: 1.1},
        home_advantage=1.3,
        rho=-0.1,
        xi=0.0018,
        fitted_at="2026-01-01T00:00:00",
        n_matches=100,
    )
    predictor._render_whatif_section(League(code="EPL", name="English Premier League", seasons=["2627"]), params, names)


def _open_tab():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_function(_tab_app, default_timeout=60).run()


def test_the_tab_is_off_without_a_key(db, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = _open_tab()
    assert not at.exception
    assert any("turned off" in i.value for i in at.info) and not at.button


def _fresh_limits(monkeypatch):
    from soccer_predictor.dashboard import whatif_access as wa
    from soccer_predictor.limits import SlidingWindow

    monkeypatch.setattr(wa, "MINUTE_LIMITER", SlidingWindow(100, 60))
    monkeypatch.setattr(wa, "DAY_LIMITER", SlidingWindow(1000, 86400))


def _home_win(at):
    rows = {row[""]: row for row in at.table[0].value.reset_index().to_dict("records")}
    return float(rows["Home win"]["With this note"].rstrip("%")), float(rows["Home win"]["Model only"].rstrip("%"))


def test_asking_shows_before_and_after_saves_nothing_and_sliders_recompute(db, monkeypatch):
    from soccer_predictor.ai import whatif_parser

    session, home_id, away_id = db
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    _fresh_limits(monkeypatch)
    whatif_parser.clear_cache()
    asked = []

    def fake_parse(note, hid, hname, aid, aname):
        asked.append(note)
        return WhatIfEntries(injuries=[_injury(hid, 0.5)], form_notes=[_form(aid, -0.5, "attack")])

    monkeypatch.setattr(whatif_parser, "parse_whatif_note", fake_parse)

    at = _open_tab()
    assert not at.exception and at.button[0].label == "Ask the AI"
    at.text_area[0].set_value("Saka is out and Chelsea can't score")
    at.button[0].click().run()

    assert not at.exception and asked == ["Saka is out and Chelsea can't score"]
    with_note, model_only = _home_win(at)
    assert with_note < model_only  # the home side lost its attacker
    assert len(at.slider) == 2
    assert any("not saved" in c.value for c in at.caption)

    at.slider[0].set_value(0.0).run()  # stop counting the injury: no new AI request
    assert asked == ["Saka is out and Chelsea can't score"]
    softer, _ = _home_win(at)
    assert softer > with_note

    for model in (Injury, TeamFormNote, PredictionRecord):
        assert session.scalar(select(func.count()).select_from(model)) == 0


def test_an_error_or_an_empty_read_is_shown_not_crashed(db, monkeypatch):
    from soccer_predictor.ai import whatif_parser

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    _fresh_limits(monkeypatch)
    whatif_parser.clear_cache()

    def at_capacity(*a):
        raise whatif_parser.WhatIfAtCapacity("used up")

    monkeypatch.setattr(whatif_parser, "parse_whatif_note", at_capacity)
    at = _open_tab()
    at.text_area[0].set_value("anything")
    at.button[0].click().run()
    assert not at.exception and any("tomorrow" in w.value for w in at.warning)

    monkeypatch.setattr(whatif_parser, "parse_whatif_note", lambda *a: WhatIfEntries())
    at = _open_tab()
    at.text_area[0].set_value("write me a poem")
    at.button[0].click().run()
    assert not at.exception and any("nothing to change" in i.value for i in at.info)
    assert len(at.table) == 0


def test_the_visitors_asks_are_limited(db, monkeypatch):
    from soccer_predictor.ai import whatif_parser
    from soccer_predictor.dashboard import whatif_access as wa

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    _fresh_limits(monkeypatch)
    whatif_parser.clear_cache()
    monkeypatch.setattr(wa, "COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(wa, "MAX_ASKS_PER_SESSION", 2)
    calls = []
    monkeypatch.setattr(
        whatif_parser, "parse_whatif_note", lambda *a: calls.append(a) or WhatIfEntries()
    )
    at = _open_tab()
    for text in ("one", "two", "three"):
        at.text_area[0].set_value(text)
        at.button[0].click().run()
    assert len(calls) == 2
    assert any("limit" in w.value for w in at.warning)


# --- the merged "AI notes" tab: free what-if on top, owner-only notes below ----------------------


def _combined_app():
    from soccer_predictor.dashboard.views import predictor
    from soccer_predictor.model.dixon_coles import DixonColesParams
    from soccer_predictor.config import League
    from soccer_predictor.storage.repository import teams_for_league

    with predictor.session_scope() as session:
        names = teams_for_league(session, "EPL")
    home_id, away_id = sorted(names)
    params = DixonColesParams(
        league_code="EPL", attack={home_id: 1.4, away_id: 1.0}, defense={home_id: 0.9, away_id: 1.1},
        home_advantage=1.3, rho=-0.1, xi=0.0018, fitted_at="2026-01-01T00:00:00", n_matches=100,
    )
    predictor._render_whatif_section(League(code="EPL", name="English Premier League", seasons=["2627"]), params, names)
    import streamlit as st

    st.divider()
    predictor._render_notes_gate()


def test_the_free_section_works_while_the_owner_section_stays_locked(db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("NOTES_PASSWORD", "a-long-owner-password")
    at = AppTest.from_function(_combined_app, default_timeout=60).run()
    assert not at.exception
    assert any("how could a note move the odds" in s.value for s in at.subheader)
    assert at.button[0].label == "Ask the AI"  # open to everyone
    assert at.text_input[0].label == "Password"  # the owner's gate, and no chat input without it
    assert not at.chat_input


def test_without_an_owner_password_only_the_free_section_is_usable(db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("NOTES_PASSWORD", raising=False)
    at = AppTest.from_function(_combined_app, default_timeout=60).run()
    assert not at.exception
    assert at.button[0].label == "Ask the AI"
    assert any("Adding notes is turned off" in i.value for i in at.info)
