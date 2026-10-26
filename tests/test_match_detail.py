"""Match detail: ESPN summary parsing, event-id storage/lookup, the page helpers."""

from __future__ import annotations

import datetime as dt
import re

import pytest
import requests
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.dashboard.views import match_detail as view
from soccer_predictor.ingest import espn_client
from soccer_predictor.ingest.fixtures import sync_fixtures_to_db_espn, sync_results_to_db_espn
from soccer_predictor.storage import db
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import (
    espn_event_id_for_match,
    get_or_create_team,
    update_espn_team_id,
    upsert_fixture,
    upsert_match,
)

LEAGUE = League(code="NL", name="UEFA Nations League", seasons=["2627"], data_source="espn", espn_league_slug="uefa.nations")


def _summary(state="post", with_rosters=True):
    """A trimmed copy of the shape of a real ESPN summary (Azerbaijan 1-0 Lithuania)."""
    def player(name, jersey, pos, starter, **extra):
        return {
            "starter": starter,
            "jersey": jersey,
            "athlete": {"displayName": name},
            "position": {"abbreviation": pos},
            "subbedIn": extra.pop("subbedIn", False),
            "subbedOut": extra.pop("subbedOut", False),
            "stats": [{"name": k, "value": float(v)} for k, v in extra.items()],
        }

    rosters = []
    if with_rosters:
        rosters = [
            {
                "team": {"id": "581"},
                "formation": "3-4-3",
                "roster": [
                    player("Emil Balayev", "1", "G", True, saves=2.0),
                    player("Abdulakh Khaybulaev", "9", "F", True, totalGoals=1.0, yellowCards=1.0, subbedOut=True),
                    player("Vusal Isgandarli", "20", "M", False, subbedIn=True),
                ],
            },
            {"team": {"id": "460"}, "formation": "4-4-2", "roster": [player("Ernestas Veliulis", "1", "G", True)]},
        ]
    return {
        "header": {
            "competitions": [
                {
                    "status": {"type": {"state": state, "completed": state == "post", "shortDetail": "FT"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"id": "581", "displayName": "Azerbaijan"}, "score": "1"},
                        {"homeAway": "away", "team": {"id": "460", "displayName": "Lithuania"}, "score": "0"},
                    ],
                }
            ]
        },
        "boxscore": {
            "teams": [
                {
                    "team": {"id": "581"},
                    "statistics": [
                        {"name": "possessionPct", "displayValue": "41.9", "label": "Possession"},
                        {"name": "totalShots", "displayValue": "9", "label": "Shots"},
                    ],
                },
                {
                    "team": {"id": "460"},
                    "statistics": [
                        {"name": "possessionPct", "displayValue": "58.1", "label": "Possession"},
                        {"name": "totalShots", "displayValue": "5", "label": "Shots"},
                    ],
                },
            ]
        },
        "rosters": rosters,
        "keyEvents": [
            {"type": {"type": "kickoff"}, "text": "First Half begins.", "clock": {"displayValue": ""}},
            {
                "type": {"type": "goal"},
                "text": "Goal! Azerbaijan 1, Lithuania 0.",
                "clock": {"displayValue": "25'"},
                "team": {"id": "581"},
            },
            {"type": {"type": "yellow-card"}, "text": "Booked.", "clock": {"displayValue": "43'"}, "team": {"id": "460"}},
            {"type": {"type": "halftime"}, "text": "Half time", "clock": {"displayValue": "45'"}},
            {"type": {"type": "substitution"}, "text": "Sub.", "clock": {"displayValue": "64'"}, "team": {"id": "581"}},
        ],
        "gameInfo": {"venue": {"fullName": "Neftci Arena", "address": {"city": "Baku"}}, "attendance": 5000},
    }


# --- ESPN client -----------------------------------------------------------------------------


def test_parse_event_keeps_the_event_id():
    event = {
        "id": 401861119,
        "date": "2026-10-04T13:00Z",
        "competitions": [
            {
                "status": {"type": {"completed": True, "state": "post"}},
                "competitors": [
                    {"homeAway": "home", "team": {"id": "581", "displayName": "Azerbaijan"}, "score": "1"},
                    {"homeAway": "away", "team": {"id": "460", "displayName": "Lithuania"}, "score": "0"},
                ],
            }
        ],
    }
    assert espn_client._parse_event(event).event_id == "401861119"
    del event["id"]
    assert espn_client._parse_event(event).event_id is None


def test_parse_match_detail_reads_teams_lineups_stats_and_timeline():
    detail = espn_client.parse_match_detail("401861119", _summary())

    assert (detail.home.name, detail.away.name) == ("Azerbaijan", "Lithuania")
    assert (detail.home_score, detail.away_score, detail.state, detail.status_label) == (1, 0, "post", "FT")
    assert (detail.venue, detail.city, detail.attendance) == ("Neftci Arena", "Baku", 5000)
    assert detail.home.formation == "3-4-3" and detail.has_lineups
    starters = [p for p in detail.home.players if p.starter]
    assert [p.name for p in starters] == ["Emil Balayev", "Abdulakh Khaybulaev"]
    scorer = starters[1]
    assert scorer.stats == {"totalGoals": 1, "yellowCards": 1} and scorer.subbed_out
    assert detail.home.stats["possessionPct"] == ("Possession", "41.9")
    # Kickoff/half-time are noise; goal, card and sub stay, in order, attributed to teams.
    assert [(e.minute, e.kind, e.team_id) for e in detail.events] == [
        ("25'", "goal", "581"),
        ("43'", "yellow", "460"),
        ("64'", "sub", "581"),
    ]


def test_parse_match_detail_before_kickoff_has_no_score_or_lineups():
    detail = espn_client.parse_match_detail("1", _summary(state="pre", with_rosters=False))
    assert detail.home_score is None and detail.away_score is None
    assert not detail.has_lineups


@pytest.mark.parametrize("payload", [{}, {"header": {}}, {"header": {"competitions": []}}, {"header": {"competitions": [{}]}}])
def test_parse_match_detail_rejects_malformed_payloads(payload):
    assert espn_client.parse_match_detail("1", payload) is None


def test_fetch_match_detail_picks_cache_lifetime_by_state_and_survives_failures(monkeypatch):
    seen = {}

    def fake_get(path, params=None, cache_ttl_seconds=0, **kwargs):
        seen["path"], seen["params"], seen["ttl"] = path, params, cache_ttl_seconds
        return _summary()

    monkeypatch.setattr(espn_client, "get", fake_get)
    assert espn_client.fetch_match_detail("uefa.nations", "99", "in").event_id == "99"
    assert seen == {"path": "/uefa.nations/summary", "params": {"event": "99"}, "ttl": 30}
    espn_client.fetch_match_detail("uefa.nations", "99", "post")
    assert seen["ttl"] == 24 * 3600

    def boom(*a, **k):
        raise requests.RequestException("down")

    monkeypatch.setattr(espn_client, "get", boom)
    assert espn_client.fetch_match_detail("uefa.nations", "99") is None
    monkeypatch.setattr(espn_client, "get", lambda *a, **k: ["not", "a", "dict"])
    assert espn_client.fetch_match_detail("uefa.nations", "99") is None


def _espn_match(event_id, home="581", away="460"):
    return espn_client.EspnMatch(
        espn_home_id=home,
        espn_away_id=away,
        home_name="Azerbaijan",
        away_name="Lithuania",
        date=dt.date(2026, 10, 4),
        kickoff_utc=dt.datetime(2026, 10, 4, 13, 0),
        completed=True,
        home_score=1,
        away_score=0,
        state="post",
        event_id=event_id,
    )


def test_find_event_id_matches_the_pair_on_the_days_scoreboard(monkeypatch):
    monkeypatch.setattr(
        espn_client, "fetch_day_matches", lambda *a, **k: [_espn_match("1", "7", "8"), _espn_match("2")]
    )
    assert espn_client.find_event_id("uefa.nations", dt.date(2026, 10, 4), "581", "460") == ("2", "post")
    assert espn_client.find_event_id("uefa.nations", dt.date(2026, 10, 4), "460", "581") is None


# --- storage ---------------------------------------------------------------------------------


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def test_ensure_espn_event_id_columns_upgrades_an_old_database_and_is_idempotent():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE matches (id INTEGER PRIMARY KEY, league_code VARCHAR)"))
        conn.execute(text("CREATE TABLE fixtures (id INTEGER PRIMARY KEY, league_code VARCHAR)"))

    assert db.ensure_espn_event_id_columns(engine) == ["matches", "fixtures"]
    for table in ("matches", "fixtures"):
        assert "espn_event_id" in {c["name"] for c in inspect(engine).get_columns(table)}
    assert db.ensure_espn_event_id_columns(engine) == []


def test_event_id_is_stored_looked_up_and_never_cleared():
    day = dt.date(2026, 10, 4)
    with _session() as s:
        home, away = get_or_create_team(s, "Azerbaijan", "NL"), get_or_create_team(s, "Lithuania", "NL")
        upsert_fixture(s, "NL", day, home.id, away.id, "SCHEDULED", espn_event_id="42")
        assert espn_event_id_for_match(s, "NL", day, home.id, away.id) == "42"

        # Once played the result row carries it; a later sync with no id keeps it.
        upsert_match(s, "NL", "2627", day, home.id, away.id, 1, 0, source="espn", espn_event_id="42")
        upsert_match(s, "NL", "2627", day, home.id, away.id, 1, 0, source="espn")
        upsert_fixture(s, "NL", day, home.id, away.id, "SCHEDULED")
        assert espn_event_id_for_match(s, "NL", day, home.id, away.id) == "42"
        assert espn_event_id_for_match(s, "NL", day, away.id, home.id) is None


def test_espn_syncs_persist_the_event_id(monkeypatch):
    today = dt.date.today()
    with _session() as s:
        for name, espn_id in (("Azerbaijan", 581), ("Lithuania", 460)):
            update_espn_team_id(s, get_or_create_team(s, name, "NL").id, espn_id)
        s.commit()
        teams = {t: get_or_create_team(s, t, "NL").id for t in ("Azerbaijan", "Lithuania")}

        upcoming = _espn_match("111")
        upcoming.date, upcoming.completed, upcoming.home_score, upcoming.away_score = today, False, None, None
        monkeypatch.setattr(espn_client, "fetch_day_fixtures", lambda slug, day, *a, **k: [upcoming] if day == today else [])
        sync_fixtures_to_db_espn(s, LEAGUE, window_days=1)
        assert espn_event_id_for_match(s, "NL", today, teams["Azerbaijan"], teams["Lithuania"]) == "111"

        played = _espn_match("222")
        monkeypatch.setattr(espn_client, "fetch_team_results", lambda *a, **k: [played])
        sync_results_to_db_espn(s, LEAGUE, "2627")
        assert espn_event_id_for_match(s, "NL", played.date, teams["Azerbaijan"], teams["Lithuania"]) == "222"


# --- page helpers ----------------------------------------------------------------------------


def test_frames_render_lineups_stats_and_timeline():
    detail = espn_client.parse_match_detail("1", _summary())

    lineup = view.lineup_frame([p for p in detail.home.players if p.starter])
    assert list(lineup.columns) == ["#", "Player", "Pos", "Match"]
    assert lineup.iloc[1].to_dict() == {
        "#": "9",
        "Player": "Abdulakh Khaybulaev",
        "Pos": "F",
        "Match": "1 goal, 1 yellow, subbed off",
    }
    assert lineup.iloc[0]["Match"] == "2 saves"

    stats = view.team_stats_frame(detail)
    assert stats.to_dict("records")[0] == {"Azerbaijan": "41.9", "Stat": "Possession", "Lithuania": "58.1"}

    timeline = view.timeline_frame(detail)
    assert timeline.iloc[0].to_dict() == {"Min": "25'", "Team": "Azerbaijan", "Event": "⚽ Goal! Azerbaijan 1, Lithuania 0."}
    assert timeline.iloc[1]["Event"].startswith("🟨")


def test_state_hint_follows_the_match_date():
    today = dt.datetime.now(dt.UTC).date()
    assert view._state_hint(today + dt.timedelta(days=3)) == "pre"
    assert view._state_hint(today) == "in"
    assert view._state_hint(today - dt.timedelta(days=1)) == "post"


def test_match_detail_query_params_round_trip():
    assert view.match_detail_query_params("NL", 5, 9, dt.date(2026, 10, 4)) == {
        "league": "NL",
        "home": "5",
        "away": "9",
        "date": "2026-10-04",
    }


def test_match_tables_are_plain_page_tables_that_keep_all_their_text():
    from streamlit.testing.v1 import AppTest

    def app():
        import pandas as pd

        from soccer_predictor.dashboard.views import match_detail

        long_text = "3 shots on target, 1 goal, 2 assists, 4 fouls, 1 yellow card, subbed off 67'"
        frame = pd.DataFrame([{"#": 9, "Player": "A. Striker", "Pos": "F", "Match": long_text}])
        match_detail.render_table(frame)  # a DataFrame
        match_detail.render_table(frame.style.apply(lambda _: frame.map(lambda v: ""), axis=None))  # and a Styler

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception
    # Not st.dataframe (a canvas that clips long cells and can't be scrolled on a phone).
    assert len(at.dataframe) == 0 and len(at.table) == 2
    assert all("3 shots on target, 1 goal" in t.value["Match"].iloc[0] for t in at.table)


def test_match_tables_show_feed_text_as_text_not_markdown():
    import pandas as pd

    hostile = "[free money](http://evil.example) ![x](http://evil.example/p.png) **bold**"
    frame = pd.DataFrame([{"#": 7, "Player": hostile, "Pos": None, "Match": "1 goal"}])
    shown = view.table_styler(frame).to_html()  # the text st.table is given to draw
    # Every markdown character is escaped, "#" (a heading in markdown) survives as a column title,
    # a missing value is blank rather than "None", and the index column is gone.
    assert "\\[free money\\]\\(http" in shown and "\\!\\[x\\]" in shown and "\\*\\*bold\\*\\*" in shown
    assert ">\\#<" in shown and "None" not in shown and "nan" not in shown
    assert shown.count("<th ") == 4  # the four column titles, no index
    # Short columns can be kept on one line ("CD-L" must not break after the hyphen).
    one_line = view.table_styler(frame, one_line=("#", "Pos")).to_html()
    assert one_line.count("white-space: nowrap") >= 1 and "white-space" not in shown
    # A Styler (the team stats colouring) keeps its colours and has its cells escaped too.
    coloured = frame.style.apply(lambda _: pd.DataFrame("color: red", index=frame.index, columns=frame.columns), axis=None)
    assert "color: red" in view.table_styler(coloured).to_html() and "\\[free money\\]" in view.table_styler(coloured).to_html()


def test_team_stats_titles_are_escaped_because_they_are_team_names_from_espn():
    detail = espn_client.parse_match_detail("1", _summary())
    detail.home.name = "St. Pauli [live](http://evil.example)"
    html = view.table_styler(view.team_stats_styled(detail)).to_html()
    assert "\\[live\\]\\(http" in html and "[live](" not in html
    assert "color: #2ea043" in html  # still coloured


def test_compare_stat_marks_the_stronger_side_and_flips_for_fouls_and_cards():
    assert view.compare_stat("possessionPct", "58.1", "41.9") == ("better", "worse")
    assert view.compare_stat("totalShots", "5", "9") == ("worse", "better")
    assert view.compare_stat("wonCorners", "4", "4") == ("equal", "equal")
    assert view.compare_stat("foulsCommitted", "14", "9") == ("worse", "better")  # fewer fouls is better
    assert view.compare_stat("yellowCards", "0", "2") == ("better", "worse")
    assert view.compare_stat("saves", "5", "1") == (None, None)  # says nothing about who is stronger
    assert view.compare_stat("totalShots", "", "9") == (None, None)


def test_team_stats_table_colours_each_teams_number():
    detail = espn_client.parse_match_detail("1", _summary())
    html = view.team_stats_styled(detail).to_html()

    def cells_shaded(colour: str) -> set[str]:
        # Pandas writes one CSS rule per distinct style listing every cell that has it.
        rule = next(r for r in html.split("}") if colour in r)
        return set(re.findall(r"row\d+_col\d+", rule))

    # Rows: possession (Azerbaijan 41.9 v Lithuania 58.1), shots (9 v 5); columns 0 = home, 2 = away.
    assert cells_shaded("color: #2ea043") == {"row0_col2", "row1_col0"}  # the stronger side of each
    assert cells_shaded("color: #f85149") == {"row0_col0", "row1_col2"}


def test_substitution_minutes_come_from_the_players_plays_and_show_only_for_those_players():
    plays = [
        {"clock": {"displayValue": "25'"}, "scoringPlay": True, "substitution": False},
        {"clock": {"displayValue": "64'"}, "substitution": True},
    ]
    entries = [
        {"athlete": {"displayName": "Starter Off"}, "starter": True, "subbedOut": True, "plays": plays, "jersey": "9"},
        {"athlete": {"displayName": "Sub On"}, "starter": False, "subbedIn": True, "plays": [{"clock": {"displayValue": "90'+1'"}, "substitution": True}], "jersey": "20"},
        {"athlete": {"displayName": "Played All"}, "starter": True, "plays": [{"clock": {"displayValue": "40'"}, "substitution": False}], "jersey": "1"},
    ]
    off, on, full = espn_client._parse_lineup_players(entries)
    assert (off.sub_minute, on.sub_minute, full.sub_minute) == ("64'", "90'+1'", None)
    assert view.player_events_text(off) == "subbed off 64'"
    assert view.player_events_text(on) == "came on 90'+1'"
    assert view.player_events_text(full) == ""  # nothing extra for a player who played throughout


def test_a_link_from_a_team_page_carries_the_team_so_back_can_return_to_it():
    day = dt.date(2026, 10, 4)
    assert view.match_detail_query_params("NL", 5, 9, day, from_team_id=441, season="2627") == {
        "league": "NL", "home": "5", "away": "9", "date": "2026-10-04", "team": "441", "season": "2627",
    }
    # From the Leagues page (no team) the URL is unchanged, and Back goes to the league.
    assert "team" not in view.match_detail_query_params("NL", 5, 9, day)
    assert "season" not in view.match_detail_query_params("NL", 5, 9, day, from_team_id=441)


def _match_page_with_back(monkeypatch, team_param):
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    teams = {
        441: SimpleNamespace(canonical_name="Switzerland", league_code="NL"),
        1: SimpleNamespace(canonical_name="Arsenal", league_code="EPL"),
    }
    monkeypatch.setattr(view, "team_by_id", lambda session, team_id: teams.get(team_id))
    monkeypatch.setattr(view, "resolve_match", lambda session, league: None)  # the page stops after the Back button

    def app(team_param):
        import streamlit as st

        from soccer_predictor.dashboard.views import match_detail

        if team_param:
            st.query_params["team"] = team_param
        match_detail.render()

    at = AppTest.from_function(app, kwargs={"team_param": team_param}, default_timeout=60)
    at.query_params["league"] = "NL"
    return at.run()


def test_the_back_button_names_the_team_page_when_the_visit_came_from_one(monkeypatch):
    at = _match_page_with_back(monkeypatch, "441")
    assert not at.exception
    labels = [b.label for b in at.button]
    assert "← Back to Switzerland" in labels and not any("UEFA Nations League" in label for label in labels)


def test_the_back_button_goes_to_the_league_when_there_is_no_known_team(monkeypatch):
    for team in (None, "999"):  # none; an unknown team
        labels = [b.label for b in _match_page_with_back(monkeypatch, team).button]
        assert "← Back to UEFA Nations League" in labels, team


def test_the_back_button_returns_to_a_team_from_another_competition(monkeypatch):
    # A La Liga club's Champions League match: the team (EPL here) is not in the match's league.
    labels = [b.label for b in _match_page_with_back(monkeypatch, "1").button]
    assert "← Back to Arsenal" in labels


def test_a_players_plays_give_the_minute_of_each_goal_assist_card_and_substitution():
    def play(minute, *flags):
        return {"clock": {"displayValue": minute}, **{flag: True for flag in flags}}

    entries = [
        {
            "athlete": {"displayName": "Busy Player"}, "starter": True, "jersey": "9",
            "plays": [
                play("23'", "scoringPlay", "didScore"),
                play("45'", "scoringPlay", "didScore", "penaltyKick"),
                play("61'", "scoringPlay", "didAssist"),
                play("70'", "yellowCard"),
                play("90'+2'", "redCard"),
                play("90'+3'", "substitution"),
                play("30'", "scoringPlay", "didScore", "ownGoal"),  # an own goal is not one of his goals
                play("12'", "penaltyKick"),  # a missed penalty is nothing to mark
                {"clock": {}, "didScore": True},  # no minute, nothing to show
            ],
        },
        {"athlete": {"displayName": "Quiet Player"}, "starter": True, "jersey": "1", "plays": []},
    ]
    busy, quiet = espn_client._parse_lineup_players(entries)
    assert busy.event_minutes == {
        "goal": ["23'", "45'"], "assist": ["61'"], "yellow": ["70'"], "red": ["90'+2'"], "sub": ["90'+3'"],
    }
    assert quiet.event_minutes == {}
