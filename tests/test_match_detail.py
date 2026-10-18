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


def test_tables_are_tall_enough_to_show_every_row_up_to_a_cap():
    assert view.full_height(0) == 38
    assert view.full_height(17) == 35 * 18 + 3  # the whole team-stats table, no inner scrolling
    assert view.full_height(500) == view.MAX_TABLE_HEIGHT_PX


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
