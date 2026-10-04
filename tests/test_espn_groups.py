"""Group-competition support in the ESPN ingest (Nations League): group and
stage parsing, per-edition group membership, and how syncs tag matches."""

from __future__ import annotations

import datetime as dt

import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import espn_client
from soccer_predictor.ingest.fixtures import (
    derive_groups_from_matches,
    sync_fixtures_to_db_espn,
    sync_results_to_db_espn,
)
from soccer_predictor.storage.models import Base, Match, Team
from soccer_predictor.storage.repository import (
    fixture_groups_for_league,
    get_or_create_team,
    matches_for_league,
    team_by_espn_id,
    update_espn_team_id,
)

GROUP_LEAGUE = League(
    code="NL",
    name="UEFA Nations League",
    seasons=["2425"],
    data_source="espn",
    espn_league_slug="uefa.nations",
    has_groups=True,
)
FLAT_LEAGUE = League(
    code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1"
)


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def _event(group=None, stage=None):
    event = {
        "date": "2024-09-05T18:45Z",
        "competitions": [
            {
                "status": {"type": {"completed": True, "state": "post"}},
                "competitors": [
                    {"homeAway": "home", "team": {"id": "1", "displayName": "Spain"}, "score": "2"},
                    {"homeAway": "away", "team": {"id": "2", "displayName": "Croatia"}, "score": "0"},
                ],
            }
        ],
    }
    if group:
        event["competitions"][0]["group"] = {"groupId": "13", "name": group}
    if stage:
        event["seasonType"] = {"name": stage}
    return event


def _match(home_id, away_id, day, stage="Group Stage", group_name=None, home_name=None, away_name=None, completed=True):
    return espn_client.EspnMatch(
        espn_home_id=home_id,
        espn_away_id=away_id,
        home_name=home_name or f"Team {home_id}",
        away_name=away_name or f"Team {away_id}",
        date=dt.date(2024, 9, day),
        kickoff_utc=dt.datetime(2024, 9, day, 18, 45),
        completed=completed,
        home_score=1 if completed else None,
        away_score=0 if completed else None,
        stage=stage,
        group_name=group_name,
    )


def _seed(s, league_code, ids):
    for espn_id in ids:
        team = get_or_create_team(s, f"Team {espn_id}", league_code)
        update_espn_team_id(s, team.id, int(espn_id))
    s.commit()


def _stored_groups(s):
    espn_id_by_team = {t.id: t.espn_team_id for t in s.query(Team)}
    return {(espn_id_by_team[m.home_team_id], espn_id_by_team[m.away_team_id]): m.group_name for m in s.query(Match)}


# --- espn_client -------------------------------------------------------------


def test_parse_event_reads_group_and_stage():
    match = espn_client._parse_event(_event(group="GROUP D1", stage="Group Stage"))
    assert match.group_name == "Group D1"  # normalized casing
    assert match.stage == "Group Stage"


def test_parse_event_group_and_stage_default_to_none():
    match = espn_client._parse_event(_event())
    assert match.group_name is None
    assert match.stage is None


def test_normalize_group_name_handles_every_casing_espn_uses():
    assert espn_client.normalize_group_name("GROUP D1") == "Group D1"
    assert espn_client.normalize_group_name("Group B1") == "Group B1"
    assert espn_client.normalize_group_name("LEAGUE D - GROUP 2") == "League D - Group 2"
    assert espn_client.normalize_group_name("  ") is None
    assert espn_client.normalize_group_name(None) is None


STANDINGS_RESPONSE = {
    "children": [
        {"name": "GROUP A1", "standings": {"entries": [{"team": {"id": "1"}}, {"team": {"id": "2"}}]}},
        {"name": "Group A2", "standings": {"entries": [{"team": {"id": "3"}}]}},
    ]
}


def test_fetch_group_membership_maps_team_ids_to_groups(monkeypatch):
    calls = []

    def fake_get(path, params=None, cache_ttl_seconds=0, base_url=espn_client.BASE_URL):
        calls.append((path, params, base_url))
        return STANDINGS_RESPONSE

    monkeypatch.setattr(espn_client, "get", fake_get)

    membership = espn_client.fetch_group_membership("uefa.nations", 2024)

    assert membership == {"1": "Group A1", "2": "Group A1", "3": "Group A2"}
    # The standings endpoint lives under a different base path than the rest.
    assert calls == [("/uefa.nations/standings", {"season": 2024}, espn_client.STANDINGS_BASE_URL)]


def test_fetch_group_membership_is_empty_on_failure_or_a_flat_table(monkeypatch):
    def raising_get(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(espn_client, "get", raising_get)
    assert espn_client.fetch_group_membership("uefa.nations", 2024) == {}

    monkeypatch.setattr(espn_client, "get", lambda *a, **k: {"standings": {"entries": []}})
    assert espn_client.fetch_group_membership("usa.1", 2026) == {}


def test_cache_key_is_unchanged_for_the_default_base_url_and_distinct_for_standings():
    default_key = espn_client._cache_path("/usa.1/teams", {"limit": 50})
    assert default_key == espn_client._cache_path("/usa.1/teams", {"limit": 50}, espn_client.BASE_URL)
    assert default_key != espn_client._cache_path("/usa.1/teams", {"limit": 50}, espn_client.STANDINGS_BASE_URL)


# --- results sync --------------------------------------------------------------


def test_results_sync_tags_group_stage_matches_from_standings_membership(monkeypatch):
    with _session() as s:
        _seed(s, "NL", [1, 2, 3, 4])
        matches = [
            _match("1", "2", 5),  # same group -> tagged
            _match("3", "4", 6),
            _match("1", "3", 20, stage="Quarterfinals"),  # different groups -> knockout
        ]
        monkeypatch.setattr(espn_client, "fetch_team_results", lambda slug, team_id, season: matches)
        monkeypatch.setattr(
            espn_client,
            "fetch_group_membership",
            lambda slug, season: {"1": "Group A1", "2": "Group A1", "3": "Group A2", "4": "Group A2"},
        )

        sync_results_to_db_espn(s, GROUP_LEAGUE, "2425")
        s.commit()

        groups = _stored_groups(s)
        assert groups[(1, 2)] == "Group A1"
        assert groups[(3, 4)] == "Group A2"
        assert groups[(1, 3)] is None


def test_results_sync_never_looks_up_groups_for_a_league_without_has_groups(monkeypatch):
    # MLS: ESPN's standings "groups" are its conferences, which most matches
    # cross -- they must never become match groups.
    with _session() as s:
        _seed(s, "MLS", [183, 20232])
        match = _match("183", "20232", 5, home_name="Columbus Crew", away_name="Inter Miami CF")
        monkeypatch.setattr(espn_client, "fetch_team_results", lambda *a, **k: [match])

        def fail(*a, **k):
            raise AssertionError("must not look up groups for a league without has_groups")

        monkeypatch.setattr(espn_client, "fetch_group_membership", fail)

        sync_results_to_db_espn(s, FLAT_LEAGUE, "2026")
        s.commit()

        assert matches_for_league(s, "MLS").iloc[0]["group_name"] is None


def test_results_sync_adds_an_opponent_missing_from_the_current_team_list(monkeypatch):
    # e.g. Russia: in 2018-19's groups, but not in ESPN's current 54 teams.
    with _session() as s:
        _seed(s, "NL", [1])
        match = _match("1", "99", 5, home_name="Team 1", away_name="Russia")
        monkeypatch.setattr(espn_client, "fetch_team_results", lambda *a, **k: [match])
        monkeypatch.setattr(espn_client, "fetch_group_membership", lambda *a, **k: {})

        synced, skipped = sync_results_to_db_espn(s, GROUP_LEAGUE, "2425")
        s.commit()

        assert (synced, skipped) == (1, 0)
        assert len(matches_for_league(s, "NL")) == 1
        assert team_by_espn_id(s, "NL", 99).canonical_name == "Russia"


def test_results_sync_still_skips_an_unknown_opponent_in_a_league_without_groups(monkeypatch):
    with _session() as s:
        _seed(s, "MLS", [183])
        match = _match("183", "999", 5, home_name="Columbus Crew", away_name="Some Liga MX Club")
        monkeypatch.setattr(espn_client, "fetch_team_results", lambda *a, **k: [match])

        assert sync_results_to_db_espn(s, FLAT_LEAGUE, "2026") == (0, 1)


def test_results_sync_keeps_a_group_filled_in_earlier(monkeypatch):
    # A later sync from a source that doesn't know the group must not wipe it.
    with _session() as s:
        _seed(s, "NL", [1, 2])
        match = _match("1", "2", 5)
        monkeypatch.setattr(espn_client, "fetch_team_results", lambda *a, **k: [match])
        monkeypatch.setattr(espn_client, "fetch_group_membership", lambda *a, **k: {"1": "Group A1", "2": "Group A1"})
        sync_results_to_db_espn(s, GROUP_LEAGUE, "2425")
        s.commit()

        monkeypatch.setattr(espn_client, "fetch_group_membership", lambda *a, **k: {})
        sync_results_to_db_espn(s, GROUP_LEAGUE, "2425")
        s.commit()

        assert _stored_groups(s)[(1, 2)] == "Group A1"


# --- fixtures sync ---------------------------------------------------------------


def test_fixtures_sync_tags_groups_from_the_scoreboard_label(monkeypatch):
    with _session() as s:
        _seed(s, "NL", [1, 2])
        today = dt.date.today()
        fixture = _match("1", "2", 5, completed=False, group_name="Group A1")
        monkeypatch.setattr(espn_client, "fetch_day_fixtures", lambda slug, day: [fixture] if day == today else [])
        monkeypatch.setattr(espn_client, "fetch_group_membership", lambda *a, **k: {})

        sync_fixtures_to_db_espn(s, GROUP_LEAGUE)
        s.commit()

        assert fixture_groups_for_league(s, "NL")["group_name"].tolist() == ["Group A1"]


def test_fixtures_sync_falls_back_to_standings_membership(monkeypatch):
    with _session() as s:
        _seed(s, "NL", [1, 2])
        today = dt.date.today()
        fixture = _match("1", "2", 5, completed=False)  # no label on the scoreboard event
        monkeypatch.setattr(espn_client, "fetch_day_fixtures", lambda slug, day: [fixture] if day == today else [])
        monkeypatch.setattr(espn_client, "fetch_group_membership", lambda *a, **k: {"1": "Group B2", "2": "Group B2"})

        sync_fixtures_to_db_espn(s, GROUP_LEAGUE)
        s.commit()

        assert fixture_groups_for_league(s, "NL")["group_name"].tolist() == ["Group B2"]


# --- derive_groups_from_matches ----------------------------------------------------


def test_derive_groups_splits_a_stage_into_round_robin_components():
    # ESPN's 2018-19 data labels matches only with the division ("League A"):
    # two disjoint round-robins are two groups.
    matches = [
        _match("1", "2", 5, stage="League A", home_name="Spain", away_name="Croatia"),
        _match("2", "1", 9, stage="League A", home_name="Croatia", away_name="Spain"),
        _match("3", "4", 5, stage="League A", home_name="Germany", away_name="France"),
    ]
    derived = derive_groups_from_matches(matches, membership={})

    spain_croatia = derived[("1", "2", dt.date(2024, 9, 5))]
    assert derived[("2", "1", dt.date(2024, 9, 9))] == spain_croatia
    # Numbered by each group's first team alphabetically: Croatia < France.
    assert spain_croatia == "League A - Group 1"
    assert derived[("3", "4", dt.date(2024, 9, 5))] == "League A - Group 2"


def test_derive_groups_skips_knockouts_and_matches_already_in_a_group():
    matches = [
        _match("1", "2", 5, stage="Final"),
        _match("3", "4", 6, stage="League A"),
    ]
    derived = derive_groups_from_matches(matches, membership={"3": "Group A1", "4": "Group A1"})
    assert derived == {}
