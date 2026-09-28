from __future__ import annotations

import datetime as dt

import requests

from soccer_predictor.ingest import espn_client


def _event(
    home_id="183",
    home_name="Columbus Crew",
    away_id="20232",
    away_name="Inter Miami CF",
    date="2026-09-27T23:00Z",
    completed=True,
    home_score="2",
    away_score="1",
):
    return {
        "date": date,
        "competitions": [
            {
                "status": {"type": {"completed": completed}},
                "competitors": [
                    {"homeAway": "home", "team": {"id": home_id, "displayName": home_name}, "score": home_score},
                    {"homeAway": "away", "team": {"id": away_id, "displayName": away_name}, "score": away_score},
                ],
            }
        ],
    }


TEAMS_RESPONSE = {
    "sports": [
        {
            "leagues": [
                {
                    "teams": [
                        {
                            "team": {
                                "id": "183",
                                "displayName": "Columbus Crew",
                                "logos": [{"href": "https://example.com/crew.png"}],
                            }
                        },
                        {"team": {"id": "20232", "displayName": "Inter Miami CF", "logos": []}},
                    ]
                }
            ]
        }
    ]
}

ROSTER_RESPONSE = {
    "team": {
        "athletes": [
            {"id": "1", "displayName": "Evan Bush", "position": {"abbreviation": "G"}, "injuries": []},
            {
                "id": "2",
                "displayName": "Bukayo Saka",
                "position": {"abbreviation": "F"},
                "injuries": [{"status": "Out", "type": {"description": "Hamstring"}}],
            },
        ]
    }
}


def test_fetch_teams_parses_team_list(monkeypatch):
    monkeypatch.setattr(espn_client, "get", lambda *a, **k: TEAMS_RESPONSE)

    teams = espn_client.fetch_teams("usa.1")

    assert [t.display_name for t in teams] == ["Columbus Crew", "Inter Miami CF"]
    assert teams[0].espn_id == "183"
    assert teams[0].logo_url == "https://example.com/crew.png"
    assert teams[1].logo_url is None


def test_fetch_teams_returns_empty_on_request_exception(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(espn_client, "get", raise_error)
    assert espn_client.fetch_teams("usa.1") == []


def test_fetch_teams_returns_empty_on_malformed_response(monkeypatch):
    monkeypatch.setattr(espn_client, "get", lambda *a, **k: {"unexpected": "shape"})
    assert espn_client.fetch_teams("usa.1") == []


def test_fetch_team_results_keeps_only_completed_with_string_score(monkeypatch):
    events = [_event(completed=True, home_score="2", away_score="1"), _event(completed=False)]
    monkeypatch.setattr(espn_client, "get", lambda *a, **k: {"events": events})

    results = espn_client.fetch_team_results("usa.1", "183", 2026)

    assert len(results) == 1
    assert results[0].home_score == 2
    assert results[0].away_score == 1
    assert results[0].date == dt.date(2026, 9, 27)


def test_fetch_team_results_handles_nested_score_object(monkeypatch):
    # The schedule endpoint's real shape (unlike the scoreboard's plain
    # string) -- {"value": 2.0, ...} -- verified live against real MLS data.
    events = [_event(home_score={"value": 2.0}, away_score={"value": 1.0})]
    monkeypatch.setattr(espn_client, "get", lambda *a, **k: {"events": events})

    results = espn_client.fetch_team_results("usa.1", "183", 2026)

    assert results[0].home_score == 2
    assert results[0].away_score == 1


def test_fetch_team_results_returns_empty_on_request_exception(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(espn_client, "get", raise_error)
    assert espn_client.fetch_team_results("usa.1", "183", 2026) == []


def test_fetch_day_fixtures_keeps_only_upcoming(monkeypatch):
    events = [_event(completed=False, home_score=None, away_score=None), _event(completed=True)]
    monkeypatch.setattr(espn_client, "get", lambda *a, **k: {"events": events})

    fixtures = espn_client.fetch_day_fixtures("usa.1", dt.date(2026, 10, 10))

    assert len(fixtures) == 1
    assert fixtures[0].completed is False
    assert fixtures[0].home_score is None


def test_fetch_day_fixtures_returns_empty_on_request_exception(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(espn_client, "get", raise_error)
    assert espn_client.fetch_day_fixtures("usa.1", dt.date(2026, 10, 10)) == []


def test_fetch_team_roster_parses_players_and_injuries(monkeypatch):
    monkeypatch.setattr(espn_client, "get", lambda *a, **k: ROSTER_RESPONSE)

    roster, injuries = espn_client.fetch_team_roster("usa.1", "183")

    assert [p.name for p in roster] == ["Evan Bush", "Bukayo Saka"]
    assert roster[0].position == "Goalkeeper"
    assert roster[0].position_bucket == "defense"
    assert roster[1].position == "Offence"
    assert roster[1].position_bucket == "attack"

    assert len(injuries) == 1
    assert injuries[0].player_name == "Bukayo Saka"
    assert injuries[0].position_bucket == "attack"
    assert injuries[0].note == "Out"


def test_fetch_team_roster_returns_empty_on_request_exception(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(espn_client, "get", raise_error)
    roster, injuries = espn_client.fetch_team_roster("usa.1", "183")
    assert roster == []
    assert injuries == []


def test_fetch_team_conference_maps_eastern(monkeypatch):
    monkeypatch.setattr(espn_client, "get", lambda *a, **k: {"team": {"groups": {"id": "1"}}})
    assert espn_client.fetch_team_conference("usa.1", "183") == "Eastern Conference"


def test_fetch_team_conference_maps_western(monkeypatch):
    monkeypatch.setattr(espn_client, "get", lambda *a, **k: {"team": {"groups": {"id": "2"}}})
    assert espn_client.fetch_team_conference("usa.1", "187") == "Western Conference"


def test_fetch_team_conference_returns_none_for_unknown_group(monkeypatch):
    monkeypatch.setattr(espn_client, "get", lambda *a, **k: {"team": {"groups": {"id": "99"}}})
    assert espn_client.fetch_team_conference("usa.1", "1") is None


def test_fetch_team_conference_returns_none_on_request_exception(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(espn_client, "get", raise_error)
    assert espn_client.fetch_team_conference("usa.1", "183") is None


def test_get_serves_from_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(espn_client, "CACHE_DIR", tmp_path)
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def fake_get(*args, **kwargs):
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(espn_client.requests, "get", fake_get)

    first = espn_client.get("/usa.1/teams", params={"limit": 50})
    second = espn_client.get("/usa.1/teams", params={"limit": 50})

    assert first == {"ok": True}
    assert second == {"ok": True}
    assert len(calls) == 1  # second call served from disk cache
