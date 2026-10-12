"""The AFC Asian Cup as an ESPN group competition: config, the stage-aware group
rule (a knockout rematch of two group-mates must not count in a group table),
edition participants, and group-label cleanup."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League, load_leagues
from soccer_predictor.ingest import espn_client
from soccer_predictor.ingest.fixtures import (
    _group_for_match,
    _is_group_stage_name,
    seed_edition_participants,
    sync_recent_results_to_db_espn,
    sync_results_to_db_espn,
)
from soccer_predictor.storage.models import Base, Match, Team
from soccer_predictor.storage.repository import get_or_create_team, matches_for_league, team_by_espn_id, update_espn_team_id

ASIA = League(
    code="ASIA", name="AFC Asian Cup", seasons=["2015", "2019"], data_source="espn",
    espn_league_slug="afc.asian.cup", has_groups=True, season_display="single_year",
)


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def _match(home, away, day, stage=None, group_name=None, scores=(2, 1), completed=True, hn=None, an=None):
    return espn_client.EspnMatch(
        espn_home_id=home, espn_away_id=away, home_name=hn or f"Team {home}", away_name=an or f"Team {away}",
        date=dt.date(2015, 1, day), kickoff_utc=dt.datetime(2015, 1, day, 12, 0), completed=completed,
        home_score=scores[0], away_score=scores[1], stage=stage, group_name=group_name,
    )


# --- config ------------------------------------------------------------------------------


def test_the_asian_cup_is_configured_like_the_other_espn_group_competitions():
    asia = load_leagues()["ASIA"]
    assert (asia.data_source, asia.espn_league_slug) == ("espn", "afc.asian.cup")
    assert asia.has_groups and asia.trainable and asia.supports_predictions
    assert asia.api_season_year("2027") == 2027  # editions are keyed by year, not by season span
    assert asia.seasons == ["2007", "2011", "2015", "2019", "2023", "2027"]
    assert asia.news_query == "AFC Asian Cup"


# --- the stage-aware group rule ---------------------------------------------------------------


def test_a_knockout_rematch_of_two_group_mates_is_not_a_group_match():
    # The 2015 final: Australia v South Korea, both from Group A.
    membership = {"628": "Group A", "902": "Group A"}
    final = _match("628", "902", 31, stage="Final")
    assert _group_for_match(final, membership) is None


def test_the_same_pairing_in_the_group_stage_is_a_group_match():
    membership = {"628": "Group A", "902": "Group A"}
    group_game = _match("628", "902", 17, stage="2015 Group Stage")
    assert _group_for_match(group_game, membership) == "Group A"


def test_a_match_with_no_known_stage_still_uses_the_shared_group_rule():
    membership = {"1": "Group B", "2": "Group B"}
    assert _group_for_match(_match("1", "2", 5), membership) == "Group B"


def test_a_label_from_the_scoreboard_is_trusted():
    assert _group_for_match(_match("1", "2", 5, group_name="Group C"), {}) == "Group C"


def test_require_label_ignores_the_shared_group_rule():
    membership = {"1": "Group B", "2": "Group B"}
    assert _group_for_match(_match("1", "2", 5), membership, require_label=True) is None
    assert _group_for_match(_match("1", "2", 5, group_name="Group B"), membership, require_label=True) == "Group B"


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("Group Stage", True), ("2015 Group Stage", True), ("2007 Group Stage", True),
        ("League Phase", True), ("League A", True),
        ("Round of 16", False), ("Quarterfinals", False), ("Quarter-finals", False), ("Semi-finals", False),
        ("Third Place", False), ("Final", False), (None, False),
    ],
)
def test_which_stage_names_count_as_group_play(stage, expected):
    assert _is_group_stage_name(stage) is expected


def test_the_group_final_rematch_is_kept_out_of_the_stored_group(monkeypatch):
    with _session() as s:
        for espn_id in (628, 902):
            update_espn_team_id(s, get_or_create_team(s, f"Team {espn_id}", "ASIA").id, espn_id)
        s.commit()
        group_game = _match("628", "902", 17, stage="2015 Group Stage")
        final = _match("902", "628", 31, stage="Final", scores=(0, 2))
        monkeypatch.setattr(espn_client, "fetch_team_results", lambda *a, **k: [group_game, final])
        monkeypatch.setattr(
            espn_client, "fetch_group_participants",
            lambda *a, **k: [espn_client.EspnParticipant("628", "Team 628", "Group A"), espn_client.EspnParticipant("902", "Team 902", "Group A")],
        )

        sync_results_to_db_espn(s, ASIA, "2015")
        s.commit()

        stored = {m.date.day: m.group_name for m in s.query(Match)}
        assert stored == {17: "Group A", 31: None}


def test_the_quick_refresh_does_not_label_a_scoreboard_knockout_as_a_group_match(monkeypatch):
    with _session() as s:
        for espn_id in (628, 902):
            update_espn_team_id(s, get_or_create_team(s, f"Team {espn_id}", "ASIA").id, espn_id)
        s.commit()
        today = dt.datetime.now(dt.UTC).date()
        final = _match("628", "902", 5, stage=None)  # a scoreboard event: no stage, no label
        final.date = today
        monkeypatch.setattr(espn_client, "fetch_group_membership", lambda *a, **k: {"628": "Group A", "902": "Group A"})
        monkeypatch.setattr(espn_client, "fetch_day_matches", lambda slug, day, cache_ttl_seconds=None: [final] if day == today else [])

        sync_recent_results_to_db_espn(s, ASIA, "2027")
        s.commit()

        assert matches_for_league(s, "ASIA").iloc[0]["group_name"] is None


# --- participants of past editions ---------------------------------------------------------------


def test_teams_that_only_played_an_older_edition_are_added_so_their_matches_are_found(monkeypatch):
    """Lebanon and Hong Kong aren't in the 2027 field, so seeding from today's team
    list alone would never reach their (Lebanon v Hong Kong) match."""
    with _session() as s:
        update_espn_team_id(s, get_or_create_team(s, "Japan", "ASIA").id, 627)  # a current team
        s.commit()
        seen = []

        def fake_results(slug, team_id, season):
            seen.append(team_id)
            if team_id == "1001":
                return [_match("1001", "1002", 12, stage="2015 Group Stage", hn="Lebanon", an="Hong Kong")]
            return []

        monkeypatch.setattr(espn_client, "fetch_team_results", fake_results)
        monkeypatch.setattr(
            espn_client, "fetch_group_participants",
            lambda *a, **k: [
                espn_client.EspnParticipant("627", "Japan", "Group A"),
                espn_client.EspnParticipant("1001", "Lebanon", "Group B", "https://logos.example/lb.png"),
                espn_client.EspnParticipant("1002", "Hong Kong", "Group B"),
            ],
        )

        sync_results_to_db_espn(s, ASIA, "2015")
        s.commit()

        assert {"1001", "1002"} <= set(seen)
        stored = matches_for_league(s, "ASIA")
        assert len(stored) == 1 and stored.iloc[0]["group_name"] == "Group B"
        assert s.query(Team).filter_by(canonical_name="Lebanon").one().crest_url == "https://logos.example/lb.png"


def test_seeding_participants_does_not_duplicate_existing_teams():
    with _session() as s:
        update_espn_team_id(s, get_or_create_team(s, "Japan", "ASIA").id, 627)
        s.commit()
        participants = [espn_client.EspnParticipant("627", "Japan", "A"), espn_client.EspnParticipant("5", "Oman", "A")]

        assert seed_edition_participants(s, ASIA, participants) == 1
        assert seed_edition_participants(s, ASIA, participants) == 0
        assert team_by_espn_id(s, "ASIA", 5).canonical_name == "Oman"
        assert s.query(Team).filter_by(league_code="ASIA").count() == 2


# --- label cleanup -----------------------------------------------------------------------------------


def test_the_2019_asian_cups_oddly_prefixed_last_group_reads_like_its_siblings():
    assert espn_client.normalize_group_name("AFC ASIAN CUP - GROUP F") == "Group F"
    assert espn_client.normalize_group_name("GROUP E") == "Group E"
    assert espn_client.normalize_group_name("LEAGUE D - GROUP 2") == "League D - Group 2"  # untouched


def test_participants_carry_names_groups_and_logos(monkeypatch):
    response = {
        "children": [
            {"name": "GROUP A", "standings": {"entries": [
                {"team": {"id": "628", "displayName": "Australia", "logos": [{"href": "https://l.example/au.png"}]}},
                {"team": {"id": "902"}},
            ]}},
            {"name": "AFC ASIAN CUP - GROUP F", "standings": {"entries": [{"team": {"id": "5", "displayName": "Oman"}}]}},
        ]
    }
    monkeypatch.setattr(espn_client, "get", lambda *a, **k: response)

    people = {p.espn_id: p for p in espn_client.fetch_group_participants("afc.asian.cup", 2019)}

    assert (people["628"].name, people["628"].group, people["628"].logo_url) == ("Australia", "Group A", "https://l.example/au.png")
    assert people["902"].name == "team#902" and people["902"].logo_url is None
    assert people["5"].group == "Group F"


# --- a team the model has never seen, and placeholder teams ----------------------------------------------------


def test_a_debutant_with_no_history_is_rated_as_an_average_team_not_a_crash():
    from soccer_predictor.model.dixon_coles import DixonColesParams

    params = DixonColesParams(
        league_code="ASIA", attack={1: 1.5, 2: 0.5}, defense={1: 0.8, 2: 1.2},
        home_advantage=1.2, rho=0.0, xi=0.0002, fitted_at="x", n_matches=10,
    )
    assert params.attack[99] == pytest.approx(1.0) and params.defense[99] == pytest.approx(1.0)
    assert 99 not in params.attack and set(params.attack) == {1, 2}  # membership still means "actually fitted"
    assert params.attack[1] == 1.5


def test_the_average_fallback_survives_a_save_and_load():
    from soccer_predictor.model.dixon_coles import DixonColesParams

    params = DixonColesParams("ASIA", {1: 2.0, 2: 1.0}, {1: 1.0, 2: 3.0}, 1.2, 0.0, 0.0002, "x", 5)
    loaded = DixonColesParams.from_json(params.to_json())
    assert loaded.attack[7] == pytest.approx(1.5) and loaded.defense[7] == pytest.approx(2.0)


def test_predicting_a_fixture_with_an_unseen_team_works(monkeypatch):
    import datetime as dt

    from soccer_predictor.model.dixon_coles import DixonColesParams
    from soccer_predictor.prediction.service import predict_fixture

    monkeypatch.setattr("soccer_predictor.prediction.service.resolve_current_attack_strength", lambda *a, **k: None)
    with _session() as s:
        japan = get_or_create_team(s, "Japan", "ASIA").id
        singapore = get_or_create_team(s, "Singapore", "ASIA").id
        s.commit()
        params = DixonColesParams("ASIA", {japan: 1.4}, {japan: 0.7}, 1.2, 0.0, 0.0002, "x", 20)

        prediction = predict_fixture(s, params, japan, singapore, "Japan", "Singapore", fixture_date=dt.date(2027, 1, 8))

        assert prediction.home_win + prediction.draw + prediction.away_win == pytest.approx(1.0)
        assert prediction.home_win > prediction.away_win  # Japan are the better side


def test_a_tbd_placeholder_is_never_stored_as_a_team():
    with _session() as s:
        added = seed_edition_participants(
            s, ASIA, [espn_client.EspnParticipant("9210", "TBD", "Group E"), espn_client.EspnParticipant("5", "Oman", "Group E")]
        )
        assert added == 1
        assert {t.canonical_name for t in s.query(Team).filter_by(league_code="ASIA")} == {"Oman"}


def test_a_match_against_a_tbd_placeholder_is_skipped(monkeypatch):
    with _session() as s:
        update_espn_team_id(s, get_or_create_team(s, "Japan", "ASIA").id, 627)
        s.commit()
        fixture = _match("627", "9210", 9, completed=False, an="TBD")
        monkeypatch.setattr(espn_client, "fetch_day_fixtures", lambda slug, day, **k: [fixture] if day == dt.date.today() else [])
        monkeypatch.setattr(espn_client, "fetch_group_membership", lambda *a, **k: {})

        from soccer_predictor.ingest.fixtures import sync_fixtures_to_db_espn

        synced, skipped = sync_fixtures_to_db_espn(s, ASIA)

        assert (synced, skipped) == (0, 1)
        assert s.query(Team).filter_by(canonical_name="TBD").count() == 0


# --- the whole schedule is fetched once, months before kickoff ----------------------------------------------------


def test_the_whole_years_fixtures_are_stored_even_though_the_day_sweep_cannot_see_them(monkeypatch):
    from soccer_predictor.ingest.fixtures import sync_fixtures_to_db_espn
    from soccer_predictor.storage.repository import fixture_groups_for_league

    year = dt.date.today().year + 1
    opener = espn_client.EspnMatch(
        espn_home_id="1", espn_away_id="2", home_name="Saudi Arabia", away_name="Palestine",
        date=dt.date(year, 1, 7), kickoff_utc=dt.datetime(year, 1, 7, 17, 0), completed=False,
        home_score=None, away_score=None, group_name="Group A",
    )
    asked_years = []
    monkeypatch.setattr(espn_client, "fetch_day_fixtures", lambda *a, **k: [])  # nothing within 14 days
    monkeypatch.setattr(espn_client, "fetch_group_membership", lambda *a, **k: {})
    monkeypatch.setattr(
        espn_client, "fetch_year_fixtures", lambda slug, y, **k: asked_years.append(y) or ([opener] if y == year else [])
    )
    with _session() as s:
        for espn_id in (1, 2):
            update_espn_team_id(s, get_or_create_team(s, f"Team {espn_id}", "ASIA").id, espn_id)
        s.commit()

        synced, skipped = sync_fixtures_to_db_espn(s, ASIA)
        s.commit()

        assert (synced, skipped) == (1, 0)
        assert fixture_groups_for_league(s, "ASIA")["group_name"].tolist() == ["Group A"]
    assert asked_years == [dt.date.today().year, dt.date.today().year + 1]


def test_a_league_without_groups_does_not_make_the_year_request(monkeypatch):
    from soccer_predictor.ingest.fixtures import sync_fixtures_to_db_espn

    def fail(*a, **k):
        raise AssertionError("only group competitions fetch a whole year of fixtures")

    monkeypatch.setattr(espn_client, "fetch_year_fixtures", fail)
    monkeypatch.setattr(espn_client, "fetch_day_fixtures", lambda *a, **k: [])
    mls = League(code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1")
    with _session() as s:
        assert sync_fixtures_to_db_espn(s, mls) == (0, 0)


@pytest.mark.real_year_fetch
def test_the_year_fetch_asks_for_dates_equals_the_year_and_drops_finished_matches(monkeypatch):
    calls = []

    def event(completed, eid):
        return {
            "date": "2027-01-07T17:00Z",
            "competitions": [{
                "status": {"type": {"completed": completed, "state": "post" if completed else "pre"}},
                "competitors": [
                    {"homeAway": "home", "team": {"id": "1", "displayName": "A"}, "score": "1"},
                    {"homeAway": "away", "team": {"id": "2", "displayName": "B"}, "score": "0"},
                ],
            }],
        }

    monkeypatch.setattr(
        espn_client, "get", lambda path, params=None, **k: calls.append((path, params)) or {"events": [event(False, 1), event(True, 2)]}
    )

    matches = espn_client.fetch_year_fixtures("afc.asian.cup", 2027)

    assert calls == [("/afc.asian.cup/scoreboard", {"dates": "2027"})]
    assert len(matches) == 1 and not matches[0].completed


@pytest.mark.parametrize(
    "name",
    [
        "TBD", "tba", "To Be Determined", "Group A Winner", "Group C 2nd Place", "3rd Place Group A/C/D",
        "3rd Place Group C/D/E", "Round of 16 1 Winner", "Quarterfinal 4 Winner", "Semifinal 2 Loser", "Group B Runner-up",
    ],
)
def test_espns_undecided_knockout_slots_are_recognised_as_placeholders(name):
    from soccer_predictor.ingest.fixtures import _is_placeholder_team

    assert _is_placeholder_team(name)


@pytest.mark.parametrize(
    "name",
    ["Saudi Arabia", "Australia", "Kyrgyz Republic", "United Arab Emirates", "Türkiye", "Republic of Ireland",
     "Bosnia-Herzegovina", "North Korea", "Czechia", "Winnersh Rovers FC", None, ""],
)
def test_real_teams_are_never_mistaken_for_placeholders(name):
    from soccer_predictor.ingest.fixtures import _is_placeholder_team

    assert not _is_placeholder_team(name)


def test_knockout_fixtures_between_placeholders_are_skipped_not_stored_or_predicted(monkeypatch):
    from soccer_predictor.ingest.fixtures import sync_fixtures_to_db_espn

    knockout = _match("5926", "5930", 22, completed=False, hn="Group A 2nd Place", an="Group C 2nd Place")
    half_known = _match("627", "5931", 23, completed=False, hn="Japan", an="3rd Place Group A/C/D")
    monkeypatch.setattr(espn_client, "fetch_day_fixtures", lambda slug, day, **k: [knockout, half_known] if day == dt.date.today() else [])
    monkeypatch.setattr(espn_client, "fetch_group_membership", lambda *a, **k: {})
    with _session() as s:
        update_espn_team_id(s, get_or_create_team(s, "Japan", "ASIA").id, 627)
        s.commit()

        assert sync_fixtures_to_db_espn(s, ASIA) == (0, 2)
        assert {t.canonical_name for t in s.query(Team)} == {"Japan"}


def test_next_and_search_are_only_filled_for_the_latest_season():
    """Regression: the 2023 Asian Cup table showed 'Qatar - next: Thailand, Jan 11'
    because stored fixtures are the current (2027) edition's and carry no season."""
    import inspect

    from soccer_predictor.dashboard.views import home

    source = inspect.getsource(home.render)
    assert "if season == league.seasons[-1]:" in source
    assert "next_opponent_names, match_links = {}, {}" in source
