"""Live banner -> match detail links, name-based ESPN lookup, URL resolution."""

from __future__ import annotations

import datetime as dt

import pytest
from types import SimpleNamespace
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.dashboard import components
from soccer_predictor.dashboard.views import match_detail as view
from soccer_predictor.ingest import espn_client, live_scores
from soccer_predictor.ingest.live_scores import LiveMatch
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team, upsert_match

EPL = League(code="EPL", name="English Premier League", seasons=["2627"], csv_code="E0", espn_league_slug="eng.1")
NL = League(code="NL", name="UEFA Nations League", seasons=["2627"], data_source="espn", espn_league_slug="uefa.nations")
LEAGUES = {"EPL": EPL, "NL": NL}


def _live(**kwargs) -> LiveMatch:
    base = dict(
        league_code="EPL",
        home_name="Arsenal",
        away_name="Chelsea",
        home_crest=None,
        away_crest=None,
        home_score=2,
        away_score=1,
        clock_label="63'",
    )
    return LiveMatch(**{**base, **kwargs})


# --- banner links ----------------------------------------------------------------------------


def test_espn_sourced_match_links_to_its_event_id_in_the_same_tab():
    markup = components.live_scores_banner_html([_live(league_code="NL", event_id="401861119")], LEAGUES)
    assert 'href="/match_detail?league=NL&amp;event=401861119"' in markup
    assert 'target="_self"' in markup and "google.com" not in markup


def test_football_data_match_links_by_names_and_date():
    match = _live(home_name="Brighton & Hove Albion", match_date=dt.date(2026, 10, 11))
    href = components.live_match_detail_href(match)
    assert href == (
        "/match_detail?league=EPL&home_name=Brighton%20%26%20Hove%20Albion&away_name=Chelsea&date=2026-10-11"
    )
    assert components.live_scores_banner_html([match], LEAGUES).count("/match_detail?") == 1


def test_a_match_with_nothing_to_look_it_up_by_falls_back_to_google():
    markup = components.live_scores_banner_html([_live()], LEAGUES)
    assert "google.com/search" in markup and 'target="_blank"' in markup and "/match_detail" not in markup


def test_hostile_names_are_neutralised_in_the_detail_link():
    match = _live(home_name='"><script>alert(1)</script>', match_date=dt.date(2026, 10, 11))
    markup = components.live_scores_banner_html([match], LEAGUES)
    assert "<script>" not in markup


def test_live_matches_carry_what_the_link_needs(monkeypatch):
    espn_match = espn_client.EspnMatch(
        espn_home_id="1",
        espn_away_id="2",
        home_name="A",
        away_name="B",
        date=dt.date(2026, 10, 11),
        kickoff_utc=dt.datetime(2026, 10, 11, 18, 0),
        completed=False,
        home_score=0,
        away_score=0,
        state="in",
        clock_label="5'",
        event_id="777",
    )
    monkeypatch.setattr(espn_client, "fetch_day_matches", lambda *a, **k: [espn_match])
    [live] = live_scores.fetch_live_matches_espn(NL, now=dt.datetime(2026, 10, 11, 18, 5))[:1]
    assert (live.event_id, live.match_date) == ("777", dt.date(2026, 10, 11))
    assert live_scores._utc_date("2026-10-11T19:00:00Z") == dt.date(2026, 10, 11)
    assert live_scores._utc_date(None) is None


# --- finding the ESPN event by team names ----------------------------------------------------


def _espn(home, away, event_id, day=dt.date(2026, 10, 11)):
    return espn_client.EspnMatch(
        espn_home_id="1",
        espn_away_id="2",
        home_name=home,
        away_name=away,
        date=day,
        kickoff_utc=dt.datetime(day.year, day.month, day.day, 15, 0),
        completed=True,
        home_score=1,
        away_score=0,
        state="post",
        event_id=event_id,
    )


def test_find_event_by_names_matches_across_naming_differences(monkeypatch):
    day = dt.date(2026, 10, 11)
    monkeypatch.setattr(
        espn_client,
        "fetch_day_matches",
        lambda slug, d, *a, **k: [_espn("Manchester City", "Brighton & Hove Albion", "10"), _espn("Arsenal", "Chelsea", "11")]
        if d == day
        else [],
    )
    assert espn_client.find_event_by_names("eng.1", day, "Manchester City FC", "Brighton & Hove Albion FC") == ("10", "post")
    assert espn_client.find_event_by_names("eng.1", day, "Arsenal FC", "Chelsea FC") == ("11", "post")
    # Home and away are not interchangeable, and unrelated teams don't match.
    assert espn_client.find_event_by_names("eng.1", day, "Chelsea", "Arsenal") is None
    assert espn_client.find_event_by_names("eng.1", day, "Everton", "Fulham") is None


def test_find_event_by_names_checks_neighbouring_days_for_late_kickoffs(monkeypatch):
    day = dt.date(2026, 10, 11)
    seen = []

    def day_matches(slug, d, *a, **k):
        seen.append(d)
        return [_espn("Arsenal", "Chelsea", "99", d)] if d == day + dt.timedelta(days=1) else []

    monkeypatch.setattr(espn_client, "fetch_day_matches", day_matches)
    assert espn_client.find_event_by_names("eng.1", day, "Arsenal", "Chelsea") == ("99", "post")
    assert seen == [day, day - dt.timedelta(days=1), day + dt.timedelta(days=1)]


def test_find_event_by_names_stops_at_the_given_day_when_it_has_the_match(monkeypatch):
    day = dt.date(2026, 10, 11)
    seen = []
    monkeypatch.setattr(
        espn_client, "fetch_day_matches", lambda slug, d, *a, **k: seen.append(d) or [_espn("Arsenal", "Chelsea", "5")]
    )
    assert espn_client.find_event_by_names("eng.1", day, "Arsenal", "Chelsea") == ("5", "post")
    assert seen == [day]


# --- resolving the page URL ------------------------------------------------------------------


def _today() -> dt.date:
    return dt.datetime.now(dt.UTC).date()


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _query(monkeypatch, **params):
    monkeypatch.setattr(view.st, "query_params", params)


def _scoreboard(monkeypatch, *event_ids):
    """Today's and yesterday's scoreboard hold exactly these events."""
    matches = [_espn("Arsenal", "Chelsea", event_id, _today()) for event_id in event_ids]
    monkeypatch.setattr(espn_client, "fetch_day_matches", lambda slug, d, *a, **k: matches if d == _today() else [])


def test_event_links_only_open_matches_on_the_recent_scoreboard(session, monkeypatch):
    _scoreboard(monkeypatch, "401861119")
    _query(monkeypatch, league="NL", event="401861119")
    assert view.resolve_match(session, NL).event_id == "401861119"
    # A number ESPN has no recent match for is never requested (no walking through ids).
    _query(monkeypatch, league="NL", event="401861120")
    assert view.resolve_match(session, NL) is None
    _query(monkeypatch, league="NL", event="1; drop table")
    assert view.resolve_match(session, NL) is None


def test_name_links_look_the_match_up_on_espn(session, monkeypatch):
    monkeypatch.setattr(espn_client, "find_event_by_names", lambda *a, **k: ("55", "in"))
    _query(monkeypatch, league="EPL", home_name="Arsenal", away_name="Chelsea", date=_today().isoformat())
    resolved = view.resolve_match(session, EPL)
    assert (resolved.event_id, resolved.state, resolved.home_name) == ("55", "in", "Arsenal")


def test_name_links_are_refused_unless_recent_short_and_confirmed_by_espn(session, monkeypatch):
    calls = []
    monkeypatch.setattr(espn_client, "find_event_by_names", lambda *a, **k: calls.append(a) or ("55", "in"))
    today = _today()

    for date in (today - dt.timedelta(days=30), today + dt.timedelta(days=30)):
        _query(monkeypatch, league="EPL", home_name="Arsenal", away_name="Chelsea", date=date.isoformat())
        assert view.resolve_match(session, EPL) is None
    _query(monkeypatch, league="EPL", home_name="A" * 500, away_name="Chelsea", date=today.isoformat())
    assert view.resolve_match(session, EPL) is None
    assert calls == []  # none of those reached ESPN

    # Text ESPN does not recognise as a match is never put on the page.
    monkeypatch.setattr(espn_client, "find_event_by_names", lambda *a, **k: None)
    _query(monkeypatch, league="EPL", home_name="[free money](http://x.example)", away_name="Chelsea", date=today.isoformat())
    assert view.resolve_match(session, EPL) is None


def test_team_id_links_fall_back_to_names_when_the_teams_have_no_espn_id(session, monkeypatch):
    home, away = get_or_create_team(session, "Arsenal", "EPL"), get_or_create_team(session, "Chelsea", "EPL")
    session.commit()
    day = dt.date(2026, 10, 11)
    upsert_match(session, "EPL", "2627", day, home.id, away.id, 1, 0)
    captured = {}

    def by_names(slug, d, home_name, away_name, **kwargs):
        captured.update(slug=slug, home=home_name, away=away_name)
        return ("66", "post")

    monkeypatch.setattr(espn_client, "find_event_by_names", by_names)
    _query(monkeypatch, league="EPL", home=str(home.id), away=str(away.id), date=day.isoformat())

    resolved = view.resolve_match(session, EPL)

    assert resolved.event_id == "66" and captured == {"slug": "eng.1", "home": "Arsenal", "away": "Chelsea"}


def test_team_id_links_for_a_match_we_do_not_have_make_no_request(session, monkeypatch):
    home, away = get_or_create_team(session, "Arsenal", "EPL"), get_or_create_team(session, "Chelsea", "EPL")
    session.commit()
    monkeypatch.setattr(espn_client, "find_event_by_names", lambda *a, **k: pytest.fail("asked ESPN about an unknown match"))
    monkeypatch.setattr(espn_client, "find_event_id", lambda *a, **k: pytest.fail("asked ESPN about an unknown match"))
    # Real teams but a made-up date: nothing stored, so nothing is looked up.
    _query(monkeypatch, league="EPL", home=str(home.id), away=str(away.id), date="2019-03-04")
    assert view.resolve_match(session, EPL) is None


def test_unusable_links_resolve_to_nothing(session, monkeypatch):
    _query(monkeypatch, league="EPL")
    assert view.resolve_match(session, EPL) is None
    _query(monkeypatch, league="EPL", home="999", away="998", date="2026-10-11")
    assert view.resolve_match(session, EPL) is None


def test_plain_strips_markdown_from_text_shown_on_the_page():
    hostile = "[free money](http://evil.example) **bold** ![img](http://evil.example/x.png) $5 <b>"
    escaped = view.plain(hostile)
    for ch in "[]()*!$<>":
        # every special character is backslash-escaped, so it renders as itself
        assert escaped.count(ch) == escaped.count("\\" + ch)
    assert view.plain(None) == "" and view.plain("Arsenal") == "Arsenal"


# --- predicted score on the page -------------------------------------------------------------


def _resolved(**kwargs):
    return view.ResolvedMatch(match_date=dt.date(2026, 10, 11), **kwargs)


def test_the_prediction_locked_in_before_kickoff_is_shown_when_there_is_one(session, monkeypatch):
    from soccer_predictor.storage.repository import insert_prediction_record

    home, away = get_or_create_team(session, "Arsenal", "EPL"), get_or_create_team(session, "Chelsea", "EPL")
    insert_prediction_record(session, "EPL", dt.date(2026, 10, 11), home.id, away.id, 2, 1)
    session.commit()
    monkeypatch.setattr(view, "load_latest_params", lambda *a: pytest.fail("the locked-in prediction should win"))

    assert view.predicted_scoreline(session, EPL, _resolved(home_team_id=home.id, away_team_id=away.id), None) == (2, 1)


def test_without_a_locked_in_prediction_the_current_model_is_used(session, monkeypatch):
    home, away = get_or_create_team(session, "Arsenal", "EPL"), get_or_create_team(session, "Chelsea", "EPL")
    session.commit()
    params = SimpleNamespace(attack={home.id: 1.0, away.id: 1.0})
    monkeypatch.setattr(view, "load_latest_params", lambda *a: params)
    monkeypatch.setattr(view, "predict_fixture", lambda *a, **k: SimpleNamespace(top_scorelines=[(1, 1, 0.12), (2, 1, 0.1)]))

    # Found by name, ignoring accents and case, when the link carries no team ids.
    assert view.predicted_scoreline(session, EPL, _resolved(home_name="ARSENAL", away_name="chelsea"), None) == (1, 1)


def test_no_prediction_for_unknown_teams_or_an_untrained_league(session, monkeypatch):
    home, away = get_or_create_team(session, "Arsenal", "EPL"), get_or_create_team(session, "Chelsea", "EPL")
    session.commit()
    assert view.predicted_scoreline(session, EPL, _resolved(home_name="Nobody FC", away_name="Chelsea"), None) is None
    monkeypatch.setattr(view, "load_latest_params", lambda *a: None)
    assert view.predicted_scoreline(session, EPL, _resolved(home_team_id=home.id, away_team_id=away.id), None) is None


def test_espn_team_ids_find_our_team_even_when_the_names_differ(session):
    team = get_or_create_team(session, "Türkiye", "NL")
    team.espn_team_id = 465
    session.commit()
    assert view._find_team(session, NL, "Turkey", "465").id == team.id


# --- team names link to the team's own page --------------------------------------------------


def test_each_side_of_a_match_is_found_whichever_way_the_link_named_it(session):
    home, away = get_or_create_team(session, "Arsenal", "EPL"), get_or_create_team(session, "Chelsea", "EPL")
    session.commit()

    by_id = view.match_teams(session, EPL, _resolved(home_team_id=home.id, away_team_id=away.id), None)
    by_name = view.match_teams(session, EPL, _resolved(home_name="arsenal", away_name="CHELSEA"), None)  # live banner link
    assert [t.id for t in by_id] == [t.id for t in by_name] == [home.id, away.id]
    # A side we have no row for gets no link, but the other side still does.
    unknown_home, known_away = view.match_teams(session, EPL, _resolved(home_name="Nobody FC", away_name="Chelsea"), None)
    assert unknown_home is None and known_away.id == away.id


def test_the_team_page_link_names_the_team_and_its_league(session):
    team = get_or_create_team(session, "Arsenal", "EPL")
    session.commit()
    assert view.team_page_params(team) == {"league": "EPL", "team": str(team.id)}
    assert view.team_page_params(None) is None


def test_the_team_link_styling_is_static_text_aimed_at_page_links():
    assert "stPageLink" in view.TEAM_LINK_CSS and "font-size" in view.TEAM_LINK_CSS
    assert view.TEAM_LINK_CSS.startswith("<style>") and "<script" not in view.TEAM_LINK_CSS


def _page_links(at):
    return [(el.proto.label, el.proto.query_string) for el in at.get("page_link")]


def test_a_team_name_is_a_link_to_its_page_and_plain_text_when_we_have_no_team():
    from streamlit.testing.v1 import AppTest

    def app():
        import streamlit as st

        from soccer_predictor.dashboard.views import match_detail

        left, right = st.columns(2)
        match_detail.render_team_name("Arsenal", {"league": "EPL", "team": "7"}, left)
        match_detail.render_team_name("Nobody *FC*", None, right)

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception
    assert _page_links(at) == [("**Arsenal**", "league=EPL&team=7")]
    assert not at.get("page_link")[0].proto.icon  # a page link shows its page's icon; the team page has none
    assert [m.value for m in at.markdown] == ["**Nobody \\*FC\\***"]  # markdown in the name stays inert


def test_the_match_header_and_both_lineups_link_each_team():
    from streamlit.testing.v1 import AppTest

    def app():
        from types import SimpleNamespace

        from soccer_predictor.dashboard.views import match_detail
        from soccer_predictor.ingest.espn_client import TeamMatchInfo

        home = TeamMatchInfo(espn_id="1", name="Arsenal", home_away="home", formation="4-3-3")
        away = TeamMatchInfo(espn_id="2", name="Chelsea", home_away="away", formation=None)
        detail = SimpleNamespace(
            home=home, away=away, home_score=2, away_score=1, status_label="Full time", venue=None, city=None,
            attendance=None, has_lineups=True, match_date=None,
        )
        params = ({"league": "EPL", "team": "1"}, None)  # no team row for the away side
        resolved = match_detail.ResolvedMatch(home_name="Arsenal", away_name="Chelsea")
        match_detail._render_header(detail, resolved, None, params)
        match_detail._render_lineup(home, params[0])
        match_detail._render_lineup(away, params[1])

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception
    # Arsenal links from the header and its lineup; Chelsea (no team) is plain text in both.
    assert _page_links(at) == [("**Arsenal**", "league=EPL&team=1")] * 2
    assert [m.value for m in at.markdown].count("**Chelsea**") == 2
    # Shown as "4-3-3"; the dashes are escaped like all feed text. Chelsea has no formation line.
    assert [c.value for c in at.caption if c.value.startswith("Formation")] == ["Formation 4\\-3\\-3"]
