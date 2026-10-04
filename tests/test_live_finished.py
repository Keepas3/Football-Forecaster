"""Recently finished matches on the LIVE NOW banner: kept (with the final
score, marked FT) for FINISHED_LINGER after the estimated end, then dropped."""

from __future__ import annotations

import datetime as dt

import pytest
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.dashboard.components import live_scores_banner_html
from soccer_predictor.ingest import espn_client, live_scores
from soccer_predictor.ingest.api_client import MissingApiKey
from soccer_predictor.storage.models import Base

NL = League(code="NL", name="UEFA Nations League", seasons=["2627"], data_source="espn", espn_league_slug="uefa.nations")
EPL = League(code="EPL", name="English Premier League", api_competition_id=2021, seasons=["2526"], csv_code="E0")

# Kickoff 18:45 UTC -> estimated end 21:00 -> kept until 00:30.
KICKOFF = dt.datetime(2026, 10, 4, 18, 45)
END = KICKOFF + live_scores.ESTIMATED_MATCH_LENGTH


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _espn_match(state="post", completed=True, home="Greece", away="Germany", scores=(1, 2), kickoff=KICKOFF, clock=None):
    return espn_client.EspnMatch(
        espn_home_id="1",
        espn_away_id="2",
        home_name=home,
        away_name=away,
        date=kickoff.date(),
        kickoff_utc=kickoff,
        completed=completed,
        home_score=scores[0] if scores else None,
        away_score=scores[1] if scores else None,
        state=state,
        clock_label=clock,
    )


def _espn(monkeypatch, matches_by_day):
    monkeypatch.setattr(
        live_scores.espn_client, "fetch_day_matches", lambda slug, day, cache_ttl_seconds=None: matches_by_day.get(day, [])
    )


# --- the linger window -------------------------------------------------------------


def test_a_match_is_kept_from_its_estimated_end_until_the_linger_runs_out():
    recently = live_scores._recently_finished
    assert not recently(KICKOFF, KICKOFF + dt.timedelta(minutes=30))  # still being played
    assert recently(KICKOFF, END)
    assert recently(KICKOFF, END + live_scores.FINISHED_LINGER - dt.timedelta(minutes=1))
    assert not recently(KICKOFF, END + live_scores.FINISHED_LINGER)


def test_the_linger_is_three_to_four_hours():
    assert dt.timedelta(hours=3) <= live_scores.FINISHED_LINGER <= dt.timedelta(hours=4)


# --- ESPN --------------------------------------------------------------------------------


def test_espn_finished_match_shows_its_final_score_as_ft(monkeypatch):
    _espn(monkeypatch, {KICKOFF.date(): [_espn_match()]})

    [m] = live_scores.fetch_live_matches_espn(NL, now=END + dt.timedelta(hours=1))

    assert (m.home_score, m.away_score, m.clock_label, m.finished) == (1, 2, "FT", True)


def test_espn_finished_match_drops_off_after_the_linger(monkeypatch):
    _espn(monkeypatch, {KICKOFF.date(): [_espn_match()]})
    assert live_scores.fetch_live_matches_espn(NL, now=END + live_scores.FINISHED_LINGER + dt.timedelta(minutes=1)) == []


def test_espn_live_match_keeps_its_running_clock_and_is_not_finished(monkeypatch):
    live = _espn_match(state="in", completed=False, scores=(0, 1), clock="33'")
    _espn(monkeypatch, {KICKOFF.date(): [live]})

    [m] = live_scores.fetch_live_matches_espn(NL, now=KICKOFF + dt.timedelta(minutes=33))

    assert (m.clock_label, m.finished) == ("33'", False)


def test_espn_upcoming_and_postponed_matches_are_not_shown(monkeypatch):
    upcoming = _espn_match(state="pre", completed=False, scores=None)
    postponed = _espn_match(state="post", completed=False, scores=None, home="A", away="B")
    _espn(monkeypatch, {KICKOFF.date(): [upcoming, postponed]})
    assert live_scores.fetch_live_matches_espn(NL, now=END + dt.timedelta(hours=1)) == []


def test_espn_reads_yesterdays_scoreboard_for_a_match_that_straddles_midnight(monkeypatch):
    late = dt.datetime(2026, 10, 4, 22, 0)  # ends ~00:15 on the 5th
    _espn(monkeypatch, {late.date(): [_espn_match(kickoff=late)]})

    results = live_scores.fetch_live_matches_espn(NL, now=dt.datetime(2026, 10, 5, 1, 0))

    assert [m.clock_label for m in results] == ["FT"]


# --- football-data.org ------------------------------------------------------------------


def _fd_match(status="FINISHED", kickoff=KICKOFF, competition_id=2021, score=(3, 1)):
    return {
        "competition": {"id": competition_id},
        "homeTeam": {"name": "Arsenal FC"},
        "awayTeam": {"name": "Chelsea FC"},
        "score": {"fullTime": {"home": score[0], "away": score[1]}},
        "status": status,
        "utcDate": kickoff.isoformat() + "Z",
    }


def _fd(monkeypatch, matches):
    monkeypatch.setattr(live_scores, "load_leagues", lambda: {"EPL": EPL})
    monkeypatch.setattr(live_scores.api_client, "get", lambda *a, **k: {"matches": matches})


def test_football_data_finished_match_is_kept_with_its_final_score(session, monkeypatch):
    _fd(monkeypatch, [_fd_match()])

    [m] = live_scores.fetch_recent_finished_football_data_org(session, now=END + dt.timedelta(hours=1))

    assert (m.league_code, m.home_score, m.away_score, m.clock_label, m.finished) == ("EPL", 3, 1, "FT", True)


def test_football_data_old_untracked_or_unfinished_matches_are_left_out(session, monkeypatch):
    now = END + dt.timedelta(hours=1)
    old = _fd_match(kickoff=KICKOFF - dt.timedelta(days=1))
    untracked = _fd_match(competition_id=99999)
    live = _fd_match(status="IN_PLAY")
    _fd(monkeypatch, [old, untracked, live])
    assert live_scores.fetch_recent_finished_football_data_org(session, now=now) == []


@pytest.mark.parametrize("error", [MissingApiKey("no key"), requests.RequestException("down")])
def test_football_data_finished_fetch_degrades_to_empty(session, monkeypatch, error):
    monkeypatch.setattr(live_scores, "load_leagues", lambda: {"EPL": EPL})

    def boom(*a, **k):
        raise error

    monkeypatch.setattr(live_scores.api_client, "get", boom)
    assert live_scores.fetch_recent_finished_football_data_org(session) == []


# --- combined ----------------------------------------------------------------------------


def _lm(home, away, finished, kickoff=KICKOFF, league="NL"):
    return live_scores.LiveMatch(
        league_code=league,
        home_name=home,
        away_name=away,
        home_crest=None,
        away_crest=None,
        home_score=1,
        away_score=0,
        clock_label="FT" if finished else "50'",
        finished=finished,
        kickoff_utc=kickoff,
    )


def test_all_matches_lists_live_first_then_finished_most_recent_first(session, monkeypatch):
    monkeypatch.setattr(live_scores, "fetch_live_matches_football_data_org", lambda s: [])
    monkeypatch.setattr(live_scores, "fetch_recent_finished_football_data_org", lambda s: [])
    earlier = _lm("A", "B", True, kickoff=KICKOFF - dt.timedelta(hours=3))
    later = _lm("C", "D", True, kickoff=KICKOFF)
    live = _lm("E", "F", False)
    monkeypatch.setattr(live_scores, "fetch_live_matches_espn", lambda league: [earlier, later, live])

    results = live_scores.fetch_all_live_matches(session, {"NL": NL})

    assert [m.home_name for m in results] == ["E", "C", "A"]


def test_a_match_both_feeds_report_appears_once_as_finished(session, monkeypatch):
    # Around the final whistle a cached "live" copy and a "finished" copy can coexist.
    monkeypatch.setattr(live_scores, "fetch_live_matches_football_data_org", lambda s: [_lm("A", "B", False)])
    monkeypatch.setattr(live_scores, "fetch_recent_finished_football_data_org", lambda s: [_lm("A", "B", True)])

    results = live_scores.fetch_all_live_matches(session, {})

    assert [(m.home_name, m.finished) for m in results] == [("A", True)]


# --- banner markup -----------------------------------------------------------------------


def test_banner_shows_final_score_marked_ft_and_muted():
    markup = live_scores_banner_html([_lm("Greece", "Germany", True)], {"NL": NL})
    assert "Greece 1-0 Germany (FT)" in markup
    assert "opacity:0.7" in markup
    assert "FULL TIME" in markup and "LIVE NOW" not in markup


def test_banner_heading_says_live_now_whenever_something_is_in_progress():
    markup = live_scores_banner_html([_lm("A", "B", True), _lm("C", "D", False)], {"NL": NL})
    assert "LIVE NOW" in markup


def test_banner_without_a_score_reads_home_vs_away_with_no_empty_parentheses():
    m = live_scores.LiveMatch("NL", "Greece", "Germany", None, None, None, None, "")
    markup = live_scores_banner_html([m], {"NL": NL})
    assert "Greece vs Germany</a>" in markup
    assert "()" not in markup


def test_finished_matches_are_still_google_links():
    markup = live_scores_banner_html([_lm("Greece", "Germany", True)], {"NL": NL})
    assert 'href="https://www.google.com/search?q=Greece+vs+Germany+UEFA+Nations+League"' in markup
