from __future__ import annotations

from soccer_predictor.config import League
from soccer_predictor.dashboard.components import render_live_scores_banner
from soccer_predictor.ingest.live_scores import LiveMatch

EPL = League(
    code="EPL", name="English Premier League", api_competition_id=2021, seasons=["2526"], csv_code="E0",
    flag_emoji="🏴󠁧󠁢󠁥󠁮󠁧󠁿",
)
MLS = League(code="MLS", name="Major League Soccer", seasons=["2026"], data_source="espn", espn_league_slug="usa.1")


def _live_match(league_code="EPL", home="Arsenal", away="Chelsea", home_score=2, away_score=1, clock="63'"):
    return LiveMatch(
        league_code=league_code,
        home_name=home,
        away_name=away,
        home_crest=None,
        away_crest=None,
        home_score=home_score,
        away_score=away_score,
        clock_label=clock,
    )


def test_empty_list_is_a_no_op():
    # No Streamlit runtime needed for this path -- it returns before any
    # st.* call.
    assert render_live_scores_banner([], {"EPL": EPL}) is None


def test_does_not_raise_for_a_real_match(capsys):
    render_live_scores_banner([_live_match()], {"EPL": EPL})


def test_does_not_raise_for_a_league_with_no_flag_emoji():
    render_live_scores_banner([_live_match(league_code="MLS")], {"MLS": MLS})


def test_does_not_raise_when_league_is_unknown():
    render_live_scores_banner([_live_match(league_code="UNKNOWN")], {"EPL": EPL})


def test_does_not_raise_for_multiple_matches():
    matches = [_live_match(home="Arsenal", away="Chelsea"), _live_match(league_code="MLS", home="LAFC", away="Inter Miami CF", clock="HT")]
    render_live_scores_banner(matches, {"EPL": EPL, "MLS": MLS})
