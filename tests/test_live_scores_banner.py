from __future__ import annotations

from soccer_predictor.config import League
from soccer_predictor.dashboard.components import live_scores_banner_html, render_live_scores_banner
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


def _rows(markup):
    # Each row is one top-level <div style="display:flex...">.
    return markup.count('<div style="display:flex')


def test_banner_names_each_matchs_league_not_its_flag():
    # The international competitions share the UN flag (shown as "UN" on
    # Windows), so each match must carry the league's name instead.
    un = League(code="NL", name="UEFA Nations League", seasons=["2627"], flag_emoji="🇺🇳")
    markup = live_scores_banner_html(
        [_live_match(league_code="NL", home="Greece", away="Germany", home_score=0, away_score=1, clock="33'")],
        {"NL": un},
    )
    assert "<b>UEFA Nations League</b> Greece 0-1 Germany (33&#x27;)" in markup
    assert "🇺🇳" not in markup


def test_banner_is_never_more_than_two_rows():
    leagues = {"EPL": EPL}
    for n, expected_rows in [(1, 1), (2, 2), (3, 2), (5, 2), (12, 2)]:
        markup = live_scores_banner_html([_live_match(home=f"Home{i}") for i in range(n)], leagues)
        assert _rows(markup) == expected_rows, n


def test_banner_splits_matches_across_the_two_rows_top_row_first():
    markup = live_scores_banner_html([_live_match(home=f"Home{i}") for i in range(5)], {"EPL": EPL})
    top, bottom = markup.split('<div style="display:flex')[1:]
    assert all(f"Home{i} " in top for i in range(3))  # the odd one out goes on top
    assert all(f"Home{i} " in bottom for i in (3, 4))
    assert "LIVE NOW" in top and "LIVE NOW" not in bottom


def test_banner_escapes_team_and_league_names():
    evil = League(code="X", name="<script>alert(1)</script>", seasons=["2627"])
    markup = live_scores_banner_html([_live_match(league_code="X", home="<img onerror=x>")], {"X": evil})
    assert "<script>" not in markup
    assert "<img" not in markup


def test_second_row_is_indented_to_line_up_under_the_first_rows_matches():
    from soccer_predictor.dashboard.components import LIVE_LABEL_WIDTH

    markup = live_scores_banner_html([_live_match(home=f"Home{i}") for i in range(4)], {"EPL": EPL})
    top, bottom = markup.split('<div style="display:flex')[1:]
    spacer = f'<span style="display:inline-block;min-width:{LIVE_LABEL_WIDTH};white-space:nowrap;"></span>'
    assert spacer in bottom and "LIVE NOW" not in bottom
    # The top row's label has the same width, so matches start at the same x.
    assert f"min-width:{LIVE_LABEL_WIDTH}" in top


# --- links ----------------------------------------------------------------------


def test_every_match_links_to_a_google_search_for_it():
    un = League(code="NL", name="UEFA Nations League", seasons=["2627"])
    markup = live_scores_banner_html(
        [_live_match(league_code="NL", home="Greece", away="Germany"), _live_match(home="Arsenal", away="Chelsea")],
        {"NL": un, "EPL": EPL},
    )
    assert markup.count("<a ") == 2
    assert 'href="https://www.google.com/search?q=Greece+vs+Germany+UEFA+Nations+League"' in markup
    assert 'href="https://www.google.com/search?q=Arsenal+vs+Chelsea+English+Premier+League"' in markup
    assert markup.count('target="_blank" rel="noopener noreferrer"') == 2


def test_link_urls_are_escaped_for_hostile_names():
    evil = League(code="X", name='"><script>alert(1)</script>', seasons=["2627"])
    markup = live_scores_banner_html([_live_match(league_code="X", home='"><b>x')], {"X": evil})
    assert "<script>" not in markup
    assert '"><b>x' not in markup  # the hostile team name never appears unescaped
    assert "&quot;&gt;&lt;b&gt;x" in markup
