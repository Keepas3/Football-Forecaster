"""A team's recent results as HTML: every cell but the highlights link opens the match's
details page, the highlights link opens YouTube in a new tab."""

from __future__ import annotations

import datetime as dt
import re

from soccer_predictor.dashboard.components import FixtureRef, results_table_html

DAY = dt.date(2026, 10, 4)
REF = FixtureRef(7, 17, "Chelsea", "Man United", DAY, league_code="EPL", match_details=True)
WATCH = "https://www.youtube.com/results?search_query=Chelsea+vs+Man+United+2026-10-04+highlights"


def _row(**overrides):
    row = {"Date": DAY, "Opponent": "vs Man United", "Score": "2-1", "Result": "W", "Watch": WATCH}
    row.update(overrides)
    return row


def _hrefs(markup: str) -> list[str]:
    return re.findall(r'href="([^"]*)"', markup)


def test_every_cell_but_the_highlights_opens_the_match_details():
    markup = results_table_html([_row()], [REF], team_id=7, season="2627")
    details = "/match_detail?league=EPL&amp;home=7&amp;away=17&amp;date=2026-10-04&amp;team=7&amp;season=2627"
    # Date, opponent, score and result each link to the details; the last link is the highlights.
    assert _hrefs(markup) == [details] * 4 + [WATCH]
    assert markup.count('target="_self"') == 4
    assert f'<a class="watch" href="{WATCH}" target="_blank" rel="noopener noreferrer">▶ Highlights</a>' in markup


def test_headers_and_the_result_is_coloured():
    markup = results_table_html([_row(), _row(Result="D"), _row(Result="L")], [REF] * 3, team_id=7)
    assert markup.count("<th>") == 5 and "<th>Watch</th>" in markup and "<th>Competition</th>" not in markup
    assert 'style="color: #2e7d32; font-weight: 600">W</a>' in markup
    assert 'style="color: #9e9e9e">D</a>' in markup and 'style="color: #c62828">L</a>' in markup
    assert "season=" not in markup  # no season on the page, none in the links


def test_a_competition_with_no_match_page_has_plain_cells_and_still_the_highlights():
    nowhere = FixtureRef(7, 17, "Chelsea", "Man United", DAY, league_code="XYZ", match_details=False)
    markup = results_table_html([_row()], [nowhere], team_id=7)
    assert _hrefs(markup) == [WATCH]  # nothing else is a link
    assert '<span style="color: #2e7d32; font-weight: 600">W</span>' in markup and "vs Man United" in markup


def test_a_row_from_another_competition_opens_that_competitions_match_page():
    elsewhere = FixtureRef(500, 501, "FC Barcelona", "Arsenal FC", DAY, league_code="UCL", match_details=True)
    row = _row(Competition="UEFA Champions League", Opponent="@ Arsenal FC")
    markup = results_table_html([row], [elsewhere], team_id=29, season="2526")
    assert "<th>Competition</th>" in markup and "UEFA Champions League" in markup
    assert _hrefs(markup)[0] == "/match_detail?league=UCL&amp;home=500&amp;away=501&amp;date=2026-10-04&amp;team=29&amp;season=2526"
    assert len(_hrefs(markup)) == 6  # date, competition, opponent, score, result, highlights


def test_text_from_outside_is_escaped():
    hostile = '"><script>alert(1)</script>'
    markup = results_table_html([_row(Opponent=hostile, Watch='x"><script>')], [REF], team_id=7)
    assert "<script" not in markup and "&lt;script&gt;" in markup
