"""The Champions League history archive (openfootball/champions-league): the 90-minute
score, the dates, and which lines are matches at all."""

from __future__ import annotations

import datetime as dt

import pytest

from soccer_predictor.ingest import ucl_archive

# Lines as they appear in the real files (2018-19, 2015-16, 2023-24): the date is carried by
# the line above, the second match of a day has no time, and a knockout tie that went on
# lists its 90-minute score first in the bracket and the half-time score second.
SEASON_1819 = """\
= UEFA Champions League 2018/19

# Date       Tue Sep 18 2018 - Sat Jun 1 2019 (256d)
# Teams      32
# Matches    125

▪ Group A
  Tue Sep 18 2018
    21:00  AS Monaco (FRA)         v Atlético Madrid (ESP)    1-2 (1-2)
           Club Brugge KV (BEL)    v Borussia Dortmund (GER)  0-1 (0-0)
  Wed Dec 12
    21:00  Borussia Dortmund (GER) v Club Brugge KV (BEL)     0-0

▪ Group C
  Tue Sep 18
    21:00  Liverpool FC (ENG)      v Paris Saint-Germain (FRA)  3-2 (2-1)

▪ Round of 16
  Wed Mar 6
    21:00  FC Porto (POR)          v AS Roma (ITA)            3-1 a.e.t. (2-1, 1-1)
  Wed Mar 13
    21:00  Bayern München (GER)    v Liverpool FC (ENG)       1-3 (1-1)
"""

SHOOTOUTS = """\
  Wed Feb 24 2016
    20:45  Atlético Madrid (ESP)   v PSV Eindhoven (NED)      8-7 pen. 0-0 a.e.t. (0-0)
  Tue Mar 12 2024
    21:00  Arsenal FC (ENG)        v FC Porto (POR)           4-2 pen. 1-0 a.e.t. (1-0, 1-0)
"""

# 2019-20 was finished behind closed doors in August 2020.
SEASON_1920 = """\
▪ Group A
  Tue Sep 17 2019
    21:00  Club Brugge KV (BEL)    v Galatasaray (TUR)        0-0
▪ Round of 16
  Wed Feb 19
    21:00  Atalanta (ITA)          v Valencia CF (ESP)        4-1 (1-1)
  Fri Aug 7
    21:00  Manchester City (ENG)   v Real Madrid (ESP)        2-1 (1-0)
"""


def _matches(text: str, season: str):
    return {(m["home_team_name"], m["away_team_name"]): m for m in ucl_archive.parse_matches(text, season)}


def test_matches_are_read_with_the_date_carried_down_from_the_header_above():
    found = _matches(SEASON_1819, "1819")

    assert len(found) == 6
    monaco = found[("AS Monaco", "Atlético Madrid")]
    assert (monaco["date"], monaco["home_goals"], monaco["away_goals"]) == (dt.date(2018, 9, 18), 1, 2)
    # The second line of a day has no time of its own but shares the day.
    assert found[("Club Brugge KV", "Borussia Dortmund")]["date"] == dt.date(2018, 9, 18)
    # A match with no half-time score in brackets, on a later date line with no year.
    assert found[("Borussia Dortmund", "Club Brugge KV")]["date"] == dt.date(2018, 12, 12)
    assert found[("Liverpool FC", "Paris Saint-Germain")]["home_goals"] == 3


def test_headers_comments_and_section_titles_are_not_matches():
    names = {name for pair in _matches(SEASON_1819, "1819") for name in pair}
    assert "Group A" not in names and not any(name.startswith(("=", "#", "▪")) for name in names)
    assert ucl_archive.parse_matches("", "1819") == []
    assert ucl_archive.parse_matches("▪ Final\n  Sat Jun 1\n    21:00  nothing to see here\n", "1819") == []


def test_the_score_is_what_stood_after_90_minutes_not_after_extra_time():
    found = _matches(SEASON_1819, "1819")
    porto = found[("FC Porto", "AS Roma")]
    # Final 3-1 after extra time, but 2-1 at 90 minutes (the first number in the brackets).
    assert (porto["home_goals"], porto["away_goals"]) == (2, 1)
    # An ordinary match's bracket is the half-time score, which must not be mistaken for the result.
    assert found[("Bayern München", "Liverpool FC")]["home_goals"] == 1
    assert found[("Bayern München", "Liverpool FC")]["away_goals"] == 3


def test_a_shootout_tie_is_stored_by_its_90_minute_score():
    found = ucl_archive.parse_matches(SHOOTOUTS, "1516")  # the dates carry their own year
    assert [(m["home_goals"], m["away_goals"]) for m in found] == [(0, 0), (1, 0)]
    assert [m["date"] for m in found] == [dt.date(2016, 2, 24), dt.date(2024, 3, 12)]


def test_an_extra_time_line_without_its_bracket_is_skipped_not_guessed():
    text = "  Wed Mar 6 2019\n    21:00  FC Porto (POR)  v AS Roma (ITA)  3-1 a.e.t.\n"
    assert ucl_archive.parse_matches(text, "1819") == []


def test_months_map_to_the_seasons_two_calendar_years():
    found = _matches(SEASON_1920, "1920")
    assert found[("Club Brugge KV", "Galatasaray")]["date"] == dt.date(2019, 9, 17)
    assert found[("Atalanta", "Valencia CF")]["date"] == dt.date(2020, 2, 19)
    # The 2019-20 knockout rounds were played in August 2020, still the season's second year.
    assert found[("Manchester City", "Real Madrid")]["date"] == dt.date(2020, 8, 7)


def test_seasons_map_to_the_repository_folders():
    assert ucl_archive.SEASON_TO_FOLDER["1516"] == "2015-16"
    assert ucl_archive.SEASON_TO_FOLDER["1920"] == "2019-20"
    assert ucl_archive.SEASON_TO_FOLDER["2223"] == "2022-23"
    # football-data.org serves 2023-24 on, so the archive stops just before it.
    assert "2324" not in ucl_archive.ARCHIVE_SEASONS and "2223" in ucl_archive.ARCHIVE_SEASONS


def test_the_frame_has_the_columns_the_ingest_expects(monkeypatch):
    monkeypatch.setattr(ucl_archive, "fetch_season_text", lambda season: SEASON_1819)
    frame = ucl_archive.matches_frame("1819")
    assert list(frame.columns) == ["date", "home_team_name", "away_team_name", "home_goals", "away_goals"]
    assert len(frame) == 6
    monkeypatch.setattr(ucl_archive, "fetch_season_text", lambda season: "")
    assert ucl_archive.matches_frame("1819").empty


def test_a_season_is_downloaded_once_and_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(ucl_archive, "CACHE_DIR", tmp_path)
    calls = []

    class Response:
        content = "▪ Group A\r\n".encode()

        def raise_for_status(self):
            pass

    def fake_get(url, timeout):
        calls.append(url)
        return Response()

    monkeypatch.setattr(ucl_archive.requests, "get", fake_get)
    assert ucl_archive.fetch_season_text("1516") == "▪ Group A\n"  # read back as text
    ucl_archive.fetch_season_text("1516")
    assert calls == ["https://raw.githubusercontent.com/openfootball/champions-league/master/2015-16/cl.txt"]
    assert ucl_archive.fetch_season_text("2425") == ""  # not an archive season: no request at all
    assert len(calls) == 1


@pytest.mark.parametrize("season", sorted(ucl_archive.ARCHIVE_SEASONS))
def test_every_archive_season_has_a_folder(season):
    assert ucl_archive.SEASON_TO_FOLDER[season][:2] == "20"
