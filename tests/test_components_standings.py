from __future__ import annotations

from soccer_predictor.config import TableZone
from soccer_predictor.dashboard.components import standings_dataframe, style_standings
from soccer_predictor.model.standings import TeamStanding

ZONES = [
    TableZone(start=1, end=4, kind="qualify", label="Champions League"),
    TableZone(start=6, end=6, kind="playoff", label="Relegation playoff"),
    TableZone(start=7, end=8, kind="relegation", label="Relegation"),
]


def _standing(team_id: int, name: str, points: int) -> TeamStanding:
    s = TeamStanding(team_id=team_id, team_name=name)
    s.won = points // 3
    return s


def test_zone_kind_assigned_by_position():
    standings = [_standing(i, f"Team {i}", points=0) for i in range(1, 9)]
    df = standings_dataframe(standings, crest_urls={}, next_opponent_names={}, zones=ZONES)

    assert list(df["_zone_kind"][:4]) == ["qualify"] * 4
    assert list(df["_zone_kind"][4:5]) == [""]
    assert df["_zone_kind"][5] == "playoff"
    assert list(df["_zone_kind"][6:8]) == ["relegation"] * 2


def test_no_zones_gives_blank_zone_kind():
    standings = [_standing(1, "Team 1", points=0)]
    df = standings_dataframe(standings, crest_urls={}, next_opponent_names={})
    assert df["_zone_kind"][0] == ""


def test_zone_column_no_longer_present():
    standings = [_standing(1, "Team 1", points=0)]
    df = standings_dataframe(standings, crest_urls={}, next_opponent_names={}, zones=ZONES)
    assert "Zone" not in df.columns


def test_style_standings_colors_pos_by_zone():
    standings = [_standing(i, f"Team {i}", points=0) for i in range(1, 9)]
    df = standings_dataframe(standings, crest_urls={}, next_opponent_names={}, zones=ZONES)
    html = style_standings(df).to_html()

    # Sanity: the hex colors chosen for each zone kind actually show up in
    # the rendered HTML (rather than asserting on cell-by-cell CSS, which
    # the pandas Styler API makes awkward to inspect directly).
    assert "#2e7d32" in html  # qualify -> green
    assert "#f9a825" in html  # playoff -> amber/yellow
    assert "#c62828" in html  # relegation -> red


def test_style_standings_no_color_when_no_zones():
    standings = [_standing(1, "Team 1", points=0)]
    df = standings_dataframe(standings, crest_urls={}, next_opponent_names={})
    html = style_standings(df).to_html()

    assert "#2e7d32" not in html
    assert "#f9a825" not in html
    assert "#c62828" not in html


# --- "Next" column labels ---------------------------------------------------------

import datetime as dt  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

import pandas as pd  # noqa: E402

from soccer_predictor.dashboard.components import next_match_labels  # noqa: E402

NAMES = {1: "Portugal", 2: "Norway", 3: "Denmark", 4: "Wales"}
NOW = dt.datetime(2026, 10, 4, 19, 0)  # naive UTC, like Fixture.kickoff_utc
UTC = ZoneInfo("UTC")


def _fixtures(rows):
    return pd.DataFrame(rows, columns=["date", "home_team_id", "away_team_id", "kickoff_utc"])


def test_next_label_is_the_opponent_and_the_date_with_no_time():
    df = _fixtures([(dt.date(2026, 10, 6), 1, 2, dt.datetime(2026, 10, 6, 18, 45))])
    labels = next_match_labels(df, NAMES, UTC, now=NOW)
    assert labels == {1: "Norway · Oct 06", 2: "Portugal · Oct 06"}


def test_next_label_shows_live_instead_of_the_date_while_the_match_is_on():
    df = _fixtures([(dt.date(2026, 10, 4), 1, 2, dt.datetime(2026, 10, 4, 18, 45))])  # kicked off 15 min ago
    labels = next_match_labels(df, NAMES, UTC, now=NOW)
    assert labels[1] == "Norway · 🔴 LIVE"
    assert labels[2] == "Portugal · 🔴 LIVE"


def test_a_finished_match_is_skipped_so_the_following_one_shows():
    df = _fixtures(
        [
            (dt.date(2026, 10, 4), 1, 2, dt.datetime(2026, 10, 4, 12, 0)),  # over (7h ago)
            (dt.date(2026, 10, 7), 1, 3, dt.datetime(2026, 10, 7, 18, 45)),
        ]
    )
    labels = next_match_labels(df, NAMES, UTC, now=NOW)
    assert labels[1] == "Denmark · Oct 07"
    assert 2 not in labels  # Norway has nothing else scheduled


def test_a_live_match_comes_before_a_later_one_for_the_same_team():
    df = _fixtures(
        [
            (dt.date(2026, 10, 7), 1, 3, dt.datetime(2026, 10, 7, 18, 45)),
            (dt.date(2026, 10, 4), 1, 2, dt.datetime(2026, 10, 4, 18, 45)),
        ]
    )
    assert next_match_labels(df, NAMES, UTC, now=NOW)[1] == "Norway · 🔴 LIVE"


def test_the_date_is_shown_in_the_chosen_timezone():
    # 01:00 UTC on Oct 7 is still the evening of Oct 6 in New York.
    df = _fixtures([(dt.date(2026, 10, 7), 1, 2, dt.datetime(2026, 10, 7, 1, 0))])
    assert next_match_labels(df, NAMES, ZoneInfo("America/New_York"), now=NOW)[1] == "Norway · Oct 06"
    assert next_match_labels(df, NAMES, UTC, now=NOW)[1] == "Norway · Oct 07"


def test_a_fixture_with_no_kickoff_time_uses_its_date_and_never_counts_as_live():
    df = _fixtures([(dt.date(2026, 10, 4), 1, 2, None)])
    assert next_match_labels(df, NAMES, UTC, now=NOW)[1] == "Norway · Oct 04"


def test_no_fixtures_gives_no_labels():
    assert next_match_labels(_fixtures([]), NAMES, UTC, now=NOW) == {}


# --- Search link column ---------------------------------------------------------------

from soccer_predictor.dashboard.components import STANDINGS_DISPLAY_COLUMNS, next_match_links  # noqa: E402


def test_every_team_with_a_next_match_gets_a_search_link_live_or_upcoming():
    df = _fixtures(
        [
            (dt.date(2026, 10, 4), 1, 2, dt.datetime(2026, 10, 4, 18, 45)),  # live (15 min in)
            (dt.date(2026, 10, 6), 3, 4, dt.datetime(2026, 10, 6, 18, 45)),  # upcoming
        ]
    )
    links = next_match_links(df, NAMES, "UEFA Nations League", now=NOW)
    assert links[1] == links[2] == "https://www.google.com/search?q=Portugal+vs+Norway+UEFA+Nations+League"
    assert links[3] == links[4] == "https://www.google.com/search?q=Denmark+vs+Wales+UEFA+Nations+League"


def test_the_link_is_for_the_same_match_the_next_column_shows():
    df = _fixtures(
        [
            (dt.date(2026, 10, 4), 1, 2, dt.datetime(2026, 10, 4, 12, 0)),  # finished -> skipped
            (dt.date(2026, 10, 7), 1, 3, dt.datetime(2026, 10, 7, 18, 45)),
            (dt.date(2026, 10, 9), 1, 4, dt.datetime(2026, 10, 9, 18, 45)),  # a later one for Portugal
        ]
    )
    links = next_match_links(df, NAMES, "UEFA Nations League", now=NOW)
    assert links[1] == "https://www.google.com/search?q=Portugal+vs+Denmark+UEFA+Nations+League"
    assert 2 not in links  # Norway's only match is over -> blank


def test_a_fixture_with_no_kickoff_time_still_gets_a_link():
    df = _fixtures([(dt.date(2026, 10, 8), 1, 2, None)])
    assert 1 in next_match_links(df, NAMES, "UEFA Nations League", now=NOW)


def test_no_fixtures_gives_no_links():
    assert next_match_links(_fixtures([]), NAMES, "UEFA Nations League", now=NOW) == {}


def test_standings_dataframe_has_a_search_column_empty_without_a_match():
    standings = [TeamStanding(1, "Portugal"), TeamStanding(2, "Norway")]
    df = standings_dataframe(standings, {}, {}, match_links={1: "https://example.com/x"})
    assert df["Search"].tolist() == ["https://example.com/x", ""]
    assert "Search" in STANDINGS_DISPLAY_COLUMNS
