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
