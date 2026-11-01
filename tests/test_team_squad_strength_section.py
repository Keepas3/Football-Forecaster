"""The team page's Squad strength section: the usual XI on a pitch beside a table of the squad,
or a plain note where the competition has no player stats."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from soccer_predictor.model.lineup_adjustment import ExpectedPlayer


def _squad():
    squad = [ExpectedPlayer("Keeper One", "defense", 0.9, 3000, True, "GK", 0, 0)]
    squad += [ExpectedPlayer(f"Outfield {i}", "attack", 0.3, 2000 - i * 10, False, "MID", i, 1) for i in range(10)]
    squad += [ExpectedPlayer("Bench Player", "attack", 0.1, 90, False, "FWD", 0, 0)]
    return squad


def _page(monkeypatch, squad):
    from soccer_predictor.dashboard.views import team_detail

    monkeypatch.setattr(team_detail, "expected_squad", lambda team_name, league: squad)

    def app():
        from soccer_predictor.config import League
        from soccer_predictor.dashboard.views import team_detail as view

        view._render_squad_strength(League(code="EPL", name="Premier League", seasons=["2627"]), "Arsenal")

    return AppTest.from_function(app, default_timeout=60).run()


def test_the_section_shows_the_table_and_explains_its_figures(monkeypatch):
    at = _page(monkeypatch, _squad())
    assert not at.exception
    assert [s.value for s in at.subheader] == ["Squad strength"]
    table = next(m.value for m in at.markdown if 'class="fb-table compact"' in m.value)
    assert "Keeper One" in table and "Bench Player" in table and table.count("<tr>") == 1 + 12  # heading + players
    captions = " ".join(c.value for c in at.caption)
    assert "usual XI" in captions and "white circle" in captions and "Click a heading to sort" in captions


def test_a_team_with_no_player_stats_this_season_says_so(monkeypatch):  # a competition without any, or a club that has left
    at = _page(monkeypatch, [])
    assert not at.exception
    assert [s.value for s in at.subheader] == ["Squad strength"]
    assert any("No player stats are available" in c.value for c in at.caption)
    assert not any("fb-table" in m.value for m in at.markdown)


def test_without_a_league_there_is_nothing_to_show():
    def app():
        from soccer_predictor.dashboard.views import team_detail as view

        view._render_squad_strength(None, "Arsenal")

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception and any("unavailable" in c.value for c in at.caption)
