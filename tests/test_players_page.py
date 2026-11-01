"""The Players page: the top list and the search results are HTML tables whose player names open
that player's stats under the table without a reload (a click types into a hidden box, as on the
Predictor), and whose club names open the team's page."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from streamlit.testing.v1 import AppTest

from soccer_predictor.dashboard.views import players as players_view
from soccer_predictor.ingest.squad import PlayerSearchResult, SquadPlayer
from soccer_predictor.ingest.top_players import TopPlayer, TopPlayers
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import get_or_create_team

SAKA = "Bukayo Saka|Arsenal FC"
YAMAL = "Lamine Yamal|FC Barcelona"


@pytest.fixture
def page(monkeypatch):
    """The Players page for the Champions League (no stats source, so nothing is fetched for a
    picked player) over an in-memory database with two clubs; returns a function that runs it,
    and the clubs' ids."""
    engine = create_engine("sqlite://", future=True, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = Session(engine)
    arsenal = get_or_create_team(session, "Arsenal FC", "UCL")
    barca = get_or_create_team(session, "FC Barcelona", "UCL")
    session.commit()

    @contextmanager
    def scope():
        yield session
        session.flush()

    squads = [
        PlayerSearchResult(arsenal.id, "Arsenal FC", SquadPlayer("Bukayo Saka", "Winger", "England", "2001-09-05")),
        PlayerSearchResult(barca.id, "FC Barcelona", SquadPlayer("Lamine Yamal", "Winger", "Spain", "2007-07-13")),
    ]
    top = TopPlayers(
        [TopPlayer("Bukayo Saka", "Arsenal FC", 5, 3), TopPlayer("Lamine Yamal", "FC Barcelona", 4, 6),
         TopPlayer("Visiting Star", "Mystery FC", 3, 1)],
        "football-data.org (top scorers)",
        None,
    )
    monkeypatch.setattr(players_view, "session_scope", scope)
    monkeypatch.setattr(players_view, "fetch_top_players", lambda league: top)
    monkeypatch.setattr(
        players_view,
        "search_players_in_league",
        lambda _session, _league, query: [r for r in squads if query.lower() in r.player.name.lower()],
    )

    def run(**query):
        def app():
            from soccer_predictor.dashboard.views import players as view

            view.render()

        at = AppTest.from_function(app, default_timeout=60)
        at.query_params["league"] = "UCL"
        for name, value in query.items():
            at.query_params[name] = value
        return at.run()

    yield run, (arsenal.id, barca.id)
    session.close()


def _table(at) -> str | None:
    return next((m.value for m in at.markdown if 'class="fb-table' in m.value), None)


def _close_buttons(at) -> list[str]:
    return [b.key for b in at.button if b.key and b.key.startswith("close_player")]


def _search(at, text):
    next(t for t in at.text_input if t.label == "Search player name").set_value(text).run()
    return at


def _pick(at, value):
    at.text_input(key="player_pick").set_value(value).run()
    return at


def test_the_top_players_are_an_html_table_with_player_and_club_links(page):
    run, (arsenal_id, barca_id) = page
    at = run()
    assert not at.exception
    markup = _table(at)
    assert markup is not None and not at.dataframe  # no canvas grid any more
    assert 'class="player"' in markup and "Bukayo%20Saka%7CArsenal%20FC" in markup
    assert f'href="/team_detail?league=UCL&amp;team={arsenal_id}"' in markup
    assert f'href="/team_detail?league=UCL&amp;team={barca_id}"' in markup
    assert ">Mystery FC<" in markup and "Mystery FC</a>" not in markup  # a club we cannot place is plain text
    assert not _close_buttons(at) and 'class="selected"' not in markup
    assert any("Click a player's name for their stats below the table" in c.value for c in at.caption)


def test_a_click_on_a_players_name_opens_their_stats_under_the_table_without_reloading(page):
    run, _ = page
    at = _pick(run(), f"{SAKA}@1000")
    assert not at.exception
    assert at.query_params["player"] == [SAKA]  # the address still names the player
    assert 'class="selected"' in _table(at) and 'class="player on"' in _table(at)
    assert _close_buttons(at) == ["close_player_UCL"]
    assert any(s.value == "Bukayo Saka" for s in at.subheader)
    assert any("Winger · England · Age" in c.value and "Arsenal FC" in c.value for c in at.caption)


def test_another_click_switches_player_and_the_panel_closes_until_clicked_again(page):
    run, _ = page
    at = _pick(run(), f"{SAKA}@1000")
    _pick(at, f"{YAMAL}@2000")
    assert at.query_params["player"] == [YAMAL]
    assert [s.value for s in at.subheader if s.value in ("Bukayo Saka", "Lamine Yamal")] == ["Lamine Yamal"]

    at.button(key="close_player_UCL").click().run()
    assert not _close_buttons(at) and "player" not in at.query_params
    at.run()  # a rerun for any other reason must not bring the old click back
    assert not _close_buttons(at)
    _pick(at, f"{YAMAL}@3000")  # the script stamps each click with the time, so the same player can reopen
    assert _close_buttons(at) == ["close_player_UCL"]


def test_a_link_to_a_player_opens_them_straight_away(page):
    run, _ = page
    at = run(player=SAKA)
    assert not at.exception and _close_buttons(at) == ["close_player_UCL"]
    assert any(s.value == "Bukayo Saka" for s in at.subheader)


@pytest.mark.parametrize("junk", ["Nobody At All|Nowhere FC", "|Arsenal FC", "   "])
def test_a_player_the_squads_do_not_have_is_dropped_from_the_link(page, junk):
    run, _ = page
    at = run(player=junk)
    assert not at.exception
    assert "player" not in at.query_params and not _close_buttons(at)
    if junk.startswith("Nobody"):
        assert any("No squad entry found for Nobody At All" in c.value for c in at.caption)


def test_a_link_from_another_page_must_open_the_player_at_the_club_it_names(page):
    # The injuries table links "<name>|<our team name>". The lookup falls back to the same name at any
    # club, which for a link like this would be somebody else (or a transferred player): not opened.
    run, _ = page
    at = run(player="Bukayo Saka|FC Barcelona")
    assert not at.exception
    assert "player" not in at.query_params and not _close_buttons(at)
    assert any("No squad entry found for Bukayo Saka at FC Barcelona" in c.value for c in at.caption)


def test_a_club_in_a_stats_sources_own_spelling_still_finds_the_player(page):
    run, _ = page
    at = run(player="Bukayo Saka|Arsenal")  # the source says "Arsenal", our team is "Arsenal FC"
    assert not at.exception and _close_buttons(at) == ["close_player_UCL"]
    assert any(s.value == "Bukayo Saka" for s in at.subheader)


def test_the_message_for_a_name_the_squads_do_not_have_says_what_to_try(page):
    run, _ = page
    at = run(player="Nobody At All|Arsenal FC")
    text = " ".join(c.value for c in at.caption)
    assert "No squad entry found for Nobody At All at Arsenal FC." in text and "try searching for part of it" in text
    assert "player" not in at.query_params


def test_searching_shows_a_table_of_the_same_kind_and_a_click_opens_the_player(page):
    run, (arsenal_id, _) = page
    at = run()
    _search(at, "yamal")
    markup = _table(at)
    assert "Lamine Yamal" in markup and "Bukayo Saka" not in markup and not at.dataframe
    assert "<th" in markup and ">Position<" in markup and ">Winger<" in markup

    _pick(at, f"{YAMAL}@1")
    assert _close_buttons(at) == ["close_player_UCL"]
    assert 'class="selected"' in _table(at)  # the row is marked in the search table too

    _search(at, "nobody")
    assert any('No player matches "nobody"' in i.value for i in at.info)
    assert _close_buttons(at) == ["close_player_UCL"]  # a different search leaves the picked player open


def test_a_picked_player_is_dropped_when_the_league_changes(page, monkeypatch):
    run, _ = page
    at = _pick(run(), f"{SAKA}@1000")
    assert _close_buttons(at) == ["close_player_UCL"]
    at.sidebar.selectbox(key="players_league").select("EPL").run()
    assert not at.exception
    assert "player" not in at.query_params and not _close_buttons(at)


def test_the_page_draws_the_one_hidden_box_the_click_script_types_into(page):
    run, _ = page
    at = run()
    assert [t.key for t in at.text_input if t.key == "player_pick"] == ["player_pick"]
