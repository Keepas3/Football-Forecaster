"""Leagues page: a selector at the top switches between leagues; a real
league table (Played/W/D/L/GF/GA/GD/Pts/Form/Next) for the selected season
is shown directly below -- straight from actual results, independent of the
Dixon-Coles model. Clicking a team's row drills into Team Detail, where the
model's attack/defense rating lives instead.
"""

from __future__ import annotations

import datetime as dt
import html

import pandas as pd
import streamlit as st

from soccer_predictor.config import League, load_leagues
from soccer_predictor.dashboard import navigation
from soccer_predictor.dashboard.components import (
    HOME_WIN_COLUMN,
    LIVE_MATCH_WINDOW,
    STANDINGS_DISPLAY_COLUMNS,
    ZONE_MARKER,
    fixture_columns_with_kickoff_first,
    fixture_percent_column_config,
    fixture_prediction_row,
    format_kickoff,
    league_option_label,
    live_sync_requirement_note,
    render_head_to_head,
    render_live_scores_banner,
    render_news_section,
    render_prediction_breakdown,
    season_already_concluded,
    season_not_yet_scheduled,
    standings_dataframe,
    style_fixture_predictions,
    style_standings,
    timezone_selector,
)
from soccer_predictor.ingest.league_meta import fetch_competition_emblem
from soccer_predictor.ingest.live_scores import fetch_all_live_matches
from soccer_predictor.ingest.news import league_news_query
from soccer_predictor.model.standings import compute_standings
from soccer_predictor.model.team_facts import compute_head_to_head
from soccer_predictor.prediction.service import load_latest_params, predict_fixture
from soccer_predictor.storage.db import session_scope
from soccer_predictor.storage.repository import (
    fixtures_for_league,
    live_fixtures_across_leagues,
    matches_for_league,
    matches_for_team,
    next_fixture_per_team,
    team_by_id,
    team_conferences_for_league,
    team_crests_for_league,
    teams_for_league,
)

RANKINGS_TABLE_KEY = "league_rankings_table"
# Only ever used for a league with a real conference split (currently just
# MLS's Eastern/Western) -- see team_conferences_for_league.
_CONFERENCE_TABLE_KEYS = {
    "Eastern Conference": "league_rankings_table_east",
    "Western Conference": "league_rankings_table_west",
}
_CLEAR_SELECTION_FLAG = "_clear_rankings_selection"
UPCOMING_FIXTURES_WINDOW_DAYS = 14


def _season_label(season: str, league) -> str:
    # "2425" -> "2024/25" (the usual case); a single_year league (e.g. the
    # Euros) shows its season code as-is since it's already just the year.
    if league.season_display == "single_year":
        return season
    return f"20{season[:2]}/{season[2:]}"


def _render_standings_table(table_df: pd.DataFrame, table_key: str, search_key: str, league_code: str) -> None:
    """One searchable, clickable standings table -- shared by the flat
    single-table case and each per-conference table (see render()). Clicking
    a row navigates to Team Detail, same as before this was split out.
    """
    # Streamlit's dataframe grid (glide-data-grid) is a canvas, not real DOM
    # text -- the browser's own Ctrl+F can't see into it at all, so this is
    # a real search, not decoration: filtering table_df server-side before
    # it's ever rendered.
    search_query = st.text_input("Search team", placeholder="e.g. Man City", key=search_key)
    if search_query:
        filtered_df = table_df[table_df["Team"].str.contains(search_query, case=False, na=False, regex=False)]
    else:
        filtered_df = table_df

    if search_query and filtered_df.empty:
        st.info(f'No team matches "{search_query}".')
        return

    # Streamlit's dataframe otherwise caps itself at a fixed default height
    # with its own internal scrollbar -- size it to fit every (post-search)
    # row instead, so the whole table renders in one page (35px/row + 35px
    # header, per Streamlit's own default row height).
    table_height = 35 * (len(filtered_df) + 1) + 3

    event = st.dataframe(
        style_standings(filtered_df),
        use_container_width=True,
        hide_index=True,
        height=table_height,
        column_order=STANDINGS_DISPLAY_COLUMNS,
        column_config={"Crest": st.column_config.ImageColumn(" ", width="small")},
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        team_id = int(filtered_df.iloc[selected_rows[0]]["team_id"])
        # Defer clearing the selection to the top of the *next* run of this
        # page (see render()) -- can't touch it in this run, the widget
        # above is already instantiated.
        st.session_state[_CLEAR_SELECTION_FLAG] = True
        st.switch_page(navigation.team_detail_page(), query_params={"league": league_code, "team": str(team_id)})


LIVE_BANNER_REFRESH_SECONDS = 40


@st.fragment(run_every=LIVE_BANNER_REFRESH_SECONDS)
def _render_live_banner(leagues: dict[str, League]) -> None:
    """Global, independent of whichever league is selected below -- so a
    live Premier League game still shows up here while browsing La Liga.
    Reruns on its own every LIVE_BANNER_REFRESH_SECONDS (a Streamlit
    fragment -- only this banner reruns, not the whole page/selection
    state) so a live score visibly updates without a manual reload.

    Tries real live scores first (ingest/live_scores.py -- a deliberate,
    narrow exception to football-data.org's "never call this per page
    render" rule: one short-TTL, disk-cached, global call shared across
    every concurrent viewer). Falls back to the older kickoff-window guess
    (team names only, no score) on any failure or when the real-score
    fetch simply has nothing to report -- never silently disappears.
    """
    with session_scope() as session:
        live_matches = fetch_all_live_matches(session, leagues)
        if live_matches:
            render_live_scores_banner(live_matches, leagues)
            return

        live_df = live_fixtures_across_leagues(
            session, dt.datetime.now(dt.UTC).replace(tzinfo=None), LIVE_MATCH_WINDOW
        )
        if live_df.empty:
            return
        live_lines = []
        for row in live_df.itertuples(index=False):
            home_team = team_by_id(session, row.home_team_id)
            away_team = team_by_id(session, row.away_team_id)
            if home_team is None or away_team is None:
                continue
            row_league = leagues.get(row.league_code)
            flag = f"{row_league.flag_emoji} " if row_league and row_league.flag_emoji else ""
            live_lines.append(f"{flag}{home_team.canonical_name} vs {away_team.canonical_name}")
        if live_lines:
            st.caption("🔴 **LIVE NOW** · " + "  ·  ".join(live_lines))


def render() -> None:
    st.title("⚽ Leagues")

    # Streamlit forbids writing to a widget's session_state key in the same
    # run after that widget has already been instantiated -- so a stale
    # selection (from the click that sent us to Team Detail) can only be
    # cleared here, before any st.dataframe(key=...) below runs. Clears
    # every possible table key up front (harmless for ones not actually
    # rendered this run) rather than tracking which one was active.
    if st.session_state.pop(_CLEAR_SELECTION_FLAG, False):
        for key in (RANKINGS_TABLE_KEY, *_CONFERENCE_TABLE_KEYS.values()):
            st.session_state[key] = {"selection": {"rows": []}}

    leagues = load_leagues()
    if not leagues:
        st.error("No leagues configured in config/leagues.yaml.")
        st.stop()

    query_league = st.query_params.get("league")
    st.session_state.setdefault(
        "home_league", query_league if query_league in leagues else next(iter(leagues))
    )

    top_cols = st.columns([2, 1])
    code = top_cols[0].selectbox(
        "League",
        options=list(leagues.keys()),
        format_func=lambda c: league_option_label(leagues[c]),
        key="home_league",
    )
    st.query_params["league"] = code
    league = leagues[code]

    season_options = list(reversed(league.seasons))
    st.session_state.setdefault("home_season", season_options[0])
    if st.session_state["home_season"] not in season_options:
        st.session_state["home_season"] = season_options[0]
    season = top_cols[1].selectbox(
        "Season", options=season_options, format_func=lambda s: _season_label(s, league), key="home_season"
    )

    # emblem_url comes straight from football-data.org's API response, not
    # our own static config -- html.escape() here (and, defensively, on the
    # otherwise-static league.flag_url/name too) so a crafted/corrupted
    # upstream value can't break out of the src="..." attribute and inject
    # arbitrary HTML into the page.
    emblem_url = fetch_competition_emblem(league)
    emblem_html = (
        f'<img src="{html.escape(emblem_url)}" style="height:40px;width:auto;">'
        if emblem_url
        else '<span style="font-size:2rem;line-height:1;">⚽</span>'
    )
    flag_html = (
        f'<img src="{html.escape(league.flag_url)}" style="height:26px;width:auto;border-radius:2px;">'
        if league.flag_url
        else ""
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:0.25rem 0 0.75rem;">'
        f"{emblem_html}{flag_html}"
        f'<span style="font-size:1.75rem;font-weight:700;line-height:1;">{html.escape(league.name)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    _render_live_banner(leagues)

    today = dt.date.today()
    with session_scope() as session:
        team_names = teams_for_league(session, league.code)
        matches = matches_for_league(session, league.code)
        crest_urls = team_crests_for_league(session, league.code)
        conference_by_team = team_conferences_for_league(session, league.code)
        next_opponent_ids = next_fixture_per_team(session, league.code, today)
        params = load_latest_params(session, league.code)
        upcoming_fixtures_df = fixtures_for_league(
            session, league.code, today, today + dt.timedelta(days=UPCOMING_FIXTURES_WINDOW_DAYS)
        )

    if not team_names:
        st.warning(
            f"No teams loaded for {league.name} yet. Run "
            f"`uv run python scripts/fetch_historical_data.py {league.code}` first."
        )
        st.stop()

    next_opponent_names = {
        team_id: team_names.get(opponent_id, f"team#{opponent_id}")
        for team_id, opponent_id in next_opponent_ids.items()
    }

    render_news_section(
        league_news_query(league),
        f"league_{league.code}",
        {team_names[team_id]: url for team_id, url in crest_urls.items() if team_id in team_names},
    )

    with st.expander("Upcoming Matches", expanded=True, key=f"upcoming_fixtures_expander_{league.code}"):
        if upcoming_fixtures_df.empty:
            if season_not_yet_scheduled(league, season):
                st.info(f"{league.name} {season} hasn't been scheduled yet - check back closer to the tournament.")
            elif season_already_concluded(league, season):
                st.caption(f"{league.name} {season} has already concluded - no upcoming matches.")
            else:
                st.info(
                    f"No upcoming Matches loaded for {league.name}. Run "
                    f"`uv run python scripts/refresh_live_data.py {league.code}`"
                    f"{live_sync_requirement_note(league)} to pull them. "
                    "(This shows the schedule, not live in-play scores - that would need a separate real-time feed.)"
                )
        else:
            tz = timezone_selector()
            fixture_rows = []
            fixture_meta = []  # parallel to fixture_rows: (home_id, away_id, home_name, away_name, date)
            with session_scope() as session:
                for row in upcoming_fixtures_df.sort_values("date").itertuples(index=False):
                    home_name = team_names.get(row.home_team_id, f"team#{row.home_team_id}")
                    away_name = team_names.get(row.away_team_id, f"team#{row.away_team_id}")
                    if params is not None:
                        prediction = predict_fixture(
                            session,
                            params,
                            row.home_team_id,
                            row.away_team_id,
                            home_name,
                            away_name,
                            fixture_date=row.date,
                        )
                        fixture_row = fixture_prediction_row(home_name, away_name, prediction)
                    else:
                        fixture_row = {"Home": home_name, "Away": away_name}
                    fixture_row["Kickoff"] = format_kickoff(row.kickoff_utc, row.date, tz)
                    fixture_rows.append(fixture_row)
                    fixture_meta.append((row.home_team_id, row.away_team_id, home_name, away_name, row.date))

            fixture_columns = fixture_columns_with_kickoff_first(fixture_rows[0])
            fixtures_height = 35 * (len(fixture_rows) + 1) + 3
            has_predictions = HOME_WIN_COLUMN in fixture_rows[0]
            fixtures_event = st.dataframe(
                style_fixture_predictions(fixture_rows) if has_predictions else fixture_rows,
                use_container_width=True,
                hide_index=True,
                height=fixtures_height,
                column_order=fixture_columns,
                column_config=fixture_percent_column_config(),
                on_select="rerun" if has_predictions else "ignore",
                selection_mode="single-row",
                key=f"upcoming_fixtures_table_{league.code}",
            )
            if params is None:
                if league.supports_predictions:
                    st.caption(
                        f"Predictions unavailable - run `uv run python scripts/run_training.py {league.code}` "
                        "to fit the model."
                    )
                else:
                    st.caption(
                        "This competition has no historical match data to train a prediction "
                        "model from, so scores aren't predicted here - just the schedule."
                    )
            else:
                st.caption("Click a fixture's row to see the full Poisson/Dixon-Coles math behind its prediction.")
                selected = fixtures_event.selection.rows if fixtures_event and fixtures_event.selection else []
                if selected:
                    home_id, away_id, home_name, away_name, fixture_date = fixture_meta[selected[0]]

                    with session_scope() as session:
                        home_matches = matches_for_team(session, home_id)
                    h2h = compute_head_to_head(home_id, away_id, home_matches)
                    if h2h is not None:
                        render_head_to_head(h2h, home_name, away_name)
                    else:
                        st.caption(f"No history on record between {home_name} and {away_name} yet.")

                    with session_scope() as session:
                        prediction = predict_fixture(
                            session, params, home_id, away_id, home_name, away_name, fixture_date=fixture_date
                        )
                        render_prediction_breakdown(
                            prediction, home_name, away_name, session, league, home_id, away_id
                        )

    st.caption("Click a team's row for its schedule, recent results, and injuries.")
    if league.zones:
        legend = " · ".join(f"{ZONE_MARKER.get(z.kind, '')} {z.label}" for z in league.zones)
        st.caption(legend)

    standings = compute_standings(matches, team_names, season)

    # Split into per-conference tables only when every team in the current
    # standings has a known conference (e.g. MLS's Eastern/Western) -- a
    # partial split (some teams unassigned) would silently drop teams from
    # both tables, so this falls back to one flat table unless the data is
    # complete.
    has_full_conference_split = bool(conference_by_team) and all(
        s.team_id in conference_by_team for s in standings
    )

    if has_full_conference_split:
        for conference_name, table_key in _CONFERENCE_TABLE_KEYS.items():
            conference_standings = [s for s in standings if conference_by_team[s.team_id] == conference_name]
            if not conference_standings:
                continue
            st.markdown(f"**{conference_name}**")
            table_df = standings_dataframe(conference_standings, crest_urls, next_opponent_names, league.zones)
            _render_standings_table(table_df, table_key, f"standings_search_{league.code}_{table_key}", league.code)
    else:
        table_df = standings_dataframe(standings, crest_urls, next_opponent_names, league.zones)
        _render_standings_table(table_df, RANKINGS_TABLE_KEY, f"standings_search_{league.code}", league.code)
