"""Leagues page: a selector at the top switches between leagues; a real
league table (Played/W/D/L/GF/GA/GD/Pts/Form/Next) for the selected season
is shown directly below -- straight from actual results, independent of the
Dixon-Coles model. Clicking a team's row drills into Team Detail, where the
model's attack/defense rating lives instead.
"""

from __future__ import annotations

import datetime as dt
import html
import re

import streamlit as st

from soccer_predictor.config import League, load_leagues
from soccer_predictor.dashboard import navigation
from soccer_predictor.dashboard.components import (
    FIXTURE_PARAM,
    LIVE_MATCH_WINDOW,
    ZONE_MARKER,
    FixtureRef,
    PREDICTION_STATUS_NOTE,
    fixture_prediction_row,
    fixtures_table_html,
    format_kickoff,
    league_option_label,
    live_sync_requirement_note,
    next_match_labels,
    next_match_links,
    parse_fixture_key,
    render_live_scores_banner,
    render_news_section,
    season_already_concluded,
    season_not_yet_scheduled,
    selected_timezone,
    sort_fixtures_by_kickoff,
    inject_standings_sort,
    standings_table_html,
    timezone_selector,
)
from soccer_predictor.dashboard.fixture_workings import render_fixture_workings
from soccer_predictor.ingest.league_meta import fetch_competition_emblem
from soccer_predictor.ingest.live_scores import LiveMatch, fetch_all_live_matches
from soccer_predictor.ingest.news import league_news_query
from soccer_predictor.model.standings import compute_group_standings, compute_standings
from soccer_predictor.prediction.service import load_latest_params, predict_fixture
from soccer_predictor.storage.db import session_scope
from soccer_predictor.storage.repository import (
    fixture_groups_for_league,
    fixtures_for_league,
    live_fixtures_across_leagues,
    locked_predictions_for_league,
    matches_for_league,
    team_by_id,
    team_conferences_for_league,
    team_crests_for_league,
    teams_for_league,
)

# The view toggle shown beside Season for group competitions.
GROUPS_VIEW = "Groups"
RANKINGS_VIEW = "Rankings"
_CLEAR_SELECTION_FLAG = "_clear_rankings_selection"
UPCOMING_FIXTURES_WINDOW_DAYS = 7


def _season_label(season: str, league) -> str:
    # "2425" -> "2024/25" (the usual case); a single_year league (e.g. the
    # Euros) shows its season code as-is since it's already just the year.
    if league.season_display == "single_year":
        return season
    return f"20{season[:2]}/{season[2:]}"


def _render_standings_table(
    standings: list,
    crest_urls: dict,
    next_labels: dict,
    zones: list,
    match_links: dict,
    search_key: str,
    league_code: str,
    season: str = "",
    tz_name: str = "",
) -> None:
    """One searchable standings table -- shared by the flat single-table case
    and each per-group / per-conference table (see render()). Hovering a form
    square shows the game behind it; a team's name opens its page.
    """
    search_query = st.text_input("Search team", placeholder="e.g. Man City", key=search_key)
    if search_query and not any(search_query.casefold() in s.team_name.casefold() for s in standings):
        st.info(f'No team matches "{search_query}".')
        return
    st.markdown(
        standings_table_html(
            standings,
            crest_urls,
            next_labels,
            zones,
            match_links=match_links,
            league_code=league_code,
            name_filter=search_query,
            season=season,
            tz_name=tz_name,
        ),
        unsafe_allow_html=True,
    )


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
        fallback_matches = []
        for row in live_df.itertuples(index=False):
            home_team = team_by_id(session, row.home_team_id)
            away_team = team_by_id(session, row.away_team_id)
            if home_team is None or away_team is None:
                continue
            fallback_matches.append(
                LiveMatch(
                    league_code=row.league_code,
                    home_name=home_team.canonical_name,
                    away_name=away_team.canonical_name,
                    home_crest=None,
                    away_crest=None,
                    home_score=None,
                    away_score=None,
                    clock_label="",
                    match_date=row.kickoff_utc.date(),
                )
            )
        # Same layout and Google links as the real banner, just with no score.
        render_live_scores_banner(fallback_matches, leagues)


def render() -> None:
    st.title("⚽ Leagues")

    leagues = load_leagues()
    if not leagues:
        st.error("No leagues configured in config/leagues.yaml.")
        st.stop()

    query_league = st.query_params.get("league")
    st.session_state.setdefault(
        "home_league", query_league if query_league in leagues else next(iter(leagues))
    )

    # A group competition (Nations League, World Cup, Euros) gets a third
    # control beside Season to flip between its group tables and one overall
    # ranking. Decided from the league already in session state, since the
    # columns have to exist before the League picker below is drawn.
    selected_league = leagues.get(st.session_state["home_league"])
    has_view_toggle = selected_league is not None and selected_league.has_groups
    top_cols = st.columns([2, 1, 1] if has_view_toggle else [2, 1])
    code = top_cols[0].selectbox(
        "League",
        options=list(leagues.keys()),
        format_func=lambda c: league_option_label(leagues[c]),
        key="home_league",
    )
    st.query_params["league"] = code
    league = leagues[code]

    season_options = list(reversed(league.seasons))
    # A team link in the standings table reloads the page (new session), so the
    # season being viewed travels in the URL and is restored from it here.
    query_season = st.query_params.get("season")
    st.session_state.setdefault("home_season", query_season if query_season in season_options else season_options[0])
    if st.session_state["home_season"] not in season_options:
        st.session_state["home_season"] = season_options[0]
    season = top_cols[1].selectbox(
        "Season", options=season_options, format_func=lambda s: _season_label(s, league), key="home_season"
    )
    st.query_params["season"] = season
    show_rankings = False
    if has_view_toggle:
        view = top_cols[2].segmented_control(
            "View", options=[GROUPS_VIEW, RANKINGS_VIEW], default=GROUPS_VIEW, key="home_table_view"
        )
        # Clicking the selected option again clears it -- treat that as Groups.
        show_rankings = view == RANKINGS_VIEW

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
        # Fixtures carry no season of their own, so their groups only apply
        # to the league's latest edition (what any stored fixture is for).
        fixture_groups_df = (
            fixture_groups_for_league(session, league.code)
            if league.has_groups and season == league.seasons[-1]
            else None
        )
        # From yesterday so a match still in progress after UTC midnight is
        # still found (next_match_labels drops anything already finished).
        next_fixtures_df = fixtures_for_league(
            session, league.code, today - dt.timedelta(days=1), today + dt.timedelta(days=365)
        )
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

    # Stored fixtures are always the league's CURRENT edition (they carry no season
    # of their own), so on an older season's table they would show next to teams
    # that played that edition years ago -- e.g. Qatar's 2027 opener on the 2023
    # Asian Cup table. Next and Search belong to the latest season only.
    if season == league.seasons[-1]:
        next_opponent_names = next_match_labels(next_fixtures_df, team_names, selected_timezone())
        match_links = next_match_links(next_fixtures_df, team_names, league.name)
    else:
        next_opponent_names, match_links = {}, {}

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
            fixture_refs: list[FixtureRef] = []  # parallel to fixture_rows
            with session_scope() as session:
                locked_predictions = locked_predictions_for_league(session, league.code)
                for row in sort_fixtures_by_kickoff(upcoming_fixtures_df).itertuples(index=False):
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
                        fixture_row = fixture_prediction_row(
                            home_name,
                            away_name,
                            prediction,
                            locked_predictions.get((row.date, row.home_team_id, row.away_team_id)),
                        )
                    else:
                        fixture_row = {"Home": home_name, "Away": away_name}
                    fixture_row["Kickoff"] = format_kickoff(row.kickoff_utc, row.date, tz)
                    fixture_rows.append(fixture_row)
                    fixture_refs.append(FixtureRef(row.home_team_id, row.away_team_id, home_name, away_name, row.date))

            # The match whose workings are open: named in the URL by the predicted-score
            # link, and only ever one of the rows shown (anything else is dropped).
            selected_index = None
            if params is not None and st.query_params.get(FIXTURE_PARAM) is not None:
                wanted = parse_fixture_key(st.query_params[FIXTURE_PARAM])
                selected_index = next(
                    (i for i, ref in enumerate(fixture_refs) if wanted == (ref.home_id, ref.away_id, ref.date)), None
                )
                if selected_index is None:
                    del st.query_params[FIXTURE_PARAM]

            st.markdown(
                fixtures_table_html(
                    fixture_rows,
                    fixture_refs,
                    league.code,
                    season=season,
                    tz_name=tz.key,
                    match_details=bool(league.espn_league_slug),
                    selected=selected_index,
                ),
                unsafe_allow_html=True,
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
                st.caption(
                    "Click a team's name for its page, the kickoff for the match details, or the predicted "
                    "score to see how it was calculated. " + PREDICTION_STATUS_NOTE
                )
                if selected_index is not None:
                    render_fixture_workings(
                        fixture_refs[selected_index], params, league, f"close_fixture_breakdown_{league.code}"
                    )

    st.caption("Click a team's name for its schedule, recent results, and injuries. Hover a form square for the game.")
    if league.zones:
        legend = " · ".join(f"{ZONE_MARKER.get(z.kind, '')} {z.label}" for z in league.zones)
        st.caption(legend)

    # One table per section: each group of a group competition (Nations
    # League, World Cup, Euros) when this season has group data, else each
    # conference of a league with a full conference split (MLS), else a
    # single flat table.
    sections: list[tuple[str | None, list]] = []
    group_standings = (
        compute_group_standings(matches, team_names, season, fixture_groups=fixture_groups_df)
        if league.has_groups
        else {}
    )
    if group_standings and not show_rankings:
        sections = list(group_standings.items())
    else:
        # Also what the Rankings view shows for a group competition: every
        # match of the season in one overall table.
        standings = compute_standings(matches, team_names, season)
        # Split into per-conference tables only when every team in the
        # current standings has a known conference (e.g. MLS's
        # Eastern/Western) -- a partial split (some teams unassigned) would
        # silently drop teams from both tables, so this falls back to one
        # flat table unless the data is complete.
        has_full_conference_split = bool(conference_by_team) and all(
            s.team_id in conference_by_team for s in standings
        )
        if has_full_conference_split:
            for conference_name in sorted({conference_by_team[s.team_id] for s in standings}):
                sections.append(
                    (conference_name, [s for s in standings if conference_by_team[s.team_id] == conference_name])
                )
        else:
            sections = [(None, standings)]

    inject_standings_sort()
    for title, section_standings in sections:
        if title is None:
            search_key = f"standings_search_{league.code}"
        else:
            search_key = f"standings_search_{league.code}_{re.sub('[^a-z0-9]+', '_', title.casefold()).strip('_')}"
            st.markdown(f"**{title}**")
        _render_standings_table(
            section_standings,
            crest_urls,
            next_opponent_names,
            league.zones,
            match_links,
            search_key,
            league.code,
            season,
            selected_timezone().key,
        )
