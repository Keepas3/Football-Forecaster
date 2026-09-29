"""Players page: search for an individual player by name across every team
in a league, then see their current club plus (opt-in, quota-conscious)
stats across past seasons. Reached from the sidebar nav.

Every fetch here reuses existing team-scoped ingest functions (squad.py,
understat_client.py, asa_client.py) -- there is no per-player database
table anywhere in this app; everything is fetched live and disk-cached by
those clients, same as Team Detail's own squad/player-stats sections.
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from soccer_predictor.config import load_leagues, load_manual_captains, load_manual_star_players
from soccer_predictor.dashboard.components import league_option_label
from soccer_predictor.ingest import understat_client
from soccer_predictor.ingest.asa_client import ASA_AVAILABLE_SEASONS
from soccer_predictor.ingest.asa_client import fetch_team_season as fetch_asa_team_season
from soccer_predictor.ingest.player_importance import (
    find_archive_player,
    find_asa_player,
    find_understat_player,
    resolve_team_historical_stats,
)
from soccer_predictor.ingest.squad import PlayerSearchResult, search_players_in_league
from soccer_predictor.storage.db import session_scope
from soccer_predictor.storage.repository import team_crests_for_league


def _season_label(season, single_year: bool = False) -> str:
    # 2024 -> "2024/25" -- duplicated from team_detail.py's own private
    # one-liner rather than shared, consistent with this codebase's existing
    # tolerance for small view-local formatting helpers (home.py has its
    # own separately-defined _season_label too, with a different signature).
    # single_year=True (World Cup/Euro -- the season code IS the year, see
    # League.season_display) just returns the plain year unchanged.
    if single_year:
        return str(season)
    return f"{season}/{str(season + 1)[2:]}"


def _age_from_date_of_birth(date_of_birth: str | None) -> int | None:
    if not date_of_birth:
        return None
    try:
        dob = dt.date.fromisoformat(date_of_birth)
    except ValueError:
        return None
    today = dt.date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def render() -> None:
    leagues = load_leagues()
    if not leagues:
        st.error("No leagues configured in config/leagues.yaml.")
        st.stop()

    query_league = st.query_params.get("league")
    st.session_state.setdefault(
        "players_league", query_league if query_league in leagues else next(iter(leagues))
    )

    with st.sidebar:
        st.header("Filters")
        league_code = st.selectbox(
            "League",
            options=list(leagues.keys()),
            format_func=lambda c: league_option_label(leagues[c]),
            key="players_league",
        )

    league = leagues[league_code]
    st.title(f"{league.name} Players")

    selected_key = f"players_selected_{league.code}"

    search_query = st.text_input("Search player name", placeholder="e.g. Salah")
    if search_query:
        with session_scope() as session:
            results = search_players_in_league(session, league, search_query)

        if not results:
            st.info(f'No player matches "{search_query}".')
        else:
            result_rows = [
                {"Name": r.player.name, "Position": r.player.position, "Club": r.team_name} for r in results
            ]
            height = 35 * (len(result_rows) + 1) + 3
            event = st.dataframe(
                result_rows,
                use_container_width=True,
                hide_index=True,
                height=height,
                on_select="rerun",
                selection_mode="single-row",
                key=f"players_search_results_{league.code}",
            )
            selected_rows = event.selection.rows if event and event.selection else []
            if selected_rows:
                st.session_state[selected_key] = results[selected_rows[0]]
    else:
        st.caption("Type a player name to search across every team in this league.")

    selection: PlayerSearchResult | None = st.session_state.get(selected_key)
    if selection is None:
        return

    player = selection.player
    st.divider()
    header_cols = st.columns([1, 4])
    with session_scope() as session:
        crest_url = team_crests_for_league(session, league.code).get(selection.team_id, "")
    if crest_url:
        header_cols[0].image(crest_url, width=64)
    with header_cols[1]:
        # Only the manual captain/star lists are checked here (not the
        # stats-based automatic signal) -- that would mean re-fetching a
        # whole team's stats just for this one player; Team Detail's squad
        # list already covers the automatic case for every player at once.
        is_captain = any(
            c.team == selection.team_name and c.player.lower() == player.name.lower()
            for c in load_manual_captains()
        )
        is_star = any(
            s.team == selection.team_name and s.player.lower() == player.name.lower()
            for s in load_manual_star_players()
        )
        st.subheader(f"👑 {player.name}" if is_captain else player.name)
        if is_star:
            st.caption("⭐ World class")
        age = _age_from_date_of_birth(player.date_of_birth)
        detail_bits = [player.position]
        if player.nationality:
            detail_bits.append(player.nationality)
        if age is not None:
            detail_bits.append(f"Age {age}")
        detail_bits.append(selection.team_name)
        st.caption(" · ".join(detail_bits))

    st.markdown("**Current season**")
    if league.code in understat_client.UNDERSTAT_LEAGUE_SLUG:
        current_season_players = understat_client.fetch_team_season(
            selection.team_name, league.code, dt.date.today().year
        )
        current_season_row = find_understat_player(player.name, current_season_players)
        current_season_source = "Understat"
    elif league.code == "MLS":
        current_season_players = fetch_asa_team_season(selection.team_name, dt.date.today().year)
        current_season_row = find_asa_player(player.name, current_season_players)
        current_season_source = "American Soccer Analysis"
    else:
        current_season_row = None
        current_season_source = None

    if current_season_row is None:
        if current_season_source is None:
            st.caption("No current-season stats source is available for this competition.")
        else:
            st.caption(
                f"No current-season data available from {current_season_source} - this player "
                "may not be matched in it."
            )
    else:
        stat_cols = st.columns(4)
        stat_cols[0].metric("Goals", current_season_row.goals)
        stat_cols[1].metric("Assists", current_season_row.assists)
        stat_cols[2].metric("xG", f"{current_season_row.xg:.2f}")
        stat_cols[3].metric("xA", f"{current_season_row.xa:.2f}")
        st.caption(f"{current_season_row.minutes} minutes played this season ({current_season_source}).")

    st.markdown("**Player Stats (Historical)**")
    is_archive_league = league.code in ("WC", "EURO")
    if league.code not in understat_client.UNDERSTAT_LEAGUE_SLUG and league.code != "MLS" and not is_archive_league:
        # No stats source at all for this competition (currently just UCL)
        # -- see team_detail.py's matching branch for why.
        st.caption("No historical player-stats source is available for this competition.")
        return

    # Gated behind an explicit click -- keyless, but still a real fetch for
    # a whole team's roster, kept opt-in rather than automatic, same
    # reasoning as Team Detail's own historical stats section.
    stats_shown_key = f"players_show_historical_{league.code}_{selection.team_id}_{player.name}"
    stats_shown = st.session_state.get(stats_shown_key, False)
    if league.code in understat_client.UNDERSTAT_LEAGUE_SLUG:
        season_options = understat_client.UNDERSTAT_AVAILABLE_SEASONS
    elif league.code == "MLS":
        season_options = ASA_AVAILABLE_SEASONS
    else:
        # WC/EURO: real tournament years, not a live source's own
        # available-seasons range.
        season_options = league.seasons

    if not stats_shown:
        source_caption = (
            "Sourced from a static historical archive (goals only)."
            if is_archive_league
            else "Sourced from Understat or American Soccer Analysis (keyless, any season)."
        )
        st.caption(source_caption)
        if st.button("Load player stats", key=f"players_load_stats_btn_{selection.team_id}_{player.name}"):
            st.session_state[stats_shown_key] = True
            stats_shown = True

    if stats_shown:
        hide_col, season_col = st.columns([1, 3])
        if hide_col.button("Hide", key=f"players_hide_stats_btn_{selection.team_id}_{player.name}"):
            st.session_state[stats_shown_key] = False
            st.rerun()
        season = season_col.selectbox(
            "Season",
            options=list(reversed(season_options)),
            format_func=lambda s: _season_label(s, single_year=is_archive_league),
            key="players_stats_season",
        )
        with session_scope() as session:
            source, team_historical_players = resolve_team_historical_stats(
                selection.team_name, league.code, season, session=session
            )

        if source == "understat":
            historical_row = find_understat_player(player.name, team_historical_players)
        elif source == "asa":
            historical_row = find_asa_player(player.name, team_historical_players)
        else:  # source == "archive"
            historical_row = find_archive_player(player.name, team_historical_players)

        if historical_row is None:
            source_name = {"asa": "American Soccer Analysis", "archive": "the historical archive"}.get(
                source, "Understat"
            )
            st.info(
                f"No stats found for this player/season "
                f"({_season_label(season, single_year=is_archive_league)}) from {source_name}."
            )
        elif source == "archive":
            stat_cols = st.columns(2)
            stat_cols[0].metric("Goals", historical_row.goals)
            stat_cols[1].metric("Own goals", historical_row.own_goals)
        elif source == "understat":
            st.caption("Understat - no saves/tackles/rating (goalkeeper stats) available.")
            stat_cols = st.columns(4)
            stat_cols[0].metric("Appearances", historical_row.appearances)
            stat_cols[1].metric("Goals", historical_row.goals)
            stat_cols[2].metric("Assists", historical_row.assists)
            stat_cols[3].metric("Shots", historical_row.shots)
            detail_cols = st.columns(4)
            detail_cols[0].metric("Key passes", historical_row.key_passes)
            detail_cols[1].metric("xG", f"{historical_row.xg:.2f}")
            detail_cols[2].metric("xA", f"{historical_row.xa:.2f}")
            detail_cols[3].metric("npxG", f"{historical_row.non_penalty_xg:.2f}")
            card_cols = st.columns(3)
            card_cols[0].metric("Non-penalty goals", historical_row.non_penalty_goals)
            card_cols[1].metric("Yellow cards", historical_row.yellow_cards)
            card_cols[2].metric("Red cards", historical_row.red_cards)
            st.caption(f"{historical_row.minutes} minutes played in {_season_label(season)}.")
        else:  # source == "asa"
            st.caption("American Soccer Analysis - no appearances/cards/rating data available.")
            stat_cols = st.columns(4)
            stat_cols[0].metric("Goals", historical_row.goals)
            stat_cols[1].metric("Assists", historical_row.assists)
            stat_cols[2].metric("Shots", historical_row.shots)
            stat_cols[3].metric("Key passes", historical_row.key_passes)
            detail_cols = st.columns(3)
            detail_cols[0].metric("xG", f"{historical_row.xg:.2f}")
            detail_cols[1].metric("xA", f"{historical_row.xa:.2f}")
            detail_cols[2].metric("Points added", f"{historical_row.points_added:.2f}")
            st.caption(f"{historical_row.minutes} minutes played in {_season_label(season)}.")
