"""One team's page: crest, current rating, upcoming schedule with
predictions, recent results, and injuries. Reached from League Detail.
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from soccer_predictor.config import load_leagues
from soccer_predictor.dashboard import navigation
from soccer_predictor.dashboard.components import (
    compute_top_scorer_and_assister,
    fixture_columns_with_kickoff_first,
    fixture_prediction_row,
    format_kickoff,
    live_sync_requirement_note,
    render_head_to_head,
    render_player_badges,
    render_prediction_breakdown,
    render_star_players,
    render_team_facts,
    render_team_rating_breakdown,
    style_fixture_predictions,
    style_match_results,
    timezone_selector,
    youtube_search_url,
)
from soccer_predictor.ingest.asa_client import ASA_AVAILABLE_SEASONS
from soccer_predictor.ingest.asa_client import fetch_team_season as fetch_asa_team_season
from soccer_predictor.ingest.player_importance import (
    resolve_current_attack_strength,
    resolve_team_goals_and_assists,
    resolve_team_historical_stats,
)
from soccer_predictor.ingest.squad import fetch_squad_for_team, fetch_team_info
from soccer_predictor.ingest.understat_client import (
    UNDERSTAT_AVAILABLE_SEASONS,
    UNDERSTAT_LEAGUE_SLUG,
    fetch_team_season,
)
from soccer_predictor.model.current_form_adjustment import adjust_for_current_attack_form
from soccer_predictor.model.injury_adjustment import adjust_strength
from soccer_predictor.model.team_facts import compute_head_to_head, compute_team_facts
from soccer_predictor.prediction.service import (
    get_injuries_for_team,
    load_latest_params,
    predict_fixture,
)
from soccer_predictor.storage.db import session_scope
from soccer_predictor.storage.repository import (
    fixtures_for_team,
    matches_for_team,
    team_by_id,
    teams_for_league,
)

# football-data.org's position strings, ordered goalkeeper-out rather than
# alphabetically, with a fallback for any value not seen before.
_POSITION_ORDER = {
    "Goalkeeper": 0,
    "Defence": 1,
    "Defender": 1,
    "Midfield": 2,
    "Midfielder": 2,
    "Offence": 3,
    "Attacker": 3,
    "Forward": 3,
}


def _season_label(season, single_year: bool = False) -> str:
    # 2024 -> "2024/25" -- but for a single_year league (e.g. World Cup,
    # Euro: the season code IS the year, see League.season_display), just
    # the plain year, unchanged: "1966" -> "1966", not "1966/67".
    if single_year:
        return str(season)
    return f"{season}/{str(season + 1)[2:]}"


def _parse_team_id() -> int | None:
    raw = st.query_params.get("team")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def render() -> None:
    team_id = _parse_team_id()

    with session_scope() as session:
        team = team_by_id(session, team_id) if team_id is not None else None
        # Extract plain values while the session is still open -- `team`
        # becomes a detached instance once this `with` block exits (the
        # session commits and closes), and touching its lazy attributes
        # afterward raises DetachedInstanceError.
        if team is not None:
            team_league_code = team.league_code
            team_canonical_name = team.canonical_name
            team_crest_url = team.crest_url
        else:
            team_league_code = team_canonical_name = team_crest_url = None

    if team is None:
        st.info("Pick a team from a league's page to see its schedule.")
        if st.button("← Back to Leagues"):
            st.switch_page(navigation.home_page())
        st.stop()

    leagues = load_leagues()
    league = leagues.get(team_league_code)

    if st.button(f"← Back to {league.name if league else team_league_code}"):
        st.switch_page(navigation.home_page(), query_params={"league": team_league_code})

    header_cols = st.columns([1, 6])
    if team_crest_url:
        header_cols[0].image(team_crest_url, width=64)
    else:
        header_cols[0].markdown("### ⚽")
    header_cols[1].title(team_canonical_name)
    if league:
        st.caption(league.name)

    with session_scope() as session:
        params = load_latest_params(session, team_league_code)
        team_names = teams_for_league(session, team_league_code)
        recent_df = matches_for_team(session, team_id)

    st.subheader("Team Info")
    if league is None:
        st.caption("Team info unavailable.")
    else:
        with session_scope() as session:
            team_info = fetch_team_info(session, league, team_id)

        if team_info is None:
            if league.data_source == "espn":
                st.caption(
                    "Club background (founded year, venue, head coach, etc.) isn't available "
                    "from this league's data source."
                )
            else:
                st.info(
                    "No team info loaded for this team yet. Requires "
                    "FOOTBALL_DATA_ORG_API_KEY in .env and this team to have appeared "
                    f"in a live sync (`uv run python scripts/refresh_live_data.py {team_league_code}`)."
                )
        else:
            info_lines = []
            if team_info.founded:
                info_lines.append(f"**Founded:** {team_info.founded}")
            if team_info.venue:
                info_lines.append(f"**Venue:** {team_info.venue}")
            if team_info.club_colors:
                info_lines.append(f"**Colors:** {team_info.club_colors}")
            if team_info.coach_name:
                coach = team_info.coach_name
                if team_info.coach_nationality:
                    coach += f" ({team_info.coach_nationality})"
                info_lines.append(f"**Head coach:** {coach}")
            if team_info.address:
                info_lines.append(f"**Address:** {team_info.address}")
            if team_info.website:
                info_lines.append(f"**Website:** [{team_info.website}]({team_info.website})")
            if info_lines:
                st.write("  \n".join(info_lines))
            else:
                st.caption("The API hasn't got any of these fields populated for this team.")

    # Not from the API -- computed straight from our own stored match
    # history, so it's shown regardless of whether the API-sourced info
    # above is available.
    team_facts = compute_team_facts(team_id, recent_df)
    if team_facts is not None:
        render_team_facts(team_facts, team_names, league.name if league else team_league_code)

    # Unlike before, a missing model no longer stops the whole page -- a
    # competition with no historical data source (League.supports_predictions
    # is False, e.g. UEFA Champions League/European Championship) will never
    # have one, but its squad/historical-stats/injuries sections below are
    # still perfectly useful without it.
    st.subheader("Current rating")
    if params is None:
        if league is not None and not league.supports_predictions:
            st.caption(
                "This competition has no historical match data to train a prediction model "
                "from - no rating or predicted scores, just the schedule and squad below."
            )
        else:
            st.warning(
                f"No trained model for this league yet. Run "
                f"`uv run python scripts/run_training.py {team_league_code}` first."
            )
    else:
        relative_xg_strength = resolve_current_attack_strength(team_canonical_name, team_league_code)
        current_attack = adjust_for_current_attack_form(params.attack[team_id], relative_xg_strength)
        if current_attack != params.attack[team_id]:
            attack_display = f"{params.attack[team_id]:.2f} → **{current_attack:.2f}** (current-form adjusted)"
        else:
            attack_display = f"**{current_attack:.2f}**"
        st.write(f"Attack {attack_display} · Defense **{params.defense[team_id]:.2f}** (lower is better)")
        render_team_rating_breakdown(params, team_id, team_canonical_name, recent_df)

    st.subheader("Upcoming schedule")
    start, end = dt.date.today(), dt.date.today() + dt.timedelta(days=60)
    with session_scope() as session:
        fixtures_df = fixtures_for_team(session, team_id, start, end)

    if fixtures_df.empty:
        if league is not None and league.season_display == "single_year":
            st.info(
                f"No scheduled fixtures in the next 60 days - {league.name} runs as a "
                "periodic tournament, not a continuous season, so this is expected between editions."
            )
        else:
            st.info(
                "No upcoming fixtures loaded. Run "
                f"`uv run python scripts/refresh_live_data.py {team_league_code}`"
                f"{live_sync_requirement_note(league)} to pull them."
            )
    else:
        tz = timezone_selector()
        rows = []
        row_meta = []  # parallel to rows: (home_id, away_id, home_name, away_name, date)
        with session_scope() as session:
            for row in fixtures_df.sort_values("date").itertuples(index=False):
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
                rows.append(fixture_row)
                row_meta.append((row.home_team_id, row.away_team_id, home_name, away_name, row.date))
        column_order = fixture_columns_with_kickoff_first(rows[0])
        has_predictions = "P(Home)" in rows[0]
        schedule_event = st.dataframe(
            style_fixture_predictions(rows) if has_predictions else rows,
            use_container_width=True,
            column_order=column_order,
            on_select="rerun" if has_predictions else "ignore",
            selection_mode="single-row",
            key=f"team_detail_schedule_table_{team_id}",
        )
        if has_predictions:
            st.caption("Click a fixture's row to see the full Poisson/Dixon-Coles math behind its prediction.")
            selected = schedule_event.selection.rows if schedule_event and schedule_event.selection else []
            if selected:
                sel_home_id, sel_away_id, sel_home_name, sel_away_name, sel_date = row_meta[selected[0]]
                is_this_team_home = sel_home_id == team_id
                opponent_id = sel_away_id if is_this_team_home else sel_home_id
                opponent_name = sel_away_name if is_this_team_home else sel_home_name

                h2h = compute_head_to_head(team_id, opponent_id, recent_df)
                if h2h is not None:
                    render_head_to_head(h2h, team_canonical_name, opponent_name)
                else:
                    st.caption(f"No history on record against {opponent_name} yet.")

                with session_scope() as session:
                    prediction = predict_fixture(
                        session, params, sel_home_id, sel_away_id, sel_home_name, sel_away_name, fixture_date=sel_date
                    )
                    render_prediction_breakdown(
                        prediction, sel_home_name, sel_away_name, session, league, sel_home_id, sel_away_id
                    )

    st.subheader("Recent results")
    if recent_df.empty:
        st.caption("No historical results on record for this team.")
    else:
        result_rows = []
        for row in recent_df.head(10).itertuples(index=False):
            is_home = row.home_team_id == team_id
            opponent_id = row.away_team_id if is_home else row.home_team_id
            opponent_name = team_names.get(opponent_id, f"team#{opponent_id}")
            goals_for = row.home_goals if is_home else row.away_goals
            goals_against = row.away_goals if is_home else row.home_goals
            outcome = "W" if goals_for > goals_against else ("D" if goals_for == goals_against else "L")
            home_name, away_name = (
                (team_canonical_name, opponent_name) if is_home else (opponent_name, team_canonical_name)
            )
            result_rows.append(
                {
                    "Date": row.date,
                    "Opponent": f"{'vs' if is_home else '@'} {opponent_name}",
                    "Score": f"{goals_for}-{goals_against}",
                    "Result": outcome,
                    "Watch": youtube_search_url(home_name, away_name, row.date),
                }
            )
        st.dataframe(
            style_match_results(result_rows),
            use_container_width=True,
            column_config={"Watch": st.column_config.LinkColumn("Watch", display_text="▶ Highlights")},
        )

    st.subheader("Squad")
    if league is None:
        st.caption("Squad unavailable.")
    else:
        with session_scope() as session:
            squad_players = fetch_squad_for_team(session, league, team_id)

        if not squad_players:
            st.info(
                "No squad data loaded for this team yet. Run "
                f"`uv run python scripts/refresh_live_data.py {team_league_code}`"
                f"{live_sync_requirement_note(league)} to pull it."
            )
        else:
            # Top-scorer/top-assister badges: fetched unconditionally (not
            # gated behind the "Player Stats (Historical)" section below
            # having been opened) -- unlike the old API-Football-backed
            # design this replaced, Understat and American Soccer Analysis
            # are both keyless with no daily cap, and both already disk-
            # cache for 24h, so a repeat page view costs nothing. No-ops
            # entirely for UCL/EURO/WC -- no stats source covers them.
            top_scorer: str | None = None
            top_assister: str | None = None
            if team_league_code in UNDERSTAT_LEAGUE_SLUG:
                understat_players = fetch_team_season(team_canonical_name, team_league_code, dt.date.today().year)
                if understat_players:
                    goals_and_assists = resolve_team_goals_and_assists(
                        squad_players, understat_players=understat_players
                    )
                    top_scorer, top_assister = compute_top_scorer_and_assister(goals_and_assists)
            elif team_league_code == "MLS":
                asa_players = fetch_asa_team_season(team_canonical_name, dt.date.today().year)
                if asa_players:
                    goals_and_assists = resolve_team_goals_and_assists(squad_players, asa_players=asa_players)
                    top_scorer, top_assister = compute_top_scorer_and_assister(goals_and_assists)

            today = dt.date.today()
            squad_rows = []
            for player in sorted(squad_players, key=lambda p: (_POSITION_ORDER.get(p.position, 99), p.name)):
                age = None
                if player.date_of_birth:
                    try:
                        dob = dt.date.fromisoformat(player.date_of_birth)
                        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    except ValueError:
                        age = None
                squad_rows.append(
                    {
                        "Name": render_player_badges(player.name, team_canonical_name, top_scorer, top_assister),
                        "Position": player.position,
                        "Nationality": player.nationality,
                        "Age": age,
                    }
                )
            # Same reasoning as the Leagues/Predictor tables' search: this
            # grid is a canvas, invisible to the browser's own Ctrl+F.
            squad_search = st.text_input(
                "Search roster", placeholder="e.g. Saka, Goalkeeper, Brazil", key=f"squad_search_{team_id}"
            )
            if squad_search:
                query = squad_search.lower()
                filtered_rows = [
                    r
                    for r in squad_rows
                    if query in (r["Name"] or "").lower()
                    or query in (r["Position"] or "").lower()
                    or query in (r["Nationality"] or "").lower()
                ]
            else:
                filtered_rows = squad_rows

            if squad_search and not filtered_rows:
                st.info(f'No player matches "{squad_search}".')
            else:
                squad_height = 35 * (len(filtered_rows) + 1) + 3
                st.dataframe(filtered_rows, use_container_width=True, hide_index=True, height=squad_height)

    st.subheader("Player Stats (Historical)")
    is_archive_league = team_league_code in ("WC", "EURO")
    if league is None:
        st.caption("Stats unavailable.")
    elif team_league_code not in UNDERSTAT_LEAGUE_SLUG and team_league_code != "MLS" and not is_archive_league:
        # No stats source at all for this competition (currently just UCL)
        # -- Understat only covers 5 domestic leagues, American Soccer
        # Analysis only MLS, the World Cup/Euro archives only those two.
        # API-Football used to fill this gap but that account is gone, not
        # just suspended -- nothing left to try.
        st.caption("No historical player-stats source is available for this competition.")
    else:
        # Gated behind an explicit click -- keyless, but still a real fetch
        # for a whole team's roster, kept opt-in rather than automatic. The
        # flag is keyed per-team so leaving and coming back to a *different*
        # team doesn't inherit "already loaded" from the last one.
        stats_shown_key = f"show_historical_stats_{team_id}"
        stats_shown = st.session_state.get(stats_shown_key, False)
        if team_league_code in UNDERSTAT_LEAGUE_SLUG:
            season_options = UNDERSTAT_AVAILABLE_SEASONS
        elif team_league_code == "MLS":
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
            if st.button("Load player stats", key=f"load_stats_btn_{team_id}"):
                st.session_state[stats_shown_key] = True
                stats_shown = True

        if stats_shown:
            hide_col, season_col = st.columns([1, 3])
            if hide_col.button("Hide", key=f"hide_stats_btn_{team_id}"):
                st.session_state[stats_shown_key] = False
                st.rerun()
            season = season_col.selectbox(
                "Season",
                options=list(reversed(season_options)),
                format_func=lambda s: _season_label(s, single_year=is_archive_league),
                key="team_detail_stats_season",
            )
            with session_scope() as session:
                source, historical_players = resolve_team_historical_stats(
                    team_canonical_name, team_league_code, season, session=session
                )

            if not historical_players:
                source_name = {"asa": "American Soccer Analysis", "archive": "the historical archive"}.get(
                    source, "Understat"
                )
                st.info(
                    f"No stats found for this team/season "
                    f"({_season_label(season, single_year=is_archive_league)}) from {source_name} - this "
                    "team may not be resolvable against its team list for that season."
                )
            elif source == "archive":
                stats_rows = [
                    {"Name": p.name, "Goals": p.goals, "Own goals": p.own_goals}
                    for p in sorted(historical_players, key=lambda p: (-p.goals, p.name))
                ]
                stats_height = 35 * (len(stats_rows) + 1) + 3
                st.dataframe(stats_rows, use_container_width=True, hide_index=True, height=stats_height)
            elif source == "understat":
                st.caption("Understat - current-season-capable, but no saves/tackles/rating (goalkeeper stats).")
                render_star_players(historical_players)
                stats_rows = [
                    {
                        "Name": p.name,
                        "Position": p.position,
                        "Apps": p.appearances,
                        "Minutes": p.minutes,
                        "Goals": p.goals,
                        "npG": p.non_penalty_goals,
                        "Assists": p.assists,
                        "Shots": p.shots,
                        "Key passes": p.key_passes,
                        "xG": round(p.xg, 2),
                        "npxG": round(p.non_penalty_xg, 2),
                        "xA": round(p.xa, 2),
                        "Yellow": p.yellow_cards,
                        "Red": p.red_cards,
                    }
                    for p in sorted(historical_players, key=lambda p: p.name)
                ]
                stats_height = 35 * (len(stats_rows) + 1) + 3
                st.dataframe(stats_rows, use_container_width=True, hide_index=True, height=stats_height)
            else:  # source == "asa"
                st.caption(
                    "American Soccer Analysis - current-season-capable, but no "
                    "appearances/cards/rating data available."
                )
                render_star_players(historical_players)
                stats_rows = [
                    {
                        "Name": p.name,
                        "Position": p.position,
                        "Minutes": p.minutes,
                        "Goals": p.goals,
                        "Assists": p.assists,
                        "Shots": p.shots,
                        "Key passes": p.key_passes,
                        "xG": round(p.xg, 2),
                        "xA": round(p.xa, 2),
                        "Points added": round(p.points_added, 2),
                    }
                    for p in sorted(historical_players, key=lambda p: p.name)
                ]
                stats_height = 35 * (len(stats_rows) + 1) + 3
                st.dataframe(stats_rows, use_container_width=True, hide_index=True, height=stats_height)

    st.subheader("Injuries")
    with session_scope() as session:
        team_injuries = get_injuries_for_team(session, team_id, team_canonical_name, dt.date.today())

    if not team_injuries.entries:
        st.caption("No injuries on record.")
    else:
        for entry, source, return_date in zip(
            team_injuries.entries, team_injuries.sources, team_injuries.expected_return_dates
        ):
            return_note = f", est. return {return_date}" if return_date else ""
            st.write(
                f"- {entry.player_name} ({entry.position}, weight={entry.importance_weight:.2f}) "
                f"- source: {source}{return_note}"
            )
        if params is not None:
            adj_attack, adj_defense = adjust_strength(
                params.attack[team_id], params.defense[team_id], team_injuries.entries
            )
            st.caption(
                f"Attack {params.attack[team_id]:.2f} → {adj_attack:.2f}, "
                f"Defense {params.defense[team_id]:.2f} → {adj_defense:.2f}"
            )
