"""The original single-league workflow: fixtures/leaderboard/injuries/notes
tabs, custom matchup, and the AI chat notes feature. Reached from the
sidebar nav; can also be pre-seeded with a league via ?league=<code>.
"""

from __future__ import annotations

import datetime as dt
import time
from uuid import uuid4

import streamlit as st

from soccer_predictor.ai.note_parser import MissingApiKey as NoteParserMissingApiKey
from soccer_predictor.ai.note_parser import NoteParsingError, parse_note
from soccer_predictor.config import load_leagues, notes_password
from soccer_predictor.dashboard import notes_access as na
from soccer_predictor.dashboard.note_format import describe_form_note, describe_injury
from soccer_predictor.dashboard.chat_notes import (
    add_reply,
    delete_exchange,
    new_message,
    replace_note_text,
)
from soccer_predictor.dashboard.components import (
    LEADERBOARD_DISPLAY_COLUMNS,
    fixture_columns_with_kickoff_first,
    fixture_prediction_row,
    format_kickoff,
    leaderboard_dataframe,
    league_option_label,
    live_sync_requirement_note,
    render_prediction_breakdown,
    scoreline_chart,
    style_fixture_predictions,
    timezone_selector,
)
from soccer_predictor.model.injury_adjustment import adjust_strength
from soccer_predictor.prediction.service import (
    get_injuries_for_team,
    load_latest_params,
    predict_fixture,
)
from soccer_predictor.prediction.tracking import compute_prediction_accuracy
from soccer_predictor.storage.db import session_scope
from soccer_predictor.storage.repository import (
    active_chat_injuries,
    active_form_notes,
    add_form_note,
    add_or_update_chat_injury,
    delete_form_note,
    delete_injury,
    fixtures_for_league,
    teams_for_league,
)


def _safe_index(options: list, value, default: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return default


def _render_active_notes(leagues: dict, can_delete: bool) -> None:
    """Saved chat-sourced notes that still count toward predictions, for EVERY
    league -- not just the one picked above. A note is saved against a team
    in one league (e.g. Poland in the Nations League), so listing only the
    selected league's teams made notes vanish whenever the picker was on a
    different league (it resets to the first league on revisiting the page).
    Anyone can read them; the Delete buttons only appear once the chat is
    unlocked, so a visitor can't remove the owner's notes."""
    st.subheader("Active chat-sourced notes")
    today = dt.date.today()

    with session_scope() as session:
        # {league_code: {team_name: ([injuries], [form notes])}}
        grouped: dict[str, dict[str, tuple[list, list]]] = {}
        for injury, team in active_chat_injuries(session, today):
            grouped.setdefault(team.league_code, {}).setdefault(team.canonical_name, ([], []))[0].append(injury)
        for note, team in active_form_notes(session, today):
            grouped.setdefault(team.league_code, {}).setdefault(team.canonical_name, ([], []))[1].append(note)

        league_order = [code for code in leagues if code in grouped] + sorted(
            code for code in grouped if code not in leagues
        )
        for league_code in league_order:
            st.markdown(f"##### {leagues[league_code].name if league_code in leagues else league_code}")
            for team_name in sorted(grouped[league_code]):
                chat_injuries, chat_forms = grouped[league_code][team_name]
                st.markdown(f"**{team_name}**")
                for row in chat_injuries:
                    cols = st.columns([5, 1])
                    headline, maths = describe_injury(
                        row.player_name, row.position, row.importance_weight, row.expected_return_date, today
                    )
                    cols[0].markdown(headline)
                    cols[0].caption(maths)
                    if can_delete and cols[1].button("Delete", key=f"del_injury_{row.id}"):
                        delete_injury(session, row.id)
                        st.rerun()
                for row in chat_forms:
                    cols = st.columns([5, 1])
                    headline, maths = describe_form_note(
                        row.summary, row.magnitude, row.affects, row.expires_on, today
                    )
                    cols[0].markdown(headline)
                    cols[0].caption(maths)
                    if can_delete and cols[1].button("Delete", key=f"del_form_{row.id}"):
                        delete_form_note(session, row.id)
                        st.rerun()

    if not grouped:
        st.caption("No active chat-sourced notes.")


def _render_notes_gate() -> bool:
    """Draws the lock/unlock control for the Notes chat and returns whether
    the chat may be used. Every note goes to the Anthropic API on the owner's
    key, so without NOTES_PASSWORD, and the right password in this session,
    visitors only see the saved notes (see dashboard/notes_access.py)."""
    password = notes_password()
    if password is None:
        st.info("Adding notes is turned off on this site (NOTES_PASSWORD isn't set). Saved notes are shown below.")
        return False

    state = st.session_state
    if na.is_unlocked(state):
        if st.button("🔒 Lock notes", key="notes_lock"):
            na.lock(state)
            st.rerun()
        return True

    st.caption(
        "🔒 Adding notes uses the site owner's AI credits, so it needs a password. "
        "Anyone can read the saved notes below."
    )
    if na.attempts_left(state) == 0:
        st.error("Too many wrong passwords in this session.")
        return False
    with st.form("notes_unlock_form"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        result = na.try_unlock(state, entered, password)
        if result == na.OK:
            st.rerun()
        time.sleep(1)  # slows down guessing
        if result == na.LOCKED_OUT:
            st.error("Too many wrong passwords in this session.")
        else:
            st.error(f"Incorrect password ({na.attempts_left(state)} attempt(s) left).")
    return False


def _run_note_parser(message_id: str, text: str, team_names: dict[int, str]) -> None:
    """Parses one note and records the outcome: review cards for whatever it
    found (tagged with the note's id, so deleting/editing the note can drop
    them) and an assistant reply -- including a failure message, so a parse
    error survives the page rerun instead of flashing and vanishing."""
    na.record_parse(st.session_state)
    with st.spinner("Parsing..."):
        try:
            result = parse_note(text, team_names, dt.date.today())
        except NoteParserMissingApiKey as exc:
            add_reply(st.session_state.chat_history, message_id, f"⚠️ {exc}")
            return
        except NoteParsingError as exc:
            add_reply(st.session_state.chat_history, message_id, f"⚠️ Couldn't parse that note: {exc}")
            return

    for injury in result.injuries:
        st.session_state.pending_cards.append(
            {"uid": uuid4().hex, "kind": "injury", "data": injury, "source_id": message_id}
        )
    for form in result.form_notes:
        st.session_state.pending_cards.append(
            {"uid": uuid4().hex, "kind": "form", "data": form, "source_id": message_id}
        )
    if result.injuries or result.form_notes:
        summary = (
            f"Found {len(result.injuries)} availability note(s) and "
            f"{len(result.form_notes)} form note(s) - review below before saving."
        )
    else:
        summary = "Didn't find anything extractable in that note."
    add_reply(st.session_state.chat_history, message_id, summary)


def _render_chat_history(team_names: dict[int, str]) -> None:
    """Every note and reply. Each note you typed has an Options (⋮) menu on its right
    with Edit (rewrite it and re-parse) and Delete (removes the note, the
    reply, and any review cards from it that are still unsaved -- notes you
    already saved are removed from "Active chat-sourced notes" below)."""
    for message in list(st.session_state.chat_history):
        message_id = message["id"]
        with st.chat_message(message["role"]):
            if message["role"] != "user":
                st.write(message["text"])
                continue

            if st.session_state.editing_message == message_id:
                new_text = st.text_area(
                    "Edit note", value=message["text"], key=f"edit_text_{message_id}", label_visibility="collapsed"
                )
                save_col, cancel_col, _ = st.columns([2, 1, 6])
                too_long = not na.note_length_ok(new_text)
                out_of_parses = na.parses_left(st.session_state) == 0
                if too_long:
                    st.caption(f"Too long (max {na.MAX_NOTE_LENGTH} characters).")
                if out_of_parses:
                    st.caption("This session has used all of its note parses.")
                if save_col.button(
                    "Save & re-parse",
                    key=f"edit_save_{message_id}",
                    disabled=not new_text.strip() or too_long or out_of_parses,
                ):
                    replace_note_text(
                        st.session_state.chat_history, st.session_state.pending_cards, message_id, new_text.strip()
                    )
                    st.session_state.editing_message = None
                    _run_note_parser(message_id, new_text.strip(), team_names)
                    st.rerun()
                if cancel_col.button("Cancel", key=f"edit_cancel_{message_id}"):
                    st.session_state.editing_message = None
                    st.rerun()
                continue

            text_col, menu_col = st.columns([6, 1])
            text_col.write(message["text"])
            with menu_col.popover("Options", icon=":material/more_vert:", use_container_width=True):
                if st.button("✏️ Edit", key=f"edit_{message_id}", use_container_width=True):
                    st.session_state.editing_message = message_id
                    st.rerun()
                if st.button("🗑️ Delete", key=f"delete_{message_id}", use_container_width=True):
                    delete_exchange(st.session_state.chat_history, st.session_state.pending_cards, message_id)
                    if st.session_state.editing_message == message_id:
                        st.session_state.editing_message = None
                    st.rerun()


def render() -> None:
    leagues = load_leagues()
    if not leagues:
        st.error("No leagues configured in config/leagues.yaml.")
        st.stop()

    # Nice-to-have continuity: pre-seed the league picker from a query param
    # set by the Leagues/League Detail pages, but only on first visit per
    # session -- a later revisit with a different ?league= won't re-sync,
    # so a manual selectbox change here is never silently overwritten.
    query_league = st.query_params.get("league")
    st.session_state.setdefault(
        "predictor_league", query_league if query_league in leagues else next(iter(leagues))
    )

    with st.sidebar:
        st.header("Filters")
        league_code = st.selectbox(
            "League",
            options=list(leagues.keys()),
            format_func=lambda c: league_option_label(leagues[c]),
            key="predictor_league",
        )
        date_range = st.date_input(
            "Fixture date range",
            value=(dt.date.today(), dt.date.today() + dt.timedelta(days=14)),
        )

        st.divider()
        st.header("Track Record")
        st.caption("Across every league - how often a locked-in prediction matched the real result.")
        with session_scope() as session:
            accuracy = compute_prediction_accuracy(session)
        # Stacked, not side-by-side columns -- the sidebar is too narrow for
        # 3 st.metric labels to fit without truncating.
        st.metric("Exact score", accuracy.exact)
        st.metric("Right outcome", accuracy.correct_outcome)
        st.metric("Wrong", accuracy.wrong)
        if accuracy.hit_rate is None:
            st.caption("No graded predictions yet.")
        else:
            st.caption(
                f"{accuracy.hit_rate:.0%} hit rate ({accuracy.graded_total} graded, "
                f"{accuracy.pending} still pending)."
            )

    league = leagues[league_code]
    st.title(f"{league.name} Match Predictor")

    with session_scope() as session:
        params = load_latest_params(session, league.code)
        team_names = teams_for_league(session, league.code)

    if params is None:
        if not league.supports_predictions:
            st.warning(
                f"{league.name} has no historical match data to train a prediction model from, "
                "so this predictions-focused page doesn't apply to it - see its Leagues page "
                "for standings/fixtures/squad instead."
            )
        else:
            st.warning(
                f"No trained model for {league.name} yet. Run "
                f"`uv run python scripts/run_training.py {league.code}` first."
            )
        st.stop()

    id_by_name = {name: team_id for team_id, name in team_names.items()}

    fixtures_tab, leaderboard_tab, injuries_tab, notes_tab = st.tabs(
        ["Upcoming Fixtures", "Team Strength Leaderboard", "Injuries", "Notes"]
    )

    with fixtures_tab:
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
        else:
            start, end = dt.date.today(), dt.date.today() + dt.timedelta(days=14)

        with session_scope() as session:
            fixtures_df = fixtures_for_league(session, league.code, start, end)

        if fixtures_df.empty:
            st.info(
                "No upcoming fixtures loaded for this range. Run "
                f"`uv run python scripts/refresh_live_data.py {league.code}`"
                f"{live_sync_requirement_note(league)} to pull them, or try "
                "a custom matchup below."
            )
        else:
            tz = timezone_selector()
            rows = []
            row_meta = []  # parallel to rows: (home_id, away_id, home_name, away_name, date)
            with session_scope() as session:
                for row in fixtures_df.sort_values("date").itertuples(index=False):
                    home_name = team_names[row.home_team_id]
                    away_name = team_names[row.away_team_id]
                    prediction = predict_fixture(
                        session,
                        params,
                        row.home_team_id,
                        row.away_team_id,
                        home_name,
                        away_name,
                        fixture_date=row.date,
                    )
                    row_dict = fixture_prediction_row(home_name, away_name, prediction)
                    row_dict["Kickoff"] = format_kickoff(row.kickoff_utc, row.date, tz)
                    rows.append(row_dict)
                    row_meta.append((row.home_team_id, row.away_team_id, home_name, away_name, row.date))
            column_order = fixture_columns_with_kickoff_first(rows[0])
            fixtures_event = st.dataframe(
                style_fixture_predictions(rows),
                use_container_width=True,
                column_order=column_order,
                on_select="rerun",
                selection_mode="single-row",
                key=f"predictor_fixtures_table_{league.code}",
            )
            st.caption("Click a fixture's row to see the full Poisson/Dixon-Coles math behind its prediction.")
            selected = fixtures_event.selection.rows if fixtures_event and fixtures_event.selection else []
            if selected:
                sel_home_id, sel_away_id, sel_home_name, sel_away_name, sel_date = row_meta[selected[0]]
                with session_scope() as session:
                    prediction = predict_fixture(
                        session, params, sel_home_id, sel_away_id, sel_home_name, sel_away_name, fixture_date=sel_date
                    )
                    render_prediction_breakdown(
                        prediction, sel_home_name, sel_away_name, session, league, sel_home_id, sel_away_id
                    )

        st.subheader("Custom matchup")
        st.caption("Predict any pairing directly from the fitted model, regardless of the fixture list.")
        col1, col2 = st.columns(2)
        team_options = sorted(id_by_name.keys())
        home_choice = col1.selectbox("Home team", team_options, key="home_choice")
        away_choice = col2.selectbox(
            "Away team", [t for t in team_options if t != home_choice], key="away_choice"
        )
        if st.button("Predict this matchup"):
            home_id, away_id = id_by_name[home_choice], id_by_name[away_choice]
            with session_scope() as session:
                prediction = predict_fixture(
                    session,
                    params,
                    home_id,
                    away_id,
                    home_choice,
                    away_choice,
                )
                row = fixture_prediction_row(home_choice, away_choice, prediction)
                st.table(style_fixture_predictions([row]))
                st.plotly_chart(scoreline_chart(prediction, home_choice, away_choice), use_container_width=True)
                render_prediction_breakdown(
                    prediction, home_choice, away_choice, session, league, home_id, away_id
                )

    with leaderboard_tab:
        leaderboard_df = leaderboard_dataframe(params, team_names)
        # Same reasoning as the Leagues page's standings search: this grid
        # is a canvas, invisible to the browser's own Ctrl+F.
        search_query = st.text_input(
            "Search team", placeholder="e.g. Man City", key=f"leaderboard_search_{league.code}"
        )
        if search_query:
            leaderboard_df = leaderboard_df[
                leaderboard_df["Team"].str.contains(search_query, case=False, na=False, regex=False)
            ]
        if search_query and leaderboard_df.empty:
            st.info(f'No team matches "{search_query}".')
        else:
            st.dataframe(
                leaderboard_df,
                use_container_width=True,
                column_order=LEADERBOARD_DISPLAY_COLUMNS,
            )

    with injuries_tab:
        any_injuries_shown = False
        for team_id in sorted(team_names, key=lambda t: team_names[t]):
            team_name = team_names[team_id]
            with session_scope() as session:
                team_injuries = get_injuries_for_team(session, team_id, team_name, dt.date.today())
            if not team_injuries.entries:
                continue

            any_injuries_shown = True
            st.markdown(f"**{team_name}**")
            for entry, source, return_date in zip(
                team_injuries.entries, team_injuries.sources, team_injuries.expected_return_dates
            ):
                return_note = f", est. return {return_date}" if return_date else ""
                st.write(
                    f"- {entry.player_name} ({entry.position}, weight={entry.importance_weight:.2f}) "
                    f"- source: {source}{return_note}"
                )
            adj_attack, adj_defense = adjust_strength(
                params.attack[team_id], params.defense[team_id], team_injuries.entries
            )
            st.caption(
                f"Attack {params.attack[team_id]:.2f} → {adj_attack:.2f}, "
                f"Defense {params.defense[team_id]:.2f} → {adj_defense:.2f}"
            )

        if not any_injuries_shown:
            if league.data_source == "espn":
                st.info(
                    "No injuries on record. Add entries to config/injuries.yaml, or run "
                    f"`uv run python scripts/refresh_live_data.py {league.code}` to pull "
                    "best-effort data from ESPN."
                )
            else:
                st.info(
                    "No injuries on record. This competition has no automatic injury source "
                    "-- add entries to config/injuries.yaml, or use the Notes tab below."
                )

    with notes_tab:
        st.caption(
            'Type notes about players or teams (e.g. "Saka is injured, ~3 weeks out", '
            '"Arsenal have been flat since the manager change"). An AI parses them into '
            "structured entries below for you to review and save - nothing is applied to "
            "predictions until you click Save."
        )

        unlocked = _render_notes_gate()
        if unlocked:
            st.session_state.setdefault("chat_history", [])
            st.session_state.setdefault("pending_cards", [])
            st.session_state.setdefault("editing_message", None)

            # A container so the history can be drawn AFTER a new note has been
            # parsed (below the input in code order, above it on the page) -- the
            # new note then appears with its own edit/delete menu straight away.
            history_box = st.container()
            prompt = st.chat_input("Enter a note about a player or team...")
            if prompt and not na.note_length_ok(prompt):
                st.warning(f"That note is too long (max {na.MAX_NOTE_LENGTH} characters).")
            elif prompt and na.parses_left(st.session_state) == 0:
                st.warning("This session has used all of its note parses - reload the page to start a new one.")
            elif prompt:
                note = new_message("user", prompt)
                st.session_state.chat_history.append(note)
                with history_box:
                    _run_note_parser(note["id"], prompt, team_names)
            with history_box:
                _render_chat_history(team_names)

            if st.session_state.pending_cards:
                st.subheader("Review pending notes")
                team_options = ["- select team -"] + sorted(team_names.values())
                position_options = ["attack", "defense"]
                affects_options = ["attack", "defense", "both"]

                for card in list(st.session_state.pending_cards):
                    uid = card["uid"]
                    with st.container(border=True):
                        if card["kind"] == "injury":
                            data = card["data"]
                            default_team = team_names.get(data.team_id, "- select team -")
                            cols = st.columns([2, 2, 1, 1, 2])
                            team_choice = cols[0].selectbox(
                                "Team", team_options, index=_safe_index(team_options, default_team), key=f"team_{uid}"
                            )
                            player_name = cols[1].text_input("Player", value=data.player_name, key=f"player_{uid}")
                            position = cols[2].selectbox(
                                "Position",
                                position_options,
                                index=_safe_index(position_options, data.position),
                                key=f"pos_{uid}",
                            )
                            weight = cols[3].slider(
                                "Weight", 0.0, 1.0, value=data.importance_weight, step=0.05, key=f"weight_{uid}"
                            )
                            return_date = cols[4].date_input(
                                "Est. return",
                                value=data.expected_return_date or (dt.date.today() + dt.timedelta(weeks=3)),
                                key=f"return_{uid}",
                            )
                            st.caption(f'Matched from "{data.team_name_raw}": {data.rationale}')

                            save_disabled = team_choice == "- select team -"
                            save_col, discard_col = st.columns([1, 1])
                            if save_col.button("Save", key=f"save_{uid}", disabled=save_disabled):
                                with session_scope() as session:
                                    add_or_update_chat_injury(
                                        session,
                                        id_by_name[team_choice],
                                        player_name,
                                        position,
                                        weight,
                                        return_date,
                                        note=data.rationale,
                                    )
                                st.session_state.pending_cards.remove(card)
                                st.rerun()
                            if discard_col.button("Discard", key=f"discard_{uid}"):
                                st.session_state.pending_cards.remove(card)
                                st.rerun()

                        else:  # form note
                            data = card["data"]
                            default_team = team_names.get(data.team_id, "- select team -")
                            cols = st.columns([2, 1, 1, 2])
                            team_choice = cols[0].selectbox(
                                "Team", team_options, index=_safe_index(team_options, default_team), key=f"team_{uid}"
                            )
                            magnitude = cols[1].slider(
                                "Magnitude (bad↔good)", -1.0, 1.0, value=data.magnitude, step=0.1, key=f"mag_{uid}"
                            )
                            affects = cols[2].selectbox(
                                "Affects",
                                affects_options,
                                index=_safe_index(affects_options, data.affects),
                                key=f"affects_{uid}",
                            )
                            expires_on = cols[3].date_input("Expires", value=data.expires_on, key=f"expires_{uid}")
                            summary_text = st.text_input("Summary", value=data.summary, key=f"summary_{uid}")
                            st.caption(
                                f'Matched from "{data.team_name_raw}": {data.rationale}. '
                                "Form notes are a rough, unvalidated heuristic (small, capped effect) - "
                                "treat with skepticism."
                            )

                            save_disabled = team_choice == "- select team -"
                            save_col, discard_col = st.columns([1, 1])
                            if save_col.button("Save", key=f"save_{uid}", disabled=save_disabled):
                                with session_scope() as session:
                                    add_form_note(
                                        session,
                                        id_by_name[team_choice],
                                        raw_text=data.rationale,
                                        summary=summary_text,
                                        magnitude=magnitude,
                                        affects=affects,
                                        expires_on=expires_on,
                                    )
                                st.session_state.pending_cards.remove(card)
                                st.rerun()
                            if discard_col.button("Discard", key=f"discard_{uid}"):
                                st.session_state.pending_cards.remove(card)
                                st.rerun()

        _render_active_notes(leagues, can_delete=unlocked)

    with st.expander("Model info"):
        st.write(f"Fitted at: {params.fitted_at}")
        st.write(f"Matches used: {params.n_matches}")
        st.write(f"Home advantage (gamma): {params.home_advantage:.3f}")
        st.write(f"Low-score correlation (rho): {params.rho:.3f}")
        st.write(f"Time-decay (xi): {params.xi}")
