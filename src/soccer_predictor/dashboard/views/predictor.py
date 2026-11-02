"""The original single-league workflow: fixtures/leaderboard/injuries/notes
tabs, custom matchup, and the AI chat notes feature. Reached from the
sidebar nav; can also be pre-seeded with a league via ?league=<code>.
"""

from __future__ import annotations

import datetime as dt
import html
import time
from uuid import uuid4

import pandas as pd
import streamlit as st

from soccer_predictor.ai import whatif_parser
from soccer_predictor.ai.note_parser import MissingApiKey as NoteParserMissingApiKey
from soccer_predictor.ai.note_parser import NoteParsingError, parse_note
from soccer_predictor.config import gemini_api_key, load_leagues, notes_password
from soccer_predictor.dashboard import injury_table as it
from soccer_predictor.dashboard import notes_access as na
from soccer_predictor.dashboard import whatif_access as wa
from soccer_predictor.dashboard.fixture_workings import render_fixture_workings
from soccer_predictor.dashboard.injury_tables import (
    injury_players_table_html,
    parse_injury_team,
    team_impact_table_html,
)
from soccer_predictor.dashboard.note_format import describe_form_note, describe_injury
from soccer_predictor.dashboard.rating_workings import render_rating_workings
from soccer_predictor.dashboard.whatif import (
    UpcomingMatch,
    build_adjustments,
    comparison_rows,
    describe_form_entry,
    describe_injury_entry,
    upcoming_matches,
)
from soccer_predictor.dashboard.chat_notes import (
    add_reply,
    delete_exchange,
    new_message,
    replace_note_text,
)
from soccer_predictor.dashboard.components import (
    FIXTURE_PARAM,
    INJURY_PANEL_CSS,
    INJURY_PARAM,
    LOCK_GROUP_LABELS,
    SCROLL_TO_INJURY_JS,
    _escape_markdown,
    FixtureRef,
    apply_fixture_pick,
    apply_injury_pick,
    apply_rating_pick,
    fixtures_table_html,
    inject_fixture_links,
    inject_injury_links,
    inject_rating_links,
    inject_standings_sort,
    injury_source_note,
    PREDICTION_STATUS_NOTE,
    RATING_PARAM,
    TIMEZONE_QUERY_KEY,
    fixture_prediction_row,
    current_season_leaderboard,
    format_kickoff,
    leaderboard_table_html,
    league_option_label,
    live_sync_requirement_note,
    parse_fixture_key,
    parse_rating_key,
    render_prediction_breakdown,
    scoreline_chart,
    style_fixture_predictions,
    team_page_href,
    timezone_selector,
)
from soccer_predictor.ingest import injuries as injuries_module
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
    locked_predictions_for_league,
    team_crests_for_league,
    teams_for_league,
)

LEADERBOARD_TAB = "Team Strength Leaderboard"
INJURIES_TAB = "Injuries"
TEAM_IMPACT_TAB = "Team impact"


def _safe_index(options: list, value, default: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return default


def default_date_range() -> tuple[dt.date, dt.date]:
    today = dt.date.today()
    return today, today + dt.timedelta(days=14)


def date_range_from_url(query_params, default: tuple[dt.date, dt.date]) -> tuple[dt.date, dt.date]:
    """The fixture date range the fixtures table's links carry in `from`/`to` (ISO dates),
    else `default`: a link reloads the page, which would otherwise reset the range and
    lose a match picked from outside it."""
    try:
        start = dt.date.fromisoformat(query_params.get("from") or "")
        end = dt.date.fromisoformat(query_params.get("to") or "")
    except ValueError:
        return default
    return (start, end) if start <= end else default


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
    if na.attempts_left(state) == 0 or na.NOTES_LIMITER.blocked():
        st.error("Too many wrong passwords. Try again later.")
        return False
    with st.form("notes_unlock_form"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        result = na.try_unlock(state, entered, password, limiter=na.NOTES_LIMITER)
        if result == na.OK:
            st.rerun()
        time.sleep(1)  # slows down guessing
        if result == na.LOCKED_OUT:
            st.error("Too many wrong passwords. Try again later.")
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


def _injury_data(params, team_names: dict[int, str]) -> tuple[pd.DataFrame, dict]:
    """Everyone currently listed as out, doubtful or suspended (injury_table.players_frame), and
    each team's ratings with its entries, which the team-impact table is worked out from."""
    today = dt.date.today()
    rows, ratings = [], {}
    with session_scope() as session:
        for team_id, team_name in sorted(team_names.items(), key=lambda item: item[1]):
            team_injuries = get_injuries_for_team(session, team_id, team_name, today)
            rows += it.player_rows(team_name, team_injuries, today)
            ratings[team_name] = (params.attack[team_id], params.defense[team_id], team_injuries.entries)
    return it.players_frame(rows), ratings


def _close_team_injuries() -> None:
    st.query_params.pop(INJURY_PARAM, None)


def _render_team_injuries(league, frame: pd.DataFrame, team: str, team_ids: dict[str, int], close_key: str) -> None:
    """Who is out for `team`, under the team-impact table, with a Close button that drops the
    `injuries` parameter again."""
    team_frame = frame[frame["Team"] == team].reset_index(drop=True)  # biggest impact first
    counts = it.summary_counts(team_frame)
    st.html(f"<style>{INJURY_PANEL_CSS}</style><script>{SCROLL_TO_INJURY_JS}</script>", unsafe_allow_javascript=True)
    with st.container(key="injury_breakdown", border=True):
        st.markdown(f"#### Injured players: {_escape_markdown(team)}")
        st.button("Close", key=close_key, on_click=_close_team_injuries)
        st.caption(
            f"{counts['players']} listed: {counts[it.OUT]} out · {counts[it.DOUBTFUL]} doubtful · "
            f"{counts[it.SUSPENDED]} suspended. The biggest impact on the team comes first."
        )
        st.markdown(injury_players_table_html(team_frame, team_ids, league.code, show_team=False), unsafe_allow_html=True)
        st.markdown(
            f'<a href="{html.escape(team_page_href(league.code, team_ids[team]))}" target="_self">'
            f"Open {html.escape(team)}'s page</a> for its schedule, squad and results.",
            unsafe_allow_html=True,
        )


def _selected_injury_team(frame: pd.DataFrame, team_ids: dict[str, int]) -> str | None:
    """The team whose injured players the `injuries` URL parameter has open: only ever a team with
    someone listed, so anything else drops the parameter."""
    if st.query_params.get(INJURY_PARAM) is None:
        return None
    team = parse_injury_team(st.query_params[INJURY_PARAM], team_ids, set(frame["Team"]))
    if team is None:
        del st.query_params[INJURY_PARAM]
    return team


@st.fragment
def _render_injuries_tab(
    league, frame: pd.DataFrame, ratings: dict, team_ids: dict[str, int], tz_name: str, page_params: str
) -> None:
    """Everyone currently listed as out, doubtful or suspended, as a table that
    can be filtered, searched and sorted, plus what that does to each team's
    ratings (dashboard/injury_table.py). A team's name opens its page; in the
    team-impact table its number of players listed opens who is out, under the table
    (`tz_name` and `page_params` ride in the link to it).

    A fragment, so a click here, or a filter, reruns only this tab rather than the whole
    page (every fixture's forecast, the leaderboard): that was most of the wait. It
    reads the open team from the URL itself for the same reason."""
    apply_injury_pick()
    selected = _selected_injury_team(frame, team_ids)
    if frame.empty:
        st.info("No injuries on record.")
        if not injuries_module.has_automatic_feed(league):
            st.caption(
                f"This site has no automatic injury feed for {league.name}, so only entries added by hand "
                "(config/injuries.yaml or saved AI notes) appear here."
            )
        return

    counts = it.summary_counts(frame)
    st.caption(
        f"{counts['players']} players across {counts['teams']} teams: {counts[it.OUT]} out · "
        f"{counts[it.DOUBTFUL]} doubtful · {counts[it.SUSPENDED]} suspended."
    )
    source_note = injury_source_note(league.code)
    if source_note:
        st.caption(source_note)
    inject_standings_sort()
    # A team's list is opened from the team-impact table, so that is the sub-tab a link to it lands on.
    players_tab, teams_tab = st.tabs(["Players", TEAM_IMPACT_TAB], default=TEAM_IMPACT_TAB if selected else None)

    with players_tab:
        col_team, col_status, col_position, col_search = st.columns([2, 2, 1.4, 2])
        teams = col_team.multiselect("Team", sorted(frame["Team"].unique()), key=f"inj_teams_{league.code}")
        statuses = col_status.multiselect(
            "Status", [s for s in it.STATUS_ORDER if s in set(frame["Status"])], key=f"inj_status_{league.code}"
        )
        positions = col_position.multiselect(
            "Position", sorted(frame["Position"].unique()), key=f"inj_position_{league.code}"
        )
        query = col_search.text_input("Search player", key=f"inj_query_{league.code}")
        shown = it.filter_players(frame, teams, statuses, positions, query)
        if shown.empty:
            st.info("No player matches those filters.")
        else:
            st.markdown(injury_players_table_html(shown, team_ids, league.code), unsafe_allow_html=True)
            st.caption(
                "Click a team's name for its page, or a column heading to sort. Within each team the biggest "
                "impact comes first."
            )

    with teams_tab:
        impact = it.team_impact_frame(ratings)
        inject_injury_links()
        st.markdown(
            team_impact_table_html(
                impact, team_ids, league.code, tz_name, page_params, team_ids.get(selected) if selected else None
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            "The model's attack and defence ratings for each team before and after these players are "
            "taken out. Defence is goals conceded, so a rise there is worse. Teams most affected come first. "
            "Click a team's name for its page, or its number of players listed to see who is out."
        )
        if selected:
            _render_team_injuries(league, frame, selected, team_ids, f"close_injury_breakdown_{league.code}")


WHATIF_RESULT_KEY = "whatif_result"


def _ask_whatif(
    note: str, home_id: int, home_name: str, away_id: int, away_name: str, fixture_date: dt.date | None = None
) -> None:
    """Runs one what-if ask (within the visitor and site-wide limits) and keeps
    the outcome in this session only -- an error message or the AI's entries.
    Nothing is written to the database. `fixture_date` is the picked match's date
    (None for a hypothetical pairing), so the forecast is made for that day."""
    cached = whatif_parser.cached_entries(home_id, away_id, note)
    if cached is not None:  # the same note on the same match: no quota used
        st.session_state[WHATIF_RESULT_KEY] = {
            "home_id": home_id, "away_id": away_id, "date": fixture_date, "entries": cached, "error": None,
            "n": st.session_state.get("whatif_runs", 0) + 1,
        }
        st.session_state["whatif_runs"] = st.session_state[WHATIF_RESULT_KEY]["n"]
        return
    verdict = wa.try_ask(st.session_state, note)
    if verdict != wa.OK:
        st.session_state[WHATIF_RESULT_KEY] = {
            "home_id": home_id, "away_id": away_id, "date": fixture_date, "error": wa.MESSAGES[verdict],
        }
        return
    entries, error = None, None
    with st.spinner("Asking the AI..."):
        try:
            entries = whatif_parser.parse_whatif_note(note, home_id, home_name, away_id, away_name)
        except whatif_parser.WhatIfAtCapacity:
            error = wa.MESSAGES[wa.DAY_LIMIT]
        except whatif_parser.WhatIfUnavailable:
            error = "The AI what-if is turned off on this site."
        except whatif_parser.WhatIfError as exc:
            error = str(exc)
    runs = st.session_state.get("whatif_runs", 0) + 1
    st.session_state["whatif_runs"] = runs
    st.session_state[WHATIF_RESULT_KEY] = {
        "home_id": home_id, "away_id": away_id, "date": fixture_date, "entries": entries, "error": error, "n": runs,
    }


def _render_whatif_results(result: dict, params, team_names: dict[int, str], league) -> None:
    home_id, away_id = result["home_id"], result["away_id"]
    home_name, away_name = team_names[home_id], team_names[away_id]
    entries = result["entries"]
    if entries.empty:
        st.info("That note didn't mention either team's players or form, so there is nothing to change.")
        return
    if entries.skipped:
        st.caption(f"{entries.skipped} item(s) the AI read were left out: not about these two teams, or over the limit.")

    st.markdown("**What the AI read from your note**")
    st.caption(
        "Drag a slider to count less of what the AI inferred - the table below updates straight away, "
        "with no new AI request."
    )
    run = result["n"]
    injury_scale, form_scale = [], []
    for index, injury in enumerate(entries.injuries):
        team = team_names.get(injury.team_id, "")
        headline, maths = describe_injury_entry(
            _escape_markdown(injury.player_name), _escape_markdown(team), injury.position, injury.importance_weight
        )
        st.markdown(headline)
        st.caption(f"{maths} · AI: {_escape_markdown(injury.rationale)}")
        injury_scale.append(
            st.slider("Count this much of it", 0.0, 1.0, 1.0, 0.05, key=f"whatif_inj_{run}_{index}", format="%.2f")
        )
    for index, form in enumerate(entries.form_notes):
        team = team_names.get(form.team_id, "")
        headline, maths = describe_form_entry(
            _escape_markdown(form.summary), _escape_markdown(team), form.magnitude, form.affects
        )
        st.markdown(headline)
        st.caption(f"{maths} · AI: {_escape_markdown(form.rationale)}")
        form_scale.append(
            st.slider("Count this much of it", 0.0, 1.0, 1.0, 0.05, key=f"whatif_form_{run}_{index}", format="%.2f")
        )

    adjustments = build_adjustments(entries, home_id, injury_scale, form_scale)
    fixture_date = result.get("date")  # a picked match's day; None for a hypothetical pairing (today)
    with session_scope() as session:  # read-only: the what-if entries never go in the database
        base = predict_fixture(session, params, home_id, away_id, home_name, away_name, fixture_date)
        adjusted = predict_fixture(
            session, params, home_id, away_id, home_name, away_name, fixture_date,
            extra_home_injuries=adjustments.home_injuries,
            extra_away_injuries=adjustments.away_injuries,
            extra_home_form=adjustments.home_form,
            extra_away_form=adjustments.away_form,
        )
    st.markdown(f"**{_escape_markdown(home_name)} v {_escape_markdown(away_name)}: the model, with and without your note**")
    st.table(pd.DataFrame(comparison_rows(base, adjusted)).set_index(""))
    st.plotly_chart(scoreline_chart(adjusted, home_name, away_name), use_container_width=True)
    st.caption(
        "What-if only: this is not saved, not shared with other visitors, and not part of the Track Record. "
        "It runs through the same adjustment formulas as saved notes, which are capped and not backtested - "
        "treat the shift as an illustration of how such news could move the odds, not a better forecast."
    )


def _fill_whatif_teams(
    pick_key: str, home_key: str, away_key: str, matches: dict[str, UpcomingMatch], team_names: dict[int, str]
) -> None:
    """The match picker's on_change: put the picked match's teams in the home and away boxes."""
    match = matches.get(st.session_state.get(pick_key))
    if match is not None:
        st.session_state[home_key] = team_names[match.home_id]
        st.session_state[away_key] = team_names[match.away_id]


def _forget_whatif_match(pick_key: str) -> None:
    """A team changed by hand is no longer the picked match, so the picker is cleared."""
    st.session_state[pick_key] = None


def _render_whatif_section(league, params, team_names: dict[int, str]) -> None:
    st.subheader("Try it with the free model")
    st.caption(
        "Open to everyone. Pick an upcoming match, or any two teams for a hypothetical, type a note "
        "about a player or a team's form (e.g. \"their top scorer is out for this one\"), and an AI reads "
        "it and shows how it would shift the odds."
    )
    if not gemini_api_key():
        st.info("The AI what-if is turned off on this site (no GEMINI_API_KEY is set).")
        return

    id_by_name = {name: team_id for team_id, name in team_names.items()}
    options = sorted(id_by_name)
    pick_key, home_key, away_key = (f"whatif_{part}_{league.code}" for part in ("match", "home", "away"))

    today, horizon = default_date_range()
    with session_scope() as session:
        matches = {m.label: m for m in upcoming_matches(fixtures_for_league(session, league.code, today, horizon), team_names)}
    if matches:
        st.selectbox(
            "Upcoming match",
            list(matches),
            index=None,
            placeholder="Pick a match to fill in the teams, or choose any two teams below",
            key=pick_key,
            on_change=_fill_whatif_teams,
            args=(pick_key, home_key, away_key, matches, team_names),
        )
    col1, col2 = st.columns(2)
    home_choice = col1.selectbox(
        "Home team", options, key=home_key, on_change=_forget_whatif_match, args=(pick_key,)
    )
    away_choice = col2.selectbox(
        "Away team",
        [t for t in options if t != home_choice],
        key=away_key,
        on_change=_forget_whatif_match,
        args=(pick_key,),
    )
    picked = matches.get(st.session_state.get(pick_key))
    fixture_date = (
        picked.date
        if picked is not None and (picked.home_id, picked.away_id) == (id_by_name[home_choice], id_by_name[away_choice])
        else None
    )
    note = st.text_area(
        "Your note",
        max_chars=wa.MAX_NOTE_LENGTH,
        height=90,
        placeholder="e.g. Their goalkeeper is injured and the team has been struggling to score lately",
        key=f"whatif_note_{league.code}",
    )
    privacy_note = st.empty()  # filled after the button is handled, so the count is current
    if st.button("Ask the AI", key=f"whatif_ask_{league.code}"):
        _ask_whatif(note, id_by_name[home_choice], home_choice, id_by_name[away_choice], away_choice, fixture_date)
    privacy_note.caption(
        "Your note is sent to Google's Gemini API on its free tier, which Google may use to improve its "
        "products - please don't include personal details. "
        f"{wa.asks_left(st.session_state)} AI ask(s) left this visit; the site shares a daily free allowance."
    )

    result = st.session_state.get(WHATIF_RESULT_KEY)
    if result is None:
        return
    if result["home_id"] != id_by_name[home_choice] or result["away_id"] != id_by_name[away_choice]:
        st.info("You changed the teams - ask the AI again to see how your note shifts this match.")
        return
    if result["error"]:
        st.warning(result["error"])
        return
    _render_whatif_results(result, params, team_names, league)


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
            value=date_range_from_url(st.query_params, default_date_range()),
        )

        st.divider()
        st.header("Track Record")
        st.caption(
            "All leagues. Each forecast is locked when the lineups are announced (about an hour before "
            "kickoff) and graded after the match. "
            "Right outcome = the model's most likely result (win, draw or loss) happened."
        )
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
            st.caption(f"{accuracy.hit_rate:.2%} success rate ({accuracy.graded_total} graded)")
            for group_key, group_label in LOCK_GROUP_LABELS.items():
                group = accuracy.groups.get(group_key)
                if group is not None and group.graded_total:
                    st.caption(f"{group_label}: {group.hit_rate:.0%} right ({group.graded_total} graded)")

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

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:
        start, end = default_date_range()

    with session_scope() as session:
        leaderboard_df = current_season_leaderboard(session, league, params, team_names)

    # The leaderboard figure whose workings are open: named in the URL by the figure's
    # link, and only ever a team on the leaderboard (anything else is dropped). A link
    # reloads the page, which would land on the first tab, so the leaderboard is opened.
    apply_rating_pick()
    rating_selected = None
    if st.query_params.get(RATING_PARAM) is not None:
        rating_selected = parse_rating_key(st.query_params[RATING_PARAM])
        if rating_selected is None or rating_selected[0] not in set(leaderboard_df["team_id"]):
            del st.query_params[RATING_PARAM]
            rating_selected = None

    # The same for a team's injured players, opened from the Injuries tab's team-impact table by
    # its players-listed link: named in the URL by the team's id, and only ever a team with
    # someone listed. That link would also land on the first tab, so the Injuries tab is opened.
    injuries_frame, injury_ratings = _injury_data(params, team_names)
    apply_injury_pick()
    injury_selected = _selected_injury_team(injuries_frame, id_by_name)

    opened_tab = LEADERBOARD_TAB if rating_selected is not None else INJURIES_TAB if injury_selected else None
    fixtures_tab, leaderboard_tab, injuries_tab, notes_tab = st.tabs(
        ["Upcoming Fixtures", LEADERBOARD_TAB, INJURIES_TAB, "AI notes"], default=opened_tab
    )

    with fixtures_tab:
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
            fixture_refs: list[FixtureRef] = []  # parallel to rows
            with session_scope() as session:
                locked_predictions = locked_predictions_for_league(session, league.code)
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
                    row_dict = fixture_prediction_row(
                        home_name,
                        away_name,
                        prediction,
                        locked_predictions.get((row.date, row.home_team_id, row.away_team_id)),
                    )
                    row_dict["Kickoff"] = format_kickoff(row.kickoff_utc, row.date, tz)
                    rows.append(row_dict)
                    fixture_refs.append(FixtureRef(row.home_team_id, row.away_team_id, home_name, away_name, row.date))

            # The match whose workings are open: named in the URL by the predicted-score
            # link, and only ever one of the rows shown (anything else is dropped).
            selected_index = None
            apply_fixture_pick()
            if st.query_params.get(FIXTURE_PARAM) is not None:
                wanted = parse_fixture_key(st.query_params[FIXTURE_PARAM])
                selected_index = next(
                    (i for i, ref in enumerate(fixture_refs) if wanted == (ref.home_id, ref.away_id, ref.date)), None
                )
                if selected_index is None:
                    del st.query_params[FIXTURE_PARAM]

            inject_fixture_links()
            st.markdown(
                fixtures_table_html(
                    rows,
                    fixture_refs,
                    league.code,
                    tz_name=tz.key,
                    match_details=bool(league.espn_league_slug),
                    selected=selected_index,
                    page_params=f"&from={start.isoformat()}&to={end.isoformat()}",
                ),
                unsafe_allow_html=True,
            )
            st.caption(
                "Click a team's name for its page, the kickoff for the match details, or the predicted "
                "score to see how it was calculated. " + PREDICTION_STATUS_NOTE
            )
            if selected_index is not None:
                render_fixture_workings(
                    fixture_refs[selected_index], params, league, f"close_fixture_breakdown_{league.code}"
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
        search_query = st.text_input(
            "Search team", placeholder="e.g. Man City", key=f"leaderboard_search_{league.code}"
        )
        if search_query and not leaderboard_df["Team"].str.contains(search_query, case=False, na=False, regex=False).any():
            st.info(f'No team matches "{search_query}".')
        else:
            with session_scope() as session:
                crest_urls = team_crests_for_league(session, league.code)
            inject_standings_sort()
            inject_rating_links()
            st.markdown(
                leaderboard_table_html(
                    leaderboard_df,
                    league.code,
                    crest_urls,
                    search_query,
                    tz_name=st.query_params.get(TIMEZONE_QUERY_KEY, ""),
                    page_params=f"&from={start.isoformat()}&to={end.isoformat()}",
                    selected=rating_selected,
                ),
                unsafe_allow_html=True,
            )
            st.caption(
                "Click a team's name for its page, with its squad and how each player counts toward these "
                "ratings, a figure to see how it is calculated, or a column heading to sort. Net strength is "
                "attack minus defense, and the table starts ranked by it."
            )
        if rating_selected is not None:
            render_rating_workings(
                *rating_selected, leaderboard_df, params, league, f"close_rating_breakdown_{league.code}"
            )

    with injuries_tab:
        _render_injuries_tab(
            league,
            injuries_frame,
            injury_ratings,
            id_by_name,
            tz_name=st.query_params.get(TIMEZONE_QUERY_KEY, ""),
            page_params=f"&from={start.isoformat()}&to={end.isoformat()}",
        )

    with notes_tab:
        # Top: the free what-if for everyone. Below: the owner-only notes that are
        # saved and change the real predictions.
        _render_whatif_section(league, params, team_names)

        st.divider()
        st.subheader("Change the predictions with the more advanced model")
        st.caption(
            'Type notes about players or teams (e.g. "Saka is injured, ~3 weeks out", '
            '"Arsenal have been flat since the manager change"). Claude, a more advanced model, parses '
            "them into structured entries below for you to review and save - nothing is applied to "
            "predictions until you click Save. Saved notes change the forecasts everyone sees, "
            "so adding them is owner-only."
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
