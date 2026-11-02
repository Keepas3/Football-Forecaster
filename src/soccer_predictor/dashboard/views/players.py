"""Players page: search for an individual player by name (or a whole club's
squad by the club's name) across every team in a league, then see their current club plus (opt-in, quota-conscious)
stats across past seasons. Reached from the sidebar nav.

Every fetch here reuses existing team-scoped ingest functions (squad.py,
understat_client.py, asa_client.py) -- there is no per-player database
table anywhere in this app; everything is fetched live and disk-cached by
those clients, same as Team Detail's own squad/player-stats sections.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import streamlit as st

from soccer_predictor.config import load_leagues, load_manual_captains, load_manual_star_players
from soccer_predictor.dashboard.components import (
    PLAYER_PANEL_CSS,
    PLAYER_PARAM,
    SCROLL_TO_PLAYER_JS,
    apply_player_pick,
    inject_player_links,
    inject_standings_sort,
    league_option_label,
)
from soccer_predictor.dashboard.players_table import (
    club_team_ids,
    parse_player_key,
    player_key,
    search_results_table_html,
    top_players_table_html,
)
from soccer_predictor.ingest import defensive_players, understat_client
from soccer_predictor.ingest.asa_client import ASA_AVAILABLE_SEASONS
from soccer_predictor.ingest.asa_client import fetch_team_season as fetch_asa_team_season
from soccer_predictor.ingest.player_importance import (
    find_archive_player,
    find_asa_player,
    find_understat_player,
    resolve_team_historical_stats,
)
from soccer_predictor.ingest.squad import PlayerSearchResult, search_players_in_league
from soccer_predictor.ingest.top_players import (
    METRIC_LABELS,
    MIN_MINUTES_PER_90,
    SALARY,
    XG_XA_PER_90,
    available_metrics,
    fetch_top_players,
    find_top_player,
    format_metric,
    league_rankings,
    normalize_name,
    per_90,
    table_rows,
)
from soccer_predictor.storage.db import session_scope
from soccer_predictor.storage.repository import all_alias_texts, team_crests_for_league, teams_for_league


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


_normalized = normalize_name


def find_squad_entry(session, league, name: str, team: str) -> PlayerSearchResult | None:
    """The squad entry for a top-players row, so the usual player detail opens.
    Tries the full name, then the surname alone (sources spell first names
    differently); among several hits prefers the same-named player at the
    same club."""
    surname = name.split()[-1] if name.split() else name
    for query in dict.fromkeys((name, surname)):
        results = search_players_in_league(session, league, query)
        exact = [r for r in results if _normalized(r.player.name) == _normalized(name)]
        same_club = [r for r in (exact or results) if _normalized(r.team_name) == _normalized(team)]
        for candidates in (same_club, exact, results if query == surname else []):
            if candidates:
                return candidates[0]
    return None


def _metric_rows(items: list[tuple[str, str | int]]) -> None:
    """`items` as rows of up to four st.metric tiles."""
    for start in range(0, len(items), 4):
        chunk = items[start : start + 4]
        for column, (label, value) in zip(st.columns(4), chunk):
            column.metric(label, value)


def _render_current_season(league, row, source: str, name: str, team_name: str) -> None:
    """The player's numbers this season: the headline four, then the volume
    stats, the per-90 rates, the source's extras and where he stands in the
    league. Everything comes from rows already fetched (the team's stats from
    Understat/ASA and the cached whole-league listing), so nothing extra is
    requested."""
    _metric_rows(
        [
            ("Goals", row.goals),
            ("Assists", row.assists),
            ("xG", f"{row.xg:.2f}"),
            ("xA", f"{row.xa:.2f}"),
        ]
    )
    volume = []
    if getattr(row, "appearances", 0):
        volume.append(("Appearances", row.appearances))
    volume += [("Shots", row.shots), ("Key passes", row.key_passes)]
    _metric_rows(volume)

    rates = []
    for label, value in (
        ("Goals per 90", row.goals),
        ("Assists per 90", row.assists),
        ("xG + xA per 90", row.xg + row.xa),
    ):
        rate = per_90(value, row.minutes)
        if rate is not None:
            rates.append((label, f"{rate:.2f}"))
    # Scoring more than the chances warranted (+) or fewer (-): a finishing
    # read, and a hint of which way the goals tally may move.
    rates.append(("Goals minus xG", f"{row.goals - row.xg:+.2f}"))
    _metric_rows(rates)

    extras = []
    if source == "Understat":
        extras += [
            ("Non-penalty goals", row.non_penalty_goals),
            ("npxG", f"{row.non_penalty_xg:.2f}"),
            ("Yellow cards", row.yellow_cards),
            ("Red cards", row.red_cards),
        ]
    else:
        extras.append(("Points added", f"{row.points_added:.2f}"))
    top = fetch_top_players(league)
    entry = find_top_player(top.players, name, team_name)
    if entry is not None and entry.salary is not None:
        extras.append(("Guaranteed salary", format_metric(SALARY, entry.salary)))
    _metric_rows(extras)

    if entry is not None:
        standing = " · ".join(
            f"{label} #{rank} of {total}" for label, rank, total in league_rankings(top.players, entry)
        )
        st.caption(f"League standing: {standing}.")


def _club_team_ids(league, rows: list[dict]) -> dict[str, int]:
    """Our team behind each club of the top list, so the club's name can link to its page."""
    with session_scope() as session:
        teams = teams_for_league(session, league.code)
        aliases = all_alias_texts(session, "api", league.code) + all_alias_texts(session, "csv", league.code)
    return club_team_ids({row["Club"] for row in rows}, teams, aliases)


ROLE_ALL = "all"
ROLE_OPTIONS = [ROLE_ALL, *defensive_players.ROLES]
ROLE_OPTION_LABELS = {ROLE_ALL: "All players", **defensive_players.ROLE_LABELS}

# What each source's defender and keeper numbers mean, shown under the table.
_DEFENSIVE_NOTES = {
    ("EPL", defensive_players.DEFENDER): (
        "Defensive actions are tackles plus clearances, blocks and interceptions. Clean sheets and goals "
        "conceded count the whole team while he was on the pitch, so a strong side flatters its defenders."
    ),
    ("EPL", defensive_players.GOALKEEPER): (
        "Goals prevented is the team's expected goals conceded minus the goals it let in while he played: a "
        "rough guide, since the defence in front of him counts too. Save % is saves out of saves plus goals conceded."
    ),
    ("MLS", defensive_players.DEFENDER): (
        "g+ (goals added) scores every action by how much it changes his team's chance of scoring or conceding. "
        "The figures are above the average player in his position; interrupting is tackles, interceptions and clearances."
    ),
    ("MLS", defensive_players.GOALKEEPER): (
        "Goals prevented is the expected goals of the shots on target he faced minus the goals he let in. "
        "Save % is saves out of shots on target faced."
    ),
}


def _no_defensive_numbers(league, role: str) -> str:
    """Why a Defenders or Goalkeepers list is empty."""
    label = defensive_players.ROLE_LABELS[role].lower()
    source = defensive_players.SOURCES.get(league.code)
    if source:
        return f"The {source} feed isn't available right now, so there are no {label} to rank. Try again later."
    leagues = load_leagues()
    covered = " and ".join(f"{leagues[code].name} ({name})" for code, name in defensive_players.SOURCES.items() if code in leagues)
    return f"No free source has {role} statistics for {league.name}; they are available for {covered}."


def _render_defensive_players(league, role: str, selected: str | None) -> None:
    """The league's best defenders or goalkeepers, where a free source has the numbers.
    See ingest/defensive_players.py."""
    data = defensive_players.fetch_defensive_players(league, role)
    metrics = defensive_players.available_metrics(role, data.players)
    if not metrics:
        st.info(_no_defensive_numbers(league, role))
        return

    metric = st.segmented_control(
        "Rank by",
        options=metrics,
        format_func=lambda m: defensive_players.METRICS[m].label,
        default=metrics[0],
        key=f"players_{role}_metric_{league.code}",
    )
    if metric not in metrics:
        metric = metrics[0]

    rows = defensive_players.table_rows(data.players, role, metric)
    if not rows:
        st.info("No player has enough minutes yet for that ranking.")
    else:
        st.markdown(
            top_players_table_html(rows, league.code, _club_team_ids(league, rows), selected),
            unsafe_allow_html=True,
        )

    season = f", {_season_label(data.season, league.season_display == 'single_year')} season" if data.season else ""
    notes = [
        f"Source: {data.source}{season}. Click a player's name for their stats below the table, a club's name "
        "for its page.",
        _DEFENSIVE_NOTES.get((league.code, role), ""),
    ]
    if any(defensive_players.METRICS[m].needs_minutes for m in metrics):
        notes.append(f"Rates and percentages leave out players with fewer than {MIN_MINUTES_PER_90} minutes.")
    st.caption(" ".join(note for note in notes if note))


def _render_defending(league, name: str, team_name: str) -> None:
    """A defender's or goalkeeper's numbers and where he stands among his league's, when the
    league's source has them; nothing otherwise."""
    for role in defensive_players.ROLES:
        data = defensive_players.fetch_defensive_players(league, role)
        entry = defensive_players.find_defensive_player(data.players, name, team_name)
        if entry is None:
            continue
        st.markdown("**Defending**" if role == defensive_players.DEFENDER else "**Goalkeeping**")
        _metric_rows(
            [
                (defensive_players.METRICS[m].label, defensive_players.format_value(m, entry.values[m]))
                for m in defensive_players.available_metrics(role, data.players)
                if entry.values.get(m) is not None
            ]
        )
        standing = " · ".join(
            f"{label} #{rank} of {total}" for label, rank, total in defensive_players.league_rankings(data.players, entry)
        )
        st.caption(
            f"{entry.minutes} minutes played ({data.source})."
            + (f" Among the league's {defensive_players.ROLE_LABELS[role].lower()}: {standing}." if standing else "")
        )
        return


def _render_top_players(league, selected: str | None) -> None:
    """The default view when nothing is searched: the league's top players.
    See ingest/top_players.py for where each league's numbers come from."""
    with st.spinner("Loading the league's players..."):  # a cold start fetches them over the network
        top = fetch_top_players(league)
    if not top.players:
        st.caption("Type a player or club name to search across every team in this league.")
        return

    st.subheader("Top players")
    role = st.segmented_control(
        "Show",
        options=ROLE_OPTIONS,
        format_func=ROLE_OPTION_LABELS.get,
        default=ROLE_ALL,
        key=f"players_role_{league.code}",
    )
    if role in defensive_players.ROLES:  # a deselected control gives None: the usual list
        _render_defensive_players(league, role, selected)
        return

    metrics = available_metrics(top.players)
    metric = st.segmented_control(
        "Rank by",
        options=metrics,
        format_func=METRIC_LABELS.get,
        default=metrics[0],
        key=f"players_metric_{league.code}",
    )
    if metric not in metrics:
        metric = metrics[0]

    rows = table_rows(top.players, metric)
    if not rows:
        st.info("No player has enough minutes yet for that ranking.")
    else:
        st.markdown(
            top_players_table_html(rows, league.code, _club_team_ids(league, rows), selected),
            unsafe_allow_html=True,
        )

    season = f", {_season_label(top.season, league.season_display == "single_year")} season" if top.season else ""
    notes = [
        f"Source: {top.source}{season}. Click a player's name for their stats below the table, a club's name "
        "for its page, or search by name above."
    ]
    if SALARY not in metrics:
        notes.append("Ratings and contract values aren't available from any free source for this league.")
    if metric == XG_XA_PER_90:
        notes.append(f"Players with fewer than {MIN_MINUTES_PER_90} minutes are left out.")
    st.caption(" ".join(notes))


SELECTED_PLAYER_KEY = "players_selected"


def _at_the_named_club(found: PlayerSearchResult | None, club: str, own_clubs: list[str]) -> bool:
    """Whether a squad entry looked up for a player at `club` is fit to open. `find_squad_entry`
    falls back to a surname match at any club, because a stats source spells clubs its own way;
    but when `club` is one of our own team names (a link from another page of this app) a
    player of that surname elsewhere is somebody else, so the entry has to be at that club."""
    if found is None:
        return False
    if normalize_name(club) not in {normalize_name(name) for name in own_clubs}:
        return True
    return normalize_name(found.team_name) == normalize_name(club)


def _selected_player(league, key: str | None, known: Sequence[PlayerSearchResult] = ()) -> PlayerSearchResult | None:
    """The squad entry the `player` URL parameter (`key`) names: one of the search results
    already in hand (`known`), else looked up in the league's squads. When it names nobody
    there, the parameter is dropped and the page says so."""
    if key is None:
        return None
    parsed = parse_player_key(key)
    found = None
    if parsed is not None:
        cached = st.session_state.get(SELECTED_PLAYER_KEY)
        if cached and cached[0] == (league.code, key):
            return cached[1]
        found = next((r for r in known if player_key(r.player.name, r.team_name) == key), None)
        if found is None:
            with session_scope() as session:
                found = find_squad_entry(session, league, *parsed)
                own_clubs = list(teams_for_league(session, league.code).values())
            found = found if _at_the_named_club(found, parsed[1], own_clubs) else None
    if found is None:
        del st.query_params[PLAYER_PARAM]
        if parsed is not None:
            at_club = f" at {parsed[1]}" if parsed[1] else ""
            st.caption(
                f"No squad entry found for {parsed[0]}{at_club}. The squad list may spell the name differently: "
                "try searching for part of it above."
            )
        return None
    st.session_state[SELECTED_PLAYER_KEY] = ((league.code, key), found)
    return found


def _forget_player() -> None:
    """The player picked belongs to the league that was showing: changing league drops them."""
    if PLAYER_PARAM in st.query_params:
        del st.query_params[PLAYER_PARAM]


def render() -> None:
    leagues = load_leagues()
    if not leagues:
        st.error("No leagues configured in config/leagues.yaml.")
        st.stop()

    apply_player_pick()  # a click on a player's name, before the parameter is read below

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
            on_change=_forget_player,
        )

    league = leagues[league_code]
    st.title(f"{league.name} Players")

    selected = st.query_params.get(PLAYER_PARAM)
    results: list[PlayerSearchResult] = []
    inject_standings_sort()
    inject_player_links()

    search_query = st.text_input("Search player or club", placeholder="e.g. Haaland or Liverpool")
    if search_query:
        with session_scope() as session:
            results = search_players_in_league(session, league, search_query, include_club=True)

        if not results:
            st.info(f'No player or club matches "{search_query}".')
        else:
            st.markdown(search_results_table_html(results, league.code, selected), unsafe_allow_html=True)
            st.caption("Click a player's name for their stats below the table, or a club's name for its page.")
    else:
        _render_top_players(league, selected)

    selection = _selected_player(league, selected, results)
    if selection is None:
        return

    st.html(f"<style>{PLAYER_PANEL_CSS}</style><script>{SCROLL_TO_PLAYER_JS}</script>", unsafe_allow_javascript=True)
    with st.container(key="player_detail", border=True):
        if st.button("Close", key=f"close_player_{league.code}"):
            del st.query_params[PLAYER_PARAM]
            st.rerun()
        _render_player(league, selection)


def _render_player(league, selection: PlayerSearchResult) -> None:
    """One player's page: who they are, this season's numbers, and (opt-in) any past season's."""
    player = selection.player
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
        _render_current_season(league, current_season_row, current_season_source, player.name, selection.team_name)
        st.caption(f"{current_season_row.minutes} minutes played this season ({current_season_source}).")

    _render_defending(league, player.name, selection.team_name)

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
