"""One match's page: lineups with formation, team stats and a goals/cards/subs
timeline, from ESPN's match summary, plus a "More on Google" button. Works for
any league with an ESPN slug (config/leagues.yaml). Reached from the live
banner, a selected row on Team Detail's recent results, or the Leagues page's
upcoming matches.

URLs (see resolve_match):
    /match_detail?league=<code>&event=<ESPN event id>
    /match_detail?league=<code>&home_name=<name>&away_name=<name>&date=<YYYY-MM-DD>
    /match_detail?league=<code>&home=<team id>&away=<team id>&date=<YYYY-MM-DD>
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from soccer_predictor.config import League, load_leagues
from soccer_predictor.ingest import espn_client
from soccer_predictor.ingest.espn_client import PLAYER_STAT_CHIPS, LineupPlayer, MatchDetail, TeamMatchInfo
from soccer_predictor.ingest.news import match_search_url
from soccer_predictor.storage.db import session_scope
from soccer_predictor.storage.repository import espn_event_id_for_match, match_exists, team_by_id

# Team stats worth showing, in order (ESPN's own stat names; the label shown
# is ESPN's, so a new wording from ESPN carries through).
TEAM_STAT_ORDER = [
    "possessionPct",
    "totalShots",
    "shotsOnTarget",
    "blockedShots",
    "wonCorners",
    "offsides",
    "foulsCommitted",
    "yellowCards",
    "redCards",
    "saves",
    "passPct",
    "accuratePasses",
    "totalPasses",
    "totalCrosses",
    "totalTackles",
    "interceptions",
    "totalClearance",
]
EVENT_ICON = {"goal": "⚽", "yellow": "🟨", "red": "🟥", "sub": "🔁"}

# A banner click is about a match under way or just finished, so the day's
# scoreboard it is looked up on must not be the 6h-old copy.
_NAME_LOOKUP_TTL_SECONDS = 120
# The page is public and its URL carries user-supplied text, so every lookup it
# can trigger is confined to matches that really exist (see resolve_match):
# without that, anyone could walk through dates or event ids and make the server
# fetch -- and store on disk -- an endless run of ESPN responses.
MAX_NAME_LENGTH = 80
# The live banner shows matches from yesterday and today (UTC); a link by team
# names is only ever made for those, so anything outside that is not ours.
NAME_LOOKUP_DAYS_BACK = 2
NAME_LOOKUP_DAYS_AHEAD = 1

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>~$:-])")


def plain(text: str | None) -> str:
    """`text` with markdown stripped of its meaning, for Streamlit calls that
    render markdown (subheader, caption, markdown). Team names and venues come
    from the URL or an external feed; as written they could carry a link or an
    image into our page."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text or "")


def _recent_event_ids(slug: str) -> set[str]:
    """ESPN event ids of the league's matches yesterday and today (UTC) -- the
    window the live banner links from. Read from the scoreboard the banner has
    just fetched, so this normally costs no request."""
    today = dt.datetime.now(dt.UTC).date()
    ids: set[str] = set()
    for day in (today - dt.timedelta(days=1), today):
        ids.update(
            m.event_id
            for m in espn_client.fetch_day_matches(slug, day, cache_ttl_seconds=_NAME_LOOKUP_TTL_SECONDS)
            if m.event_id
        )
    return ids


def _recent_enough_for_name_lookup(match_date: dt.date) -> bool:
    today = dt.datetime.now(dt.UTC).date()
    return today - dt.timedelta(days=NAME_LOOKUP_DAYS_BACK) <= match_date <= today + dt.timedelta(days=NAME_LOOKUP_DAYS_AHEAD)


def match_detail_query_params(league_code: str, home_id: int, away_id: int, date: dt.date) -> dict[str, str]:
    return {"league": league_code, "home": str(home_id), "away": str(away_id), "date": date.isoformat()}


def render_match_detail_button(league: League, home_id: int, away_id: int, date: dt.date, key: str) -> None:
    """A "Match details" button that opens this page for the given match.
    Does nothing for leagues with no ESPN slug."""
    if not league.espn_league_slug:
        return
    # Imported here, not at module level: navigation imports every view (this
    # one included), so a top-level import would be circular.
    from soccer_predictor.dashboard import navigation

    if st.button("Match details →", key=key):
        st.switch_page(
            navigation.match_detail_page(),
            query_params=match_detail_query_params(league.code, home_id, away_id, date),
        )


def _int_param(name: str) -> int | None:
    try:
        return int(st.query_params.get(name))
    except (TypeError, ValueError):
        return None


def _event_param() -> str | None:
    """ESPN's event id from the URL -- digits only, since it goes into a request."""
    raw = st.query_params.get("event", "")
    return raw if re.fullmatch(r"\d{1,12}", raw) else None


def _date_param() -> dt.date | None:
    try:
        return dt.date.fromisoformat(st.query_params.get("date", ""))
    except ValueError:
        return None


def _state_hint(match_date: dt.date) -> str:
    """Cache lifetime hint when ESPN's own state isn't known."""
    today = dt.datetime.now(dt.UTC).date()
    if match_date > today:
        return "pre"
    return "in" if match_date == today else "post"


_SINGULAR = {"goals": "goal", "assists": "assist", "shots": "shot", "saves": "save"}


def _stat_chip(name: str, value: int) -> str:
    label = PLAYER_STAT_CHIPS[name]
    return f"{value} {_SINGULAR.get(label, label) if value == 1 else label}"


def player_events_text(player: LineupPlayer) -> str:
    parts = [_stat_chip(name, value) for name, value in player.stats.items()]
    if player.subbed_out:
        parts.append("subbed off")
    if player.subbed_in and not player.starter:
        parts.append("came on")
    return ", ".join(parts)


def lineup_frame(players: list[LineupPlayer]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"#": p.jersey, "Player": p.name, "Pos": p.position, "Match": player_events_text(p)} for p in players],
        columns=["#", "Player", "Pos", "Match"],
    )


def team_stats_frame(detail: MatchDetail) -> pd.DataFrame:
    """One row per stat both teams report, ordered by TEAM_STAT_ORDER."""
    rows = []
    for name in TEAM_STAT_ORDER:
        home_stat, away_stat = detail.home.stats.get(name), detail.away.stats.get(name)
        if home_stat is None or away_stat is None:
            continue
        rows.append({detail.home.name: home_stat[1], "Stat": home_stat[0], detail.away.name: away_stat[1]})
    return pd.DataFrame(rows, columns=[detail.home.name, "Stat", detail.away.name])


def timeline_frame(detail: MatchDetail) -> pd.DataFrame:
    team_name = {detail.home.espn_id: detail.home.name, detail.away.espn_id: detail.away.name}
    return pd.DataFrame(
        [
            {
                "Min": e.minute,
                "Team": team_name.get(e.team_id, ""),
                "Event": f"{EVENT_ICON.get(e.kind, '')} {e.text}".strip(),
            }
            for e in detail.events
        ],
        columns=["Min", "Team", "Event"],
    )


def _render_lineup(team: TeamMatchInfo) -> None:
    st.markdown(f"**{plain(team.name)}**" + (f" · {plain(team.formation)}" if team.formation else ""))
    starters = [p for p in team.players if p.starter]
    bench = [p for p in team.players if not p.starter]
    if starters:
        st.dataframe(lineup_frame(starters), hide_index=True, use_container_width=True)
    if bench:
        st.caption("Substitutes")
        st.dataframe(lineup_frame(bench), hide_index=True, use_container_width=True)


@dataclass
class ResolvedMatch:
    """What a match detail URL points at, before ESPN is asked for the detail."""

    home_name: str | None = None
    away_name: str | None = None
    home_crest: str | None = None
    away_crest: str | None = None
    match_date: dt.date | None = None
    event_id: str | None = None
    state: str = "in"  # cache lifetime hint only, see _state_hint


def resolve_match(session, league: League) -> ResolvedMatch | None:
    """Works out which ESPN event the URL means. Three kinds of link exist:
    `event=<ESPN id>` (the live banner, for ESPN-sourced matches),
    `home_name`/`away_name`/`date` (the banner, for football-data.org
    matches) and `home`/`away` team ids with `date` (a row on Team Detail or
    the Leagues page). None when the URL names no match at all."""
    slug = league.espn_league_slug
    match_date = _date_param()
    event_id = _event_param()
    if event_id is not None:
        # Only a match the live banner could be linking to; any other number is
        # not a request we should make.
        if event_id not in _recent_event_ids(slug):
            return None
        return ResolvedMatch(event_id=event_id, match_date=match_date)

    home_id, away_id = _int_param("home"), _int_param("away")
    if home_id is not None and away_id is not None and match_date is not None:
        home, away = team_by_id(session, home_id), team_by_id(session, away_id)
        if home is None or away is None or not match_exists(session, league.code, match_date, home_id, away_id):
            return None
        resolved = ResolvedMatch(
            home_name=home.canonical_name,
            away_name=away.canonical_name,
            home_crest=home.crest_url,
            away_crest=away.crest_url,
            match_date=match_date,
            state=_state_hint(match_date),
        )
        resolved.event_id = espn_event_id_for_match(session, league.code, match_date, home_id, away_id)
        if resolved.event_id is None and home.espn_team_id is not None and away.espn_team_id is not None:
            found = espn_client.find_event_id(slug, match_date, str(home.espn_team_id), str(away.espn_team_id))
            if found is not None:
                resolved.event_id, resolved.state = found
        if resolved.event_id is None:
            found = espn_client.find_event_by_names(slug, match_date, resolved.home_name, resolved.away_name)
            if found is not None:
                resolved.event_id, resolved.state = found
        return resolved

    home_name, away_name = st.query_params.get("home_name", ""), st.query_params.get("away_name", "")
    if (
        home_name
        and away_name
        and len(home_name) <= MAX_NAME_LENGTH
        and len(away_name) <= MAX_NAME_LENGTH
        and match_date is not None
        and _recent_enough_for_name_lookup(match_date)
    ):
        found = espn_client.find_event_by_names(
            slug, match_date, home_name, away_name, cache_ttl_seconds=_NAME_LOOKUP_TTL_SECONDS
        )
        if found is None:
            return None  # the text in the URL is only ever shown once ESPN has confirmed such a match
        return ResolvedMatch(
            home_name=home_name, away_name=away_name, match_date=match_date, event_id=found[0], state=found[1]
        )
    return None


def _render_header(detail: MatchDetail | None, resolved: ResolvedMatch) -> None:
    home_name = resolved.home_name or (detail.home.name if detail else "Home")
    away_name = resolved.away_name or (detail.away.name if detail else "Away")
    home_crest = resolved.home_crest or (detail.home.logo if detail else None)
    away_crest = resolved.away_crest or (detail.away.logo if detail else None)
    cols = st.columns([3, 2, 3])
    for col, name, crest in ((cols[0], home_name, home_crest), (cols[2], away_name, away_crest)):
        if crest:
            col.image(crest, width=64)
        col.subheader(plain(name))
    if detail is not None and detail.home_score is not None and detail.away_score is not None:
        cols[1].markdown(f"## {detail.home_score} - {detail.away_score}")
    else:
        cols[1].markdown("## vs")
    if detail is not None:
        cols[1].caption(plain(detail.status_label))
    elif resolved.match_date is not None:
        cols[1].caption(resolved.match_date.strftime("%d %b %Y"))
    if detail is not None:
        place = ", ".join(x for x in (detail.venue, detail.city) if x)
        if detail.attendance:
            place += f" · attendance {detail.attendance:,}" if place else f"Attendance {detail.attendance:,}"
        if place:
            st.caption(plain(place))


def render() -> None:
    from soccer_predictor.dashboard import navigation

    league = load_leagues().get(st.query_params.get("league", ""))
    if league is None or not league.espn_league_slug:
        st.info("Open a match from the live banner, a team's recent results or a league's upcoming matches.")
        if st.button("← Back to Leagues"):
            st.switch_page(navigation.home_page())
        st.stop()

    if st.button(f"← Back to {league.name}"):
        st.switch_page(navigation.home_page(), query_params={"league": league.code})

    with session_scope() as session:
        resolved = resolve_match(session, league)
    if resolved is None:
        st.info("That match couldn't be found. Live-banner links only work while the match is on the banner.")
        st.stop()

    detail = (
        espn_client.fetch_match_detail(league.espn_league_slug, resolved.event_id, resolved.state)
        if resolved.event_id
        else None
    )
    _render_header(detail, resolved)

    home_name = resolved.home_name or (detail.home.name if detail else None)
    away_name = resolved.away_name or (detail.away.name if detail else None)
    if home_name and away_name:
        st.link_button("More on Google ↗", match_search_url(home_name, away_name, league.name))

    if detail is None:
        st.info("Match details aren't available from ESPN for this game right now. Try again shortly.")
        return

    lineups_tab, stats_tab, timeline_tab = st.tabs(["Lineups", "Team stats", "Timeline"])
    with lineups_tab:
        if detail.has_lineups:
            left, right = st.columns(2)
            with left:
                _render_lineup(detail.home)
            with right:
                _render_lineup(detail.away)
        else:
            st.caption("Lineups usually appear about an hour before kickoff.")
    with stats_tab:
        stats = team_stats_frame(detail)
        if stats.empty:
            st.caption("Team stats appear once the match is under way.")
        else:
            st.dataframe(stats, hide_index=True, use_container_width=True)
    with timeline_tab:
        timeline = timeline_frame(detail)
        if timeline.empty:
            st.caption("No goals, cards or substitutions yet.")
        else:
            st.dataframe(timeline, hide_index=True, use_container_width=True)
    st.caption("Match data via ESPN.")
