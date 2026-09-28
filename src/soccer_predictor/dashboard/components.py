"""Reusable table/chart builders for the Streamlit dashboard."""

from __future__ import annotations

import datetime as dt
import urllib.parse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson as poisson_dist

from sqlalchemy.orm import Session

from soccer_predictor.config import League, TableZone, load_manual_captains, load_manual_star_players
from soccer_predictor.ingest.squad import fetch_squad_for_team
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.model.dixon_coles import tau as dixon_coles_tau
from soccer_predictor.model.markets import MatchPrediction
from soccer_predictor.model.scoreline_matrix import build_matrix
from soccer_predictor.model.standings import TeamStanding
from soccer_predictor.model.team_facts import HeadToHeadRecord, TeamFacts


def season_not_yet_scheduled(league: League, season: str, today: dt.date | None = None) -> bool:
    """True for a single_year (tournament-style) league's season that's
    still years away -- e.g. Euro 2028 while it's 2026 -- where
    football-data.org genuinely has nothing to sync yet (confirmed live: a
    404 from its matches endpoint), so suggesting `refresh_live_data.py`
    there would read as actionable but not actually fix anything. Domestic
    leagues (season_display="range") always have a continuously-running
    current season, so this never applies to them.

    `today` defaults to the real current date; overridable for tests.
    """
    if league.season_display != "single_year":
        return False
    today = today if today is not None else dt.date.today()
    return int(season) > today.year


def league_option_label(league: League) -> str:
    """A League selectbox option's display text -- st.selectbox only ever
    renders plain text for its options (no <img>), so a country flag can
    only show up there as a unicode emoji, not the flag_url image used
    elsewhere (e.g. the header below the picker)."""
    return f"{league.flag_emoji} {league.name}" if league.flag_emoji else league.name

# A shared session_state key (see timezone_selector) so picking a timezone
# in any one fixtures table applies to every other one too, rather than
# each table remembering its own independent choice.
TIMEZONE_CHOICES = {
    "Eastern (EST/EDT)": "America/New_York",
    "Central (CST/CDT)": "America/Chicago",
    "Mountain (MST/MDT)": "America/Denver",
    "Pacific (PST/PDT)": "America/Los_Angeles",
    "UTC": "UTC",
    "London (GMT/BST)": "Europe/London",
    "Central Europe (CET/CEST)": "Europe/Berlin",
}
DEFAULT_TIMEZONE_LABEL = "Eastern (EST/EDT)"
_TIMEZONE_SESSION_KEY = "display_timezone_label"


def timezone_selector() -> ZoneInfo:
    """Renders a compact timezone picker and returns the chosen zone.

    Streamlit's dataframe grid has no per-cell click-to-edit -- this
    dropdown is the practical equivalent of "click a date to adjust it":
    one control, shared across every fixtures table via session_state, so
    changing it anywhere updates the displayed kickoff time everywhere.
    """
    st.session_state.setdefault(_TIMEZONE_SESSION_KEY, DEFAULT_TIMEZONE_LABEL)
    label = st.selectbox(
        "Timezone", options=list(TIMEZONE_CHOICES.keys()), key=_TIMEZONE_SESSION_KEY
    )
    return ZoneInfo(TIMEZONE_CHOICES[label])


# A generous estimate (90 min + halftime + typical stoppage time) for how
# long a match stays "live" after kickoff -- there's no real-time score feed
# wired up (see the Upcoming Fixtures caption), so this is a clock-based
# guess rather than the match's actual live/finished status.
LIVE_MATCH_WINDOW = dt.timedelta(hours=2, minutes=15)


def format_kickoff(
    kickoff_utc, fallback_date: dt.date, tz: ZoneInfo, now: dt.datetime | None = None
) -> str:
    """`kickoff_utc` is a naive-but-UTC datetime (or None/NaT for a fixture
    synced before this column existed) -- falls back to date-only display.

    Prefixes a "LIVE" marker when `now` (defaults to the real current time;
    overridable for tests) falls within LIVE_MATCH_WINDOW of kickoff.
    """
    if pd.isna(kickoff_utc):
        return fallback_date.strftime("%b %d, %Y")
    local = kickoff_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
    formatted = local.strftime("%b %d, %Y %I:%M %p %Z")

    now = now if now is not None else dt.datetime.now(dt.UTC).replace(tzinfo=None)
    if kickoff_utc <= now < kickoff_utc + LIVE_MATCH_WINDOW:
        return f"🔴 LIVE · {formatted}"
    return formatted


def youtube_search_url(home_name: str, away_name: str, match_date: dt.date) -> str:
    """A YouTube search-results link for one specific match -- no API key or
    network call, just a query built to reliably surface that match's real
    highlights near the top of YouTube's own results (home team first,
    since that's how highlight videos are usually titled).
    """
    query = f"{home_name} vs {away_name} {match_date.isoformat()} highlights"
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"


def live_sync_requirement_note(league: League | None) -> str:
    """The parenthetical explaining what running refresh_live_data.py needs
    for this league -- football-data.org leagues need an API key; ESPN
    -backed leagues (data_source == "espn", e.g. MLS) need nothing at all.
    Returns "" (not a placeholder) so callers can splice it straight into a
    sentence without a leading space appearing for ESPN leagues. `league`
    may be None at some call sites (e.g. an unresolved league code) -- falls
    back to the football-data.org note in that case, same as this app's
    prior (pre-ESPN) behavior everywhere.
    """
    if league is not None and league.data_source == "espn":
        return ""
    return " (needs FOOTBALL_DATA_ORG_API_KEY in .env)"


_FORM_EMOJI = {"W": "🟩", "D": "⬜", "L": "🟥"}

# Colored square rather than a colored arrow glyph: reliably renders the
# same way everywhere (same trick as _FORM_EMOJI above), unlike arrow emoji
# whose color varies by platform/font and is never actually green or red.
ZONE_MARKER = {"qualify": "🟩", "playoff": "🟧", "relegation": "🟥"}

STANDINGS_DISPLAY_COLUMNS = (
    "Pos",
    "Crest",
    "Team",
    "Pl",
    "W",
    "D",
    "L",
    "GF",
    "GA",
    "GD",
    "Pts",
    "Form",
    "Next",
)

# No separate "Zone" text column (it ate too much width) -- the Pos number
# itself is colored instead, via style_standings below. Same three kinds as
# ZONE_MARKER's emoji legend, just as text colors: qualify=green,
# playoff=amber ("yellow" per the request), relegation=red.
_ZONE_TEXT_COLOR = {
    "qualify": "color: #2e7d32; font-weight: 700",
    "playoff": "color: #f9a825; font-weight: 700",
    "relegation": "color: #c62828; font-weight: 700",
}


def _zone_kind(pos: int, zones: list[TableZone]) -> str:
    for zone in zones:
        if zone.start <= pos <= zone.end:
            return zone.kind
    return ""


def standings_dataframe(
    standings: list[TeamStanding],
    crest_urls: dict[int, str],
    next_opponent_names: dict[int, str],
    zones: list[TableZone] = (),
) -> pd.DataFrame:
    """Includes a `team_id` column for callers that need to map a selected
    row back to a team (e.g. clickable rankings), and a hidden `_zone_kind`
    column ("qualify"/"playoff"/"relegation"/"") for style_standings to color
    the Pos cell by -- pass `column_order=STANDINGS_DISPLAY_COLUMNS` to
    st.dataframe to hide both (a league with no zones configured, e.g. the
    World Cup/Euros, just gets "" for every row, so Pos renders uncolored).
    """
    rows = [
        {
            "team_id": standing.team_id,
            "Pos": pos,
            "_zone_kind": _zone_kind(pos, zones),
            "Crest": crest_urls.get(standing.team_id, ""),
            "Team": standing.team_name,
            "Pl": standing.played,
            "W": standing.won,
            "D": standing.drawn,
            "L": standing.lost,
            "GF": standing.goals_for,
            "GA": standing.goals_against,
            "GD": standing.goal_diff,
            "Pts": standing.points,
            "Form": " ".join(_FORM_EMOJI[r] for r in standing.form),
            "Next": next_opponent_names.get(standing.team_id, ""),
        }
        for pos, standing in enumerate(standings, start=1)
    ]
    return pd.DataFrame(rows)


def _pos_zone_style(row: pd.Series) -> pd.Series:
    styles = pd.Series("", index=row.index)
    color = _ZONE_TEXT_COLOR.get(row.get("_zone_kind", ""), "")
    if color and "Pos" in styles.index:
        styles["Pos"] = color
    return styles


def style_standings(df: pd.DataFrame):
    """Colors the Pos number by zone (see _ZONE_TEXT_COLOR) instead of the
    old separate Zone text column. Pass the returned Styler straight to
    st.dataframe in place of the plain DataFrame; `_zone_kind` stays hidden
    via STANDINGS_DISPLAY_COLUMNS' column_order, same as `team_id` always was.
    """
    return df.style.apply(_pos_zone_style, axis=1)


LEADERBOARD_DISPLAY_COLUMNS = (
    "Team",
    "Attack",
    "Defense (goals-conceded multiplier, lower is better)",
    "Net strength",
)


def leaderboard_dataframe(params: DixonColesParams, team_names: dict[int, str]) -> pd.DataFrame:
    """Includes a `team_id` column for callers that need to map a selected
    row back to a team (e.g. clickable rankings) -- pass
    `column_order=LEADERBOARD_DISPLAY_COLUMNS` to st.dataframe to hide it.
    """
    rows = [
        {
            "team_id": team_id,
            "Team": team_names.get(team_id, f"team#{team_id}"),
            "Attack": params.attack[team_id],
            "Defense (goals-conceded multiplier, lower is better)": params.defense[team_id],
            "Net strength": params.attack[team_id] - params.defense[team_id],
        }
        for team_id in params.attack
    ]
    return pd.DataFrame(rows).sort_values("Net strength", ascending=False).reset_index(drop=True)


def fixture_prediction_row(
    home_name: str, away_name: str, prediction: MatchPrediction
) -> dict:
    top_home, top_away, _ = prediction.top_scorelines[0]
    return {
        "Home": home_name,
        "Away": away_name,
        "P(Home)": round(prediction.home_win, 3),
        "P(Draw)": round(prediction.draw, 3),
        "P(Away)": round(prediction.away_win, 3),
        "Predicted score": f"{top_home}-{top_away}",
        "P(Over 2.5)": round(prediction.over_2_5, 3),
        "P(BTTS)": round(prediction.both_teams_to_score, 3),
        # Hidden (never in a display column_order): styling keys off these,
        # not P(Home)/P(Draw)/P(Away) -- see _predicted_outcome_styles.
        "_predicted_home_goals": top_home,
        "_predicted_away_goals": top_away,
    }


_WINNER_STYLE = "color: #2e7d32; font-weight: 600"  # green
_LOSER_STYLE = "color: #c62828"  # red
_DRAW_STYLE = "color: #9e9e9e"  # grey


def _predicted_outcome_styles(row: pd.Series) -> pd.Series:
    styles = pd.Series("", index=row.index)
    # Keyed off the same top scoreline shown in "Predicted score", not the
    # aggregate P(Home)/P(Draw)/P(Away) split -- the two can legitimately
    # disagree in a Poisson model (a win's probability mass is spread across
    # many scorelines, e.g. 2-1/2-0/3-1, while a draw concentrates on just a
    # few, e.g. 0-0/1-1 -- so "most likely single score" and "most likely
    # aggregate outcome" are different questions). Coloring off the same
    # number the column displays keeps the two from visually contradicting
    # each other.
    home_goals = row["_predicted_home_goals"]
    away_goals = row["_predicted_away_goals"]
    if home_goals > away_goals:
        home_style, away_style = _WINNER_STYLE, _LOSER_STYLE
    elif away_goals > home_goals:
        home_style, away_style = _LOSER_STYLE, _WINNER_STYLE
    else:
        home_style, away_style = _DRAW_STYLE, _DRAW_STYLE
    if "Home" in styles.index:
        styles["Home"] = home_style
    if "Away" in styles.index:
        styles["Away"] = away_style
    return styles


_RESULT_STYLE = {"W": _WINNER_STYLE, "D": _DRAW_STYLE, "L": _LOSER_STYLE}


def _result_style(row: pd.Series) -> pd.Series:
    styles = pd.Series("", index=row.index)
    if "Result" in styles.index:
        styles["Result"] = _RESULT_STYLE.get(row.get("Result", ""), "")
    return styles


def style_match_results(rows: list[dict]):
    """Colors a "Result" column of W/D/L values green/grey/red (same palette
    as style_fixture_predictions' predicted-outcome colors) -- shared by
    Team Detail's "Recent results" table and render_head_to_head's table,
    both of which use this exact W/D/L convention. Pass the returned Styler
    straight to st.dataframe in place of the plain list/DataFrame.
    """
    df = pd.DataFrame(rows)
    return df.style.apply(_result_style, axis=1)


def fixture_columns_with_kickoff_first(row: dict) -> list[str]:
    """Column order for a fixture_prediction_row dict once "Kickoff" has
    been added: Kickoff first, then everything else -- except any hidden
    `_`-prefixed key (see style_fixture_predictions), which never displays.
    """
    return ["Kickoff"] + [c for c in row if c != "Kickoff" and not c.startswith("_")]


_HIDDEN_FIXTURE_COLUMNS = ["_predicted_home_goals", "_predicted_away_goals"]


def style_fixture_predictions(rows: list[dict]):
    """Colors the Home/Away team names green (predicted winner), red
    (predicted loser), or grey (predicted draw) -- based on the same top
    scoreline shown in the "Predicted score" column. Needs the
    `_predicted_home_goals`/`_predicted_away_goals` keys from
    fixture_prediction_row; pass the returned Styler straight to
    st.dataframe/st.table in place of the plain list/DataFrame. The two
    hidden keys are hidden at the Styler level (`.hide(...)`), so this is
    safe to use with st.table too, which has no `column_order` of its own
    to exclude them with -- callers should still put "Kickoff" first via
    fixture_columns_with_kickoff_first() when calling st.dataframe, since
    hiding controls visibility, not column order.
    """
    df = pd.DataFrame(rows)
    styler = df.style.apply(_predicted_outcome_styles, axis=1)
    present = [c for c in _HIDDEN_FIXTURE_COLUMNS if c in df.columns]
    if present:
        styler = styler.hide(axis="columns", subset=present)
    return styler


_SQUAD_REFERENCE_HEIGHT = 180  # fixed + scrollable, regardless of squad size


def _render_squad_reference(
    session: Session | None,
    league: League | None,
    home_team_id: int | None,
    home_name: str,
    away_team_id: int | None,
    away_name: str,
) -> None:
    if session is None or league is None or home_team_id is None or away_team_id is None:
        return
    home_squad = fetch_squad_for_team(session, league, home_team_id)
    away_squad = fetch_squad_for_team(session, league, away_team_id)
    if not home_squad and not away_squad:
        return

    st.caption(
        "Current squads (shown for reference - browsing this list has no effect; only players "
        "flagged injured/absent on Team Detail affect the calculation above):"
    )
    cols = st.columns(2)
    for col, name, squad in ((cols[0], home_name, home_squad), (cols[1], away_name, away_squad)):
        with col:
            st.markdown(f"**{name}**")
            if not squad:
                st.caption("No squad data loaded.")
                continue
            rows = [
                {"Name": p.name, "Position": p.position}
                for p in sorted(squad, key=lambda p: p.name)
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True, height=_SQUAD_REFERENCE_HEIGHT)


def render_team_rating_breakdown(
    params: DixonColesParams, team_id: int, team_name: str, matches_df: pd.DataFrame
) -> None:
    """Explains where a team's standalone attack/defense rating comes from
    -- unlike render_prediction_breakdown's per-fixture lambda math, there's
    no standalone formula for one team's number: it's one solution of a
    joint optimization fit across the whole league at once. So this shows
    the fit's provenance (match count/recency) plus the team's own simple
    W-D-L/goals record from the same underlying results, as a sanity-check
    anchor for the fitted number rather than a derivation of it.
    """
    with st.expander(f"How {team_name}'s rating was calculated", expanded=False):
        st.write(
            f"Fit from **{params.n_matches:,} matches league-wide** (every team at once, not "
            f"just {team_name}), last fitted {params.fitted_at[:10]}. Recent results count for "
            f"more than old ones (time-decay rate ξ = {params.xi:.4f})."
        )
        st.write(
            "This is a joint statistical fit (Poisson/Dixon-Coles maximum likelihood), not a "
            "per-team formula - the optimizer solves for every team's attack/defense "
            "simultaneously so that, together, they best explain the actual final scores across "
            "every match in the league. There's no standalone equation that produces one team's "
            'number in isolation (see a fixture\'s "How this prediction was calculated" for how '
            "two teams' numbers are then combined to predict a specific match)."
        )

        if matches_df.empty:
            st.caption("No historical results on record for this team to compare against.")
            return

        played = len(matches_df)
        wins = draws = losses = goals_for = goals_against = 0
        for row in matches_df.itertuples(index=False):
            is_home = row.home_team_id == team_id
            scored = row.home_goals if is_home else row.away_goals
            conceded = row.away_goals if is_home else row.home_goals
            goals_for += scored
            goals_against += conceded
            if scored > conceded:
                wins += 1
            elif scored == conceded:
                draws += 1
            else:
                losses += 1

        st.write(
            f"**{team_name}'s actual record** (same underlying results the fit uses): "
            f"{played} played, {wins}W-{draws}D-{losses}L, {goals_for} scored "
            f"({goals_for / played:.2f}/game), {goals_against} conceded ({goals_against / played:.2f}/game)."
        )
        st.caption(
            f"Fitted Attack **{params.attack[team_id]:.3f}** / Defense **{params.defense[team_id]:.3f}** "
            "should broadly track these per-game averages - higher attack ≈ scores more than a "
            "league-average team, lower defense ≈ concedes fewer - but the fit also accounts for "
            "opponent strength and recency, so it won't match the raw average exactly."
        )


def render_team_facts(facts: TeamFacts, team_names: dict[int, str], league_name: str) -> None:
    """"Fun facts" computed purely from our own stored match history -- not
    from any API -- for the Team Info section: biggest win/loss on record,
    longest streaks, goal totals, and the most-played opponent.

    Deliberately scoped in the label to `league_name` (not "all-time"/"any
    competition") -- a team here only ever has one league's own results
    (see storage.models.Team's docstring: the same real-world club gets a
    separate row per competition), so e.g. Barcelona's La Liga page can
    never surface a Champions League result like the 2020 8-2 to Bayern --
    that lives under a different competition's data entirely, if we have it
    at all. Overclaiming "all-time" here would just relocate the same
    confusion, not fix it.
    """

    def _opponent_name(opponent_id: int) -> str:
        return team_names.get(opponent_id, f"team#{opponent_id}")

    def _describe(result) -> str:
        venue = "H" if result.is_home else "A"
        return f"{result.goals_for}-{result.goals_against} vs {_opponent_name(result.opponent_id)} ({venue}), {result.date}"

    st.write(
        f"**{league_name} record** (from {facts.total_matches:,} matches on record - this "
        f"league only, not cup/continental competitions this team has also played in): "
        f"{facts.total_goals_for:,} scored, {facts.total_goals_against:,} conceded."
    )
    cols = st.columns(2)
    with cols[0]:
        st.caption("Biggest win: " + (_describe(facts.biggest_win) if facts.biggest_win else "none on record"))
        st.caption(f"Longest win streak: {facts.longest_win_streak} matches")
    with cols[1]:
        st.caption("Biggest loss: " + (_describe(facts.biggest_loss) if facts.biggest_loss else "none on record"))
        st.caption(f"Longest unbeaten streak: {facts.longest_unbeaten_streak} matches")

    w, d, l = facts.most_played_opponent_record
    st.caption(
        f"Most-played opponent: {_opponent_name(facts.most_played_opponent_id)} "
        f"({facts.most_played_opponent_matches} matches, {w}W-{d}D-{l}L)"
    )


def render_star_players(players: list, top_n: int = 3) -> None:
    """Highlights the top scorers from a season's HistoricalPlayerStats list
    (ingest/player_stats.py) as metric cards -- e.g. Haaland's 22 goals for
    Man City -- rather than making the user scan the full stats table for
    who actually stood out. No-op if nobody on the list has scored (some
    goalkeeper-heavy or stats-sparse rosters), since a "top scorer" of 0
    goals isn't a meaningful highlight.
    """
    top_scorers = sorted(players, key=lambda p: p.goals or 0, reverse=True)[:top_n]
    top_scorers = [p for p in top_scorers if (p.goals or 0) > 0]
    if not top_scorers:
        return

    st.markdown("**⭐ Top scorers this season**")
    cols = st.columns(len(top_scorers))
    for col, p in zip(cols, top_scorers):
        with col:
            st.metric(p.name, f"{p.goals} goals", f"{p.assists or 0} assists")
            detail = p.position
            if p.rating:
                detail += f" · Rating {p.rating:.2f}"
            st.caption(detail)


def compute_automatic_stars(weights: dict[str, float], top_n: int = 2, min_weight: float = 0.5) -> set[str]:
    """Player names from a resolve_team_importance_weights() result that
    qualify as an automatic "star" badge -- the top `top_n` by weight,
    provided they actually clear `min_weight` (roughly the midpoint of
    model/player_importance.py's MIN/MAX_IMPORTANCE_WEIGHT 0.05-0.95 range),
    so a genuinely middling squad with no real standout gets zero forced
    stars rather than two arbitrary ones.
    """
    ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    return {name for name, weight in ranked[:top_n] if weight >= min_weight}


def render_player_badges(name: str, team_name: str, automatic_stars: set[str] | None = None) -> str:
    """Prefixes `name` with a captain crown and/or star badge for display --
    NOT related to render_star_players() above (that's an ephemeral "top-3
    scorers in the currently-loaded stats table" highlight; this is a
    persistent captain/world-class badge looked up from config/captains.yaml
    and config/star_players.yaml, unioned with an optional stats-based
    `automatic_stars` set from compute_automatic_stars()).

    Matches team name exactly and player name case-insensitively, mirroring
    prediction/service.py::get_injuries_for_team's manual-entry matching.
    """
    is_captain = any(
        c.team == team_name and c.player.lower() == name.lower() for c in load_manual_captains()
    )
    is_star = any(
        s.team == team_name and s.player.lower() == name.lower() for s in load_manual_star_players()
    ) or (automatic_stars is not None and name in automatic_stars)

    prefix = ("👑 " if is_captain else "") + ("⭐ " if is_star else "")
    return f"{prefix}{name}"


def render_head_to_head(h2h: HeadToHeadRecord, team_name: str, opponent_name: str) -> None:
    """Head-to-head record between `team_name` and one specific opponent --
    shown when a fixture row is clicked on Team Detail's own schedule, so
    "Man City vs Liverpool" surfaces their history against each other, not
    the generic prediction breakdown's league-wide numbers.
    """
    st.markdown(f"**{team_name} vs {opponent_name}: head-to-head**")
    st.write(
        f"{h2h.total_matches} meeting{'s' if h2h.total_matches != 1 else ''} on record - "
        f"**{h2h.wins}W-{h2h.draws}D-{h2h.losses}L**, {h2h.goals_for}-{h2h.goals_against} goals "
        f"(from {team_name}'s side)."
    )
    rows = []
    for m in h2h.recent_matches:
        home_name, away_name = (team_name, opponent_name) if m.is_home else (opponent_name, team_name)
        rows.append(
            {
                "Date": m.date,
                "Venue": "Home" if m.is_home else "Away",
                "Score": f"{m.goals_for}-{m.goals_against}",
                "Result": "W" if m.goal_diff > 0 else ("D" if m.goal_diff == 0 else "L"),
                "Watch": youtube_search_url(home_name, away_name, m.date),
            }
        )
    st.dataframe(
        style_match_results(rows),
        use_container_width=True,
        hide_index=True,
        column_config={"Watch": st.column_config.LinkColumn("Watch", display_text="▶ Highlights")},
    )


def render_prediction_breakdown(
    prediction: MatchPrediction,
    home_name: str,
    away_name: str,
    session: Session | None = None,
    league: League | None = None,
    home_team_id: int | None = None,
    away_team_id: int | None = None,
) -> None:
    """Shows every number behind one prediction -- attack/defense ratings,
    the lambda (expected goals) formulas, the Poisson formula, the
    Dixon-Coles tau correction, and a fully worked example for the model's
    top scoreline -- so the actual math can be redone by hand, not just
    trusted as a black box. No-op if `prediction` has no breakdown attached
    (e.g. a league with no trained model never reaches this).

    Pass `session`/`league`/`home_team_id`/`away_team_id` to also show each
    team's current squad for reference -- attack/defense come entirely from
    historical TEAM match results, never from individual player data. The
    squad list itself is shown purely as context (see the "0." section
    below): browsing it has no effect. A player only matters to the
    calculation once they're actually flagged injured/absent (Team Detail's
    Injuries section, sourced from config/injuries.yaml, the live API, or
    chat notes) -- at which point their OWN stats size the adjustment (see
    injury_adjustment.adjust_strength / ingest.player_importance).
    """
    b = prediction.breakdown
    if b is None:
        return

    with st.expander(f"How this prediction was calculated: {home_name} vs {away_name}", expanded=True):
        st.markdown("**Where these ratings actually come from**")
        st.write(
            f"Attack/defense are fit **only from past match results** (final scores) - "
            f"{b.n_matches:,} matches across this whole league (every team at once, not just "
            f"{home_name}/{away_name}), last fitted {b.fitted_at[:10]}. This base rating never "
            f"looks at individual player data - only how many goals this team has scored and "
            f"conceded historically. Recent results count for more than old ones (time-decay rate "
            f"ξ = {b.xi:.4f})."
        )
        st.write(
            "Player data *does* feed into the adjusted numbers in step 1 below, but only for "
            "whoever is actually flagged injured/absent on Team Detail's Injuries section - "
            "simply being on the squad has no effect. Once a player is marked out, the *size* of "
            "the hit to their team's attack/defense is based on that player's own real "
            "goals/assists/minutes (and, when available, current-season xG/xA from Understat), "
            "not a flat guess - see Team Detail's Injuries section for each flagged player's "
            "computed weight."
        )
        _render_squad_reference(session, league, home_team_id, home_name, away_team_id, away_name)

        st.markdown("**1. Team strength going into this match**")
        cols = st.columns(2)
        for col, name, attack, base_attack, defense, base_defense in (
            (cols[0], f"{home_name} (home)", b.home_attack, b.base_home_attack, b.home_defense, b.base_home_defense),
            (cols[1], f"{away_name} (away)", b.away_attack, b.base_away_attack, b.away_defense, b.base_away_defense),
        ):
            with col:
                st.markdown(f"**{name}**")
                if attack != base_attack:
                    col.write(f"Attack: {base_attack:.3f} → **{attack:.3f}** (adjusted for injuries/form)")
                else:
                    col.write(f"Attack: **{attack:.3f}**")
                if defense != base_defense:
                    col.write(f"Defense: {base_defense:.3f} → **{defense:.3f}** (adjusted; lower = stronger)")
                else:
                    col.write(f"Defense: **{defense:.3f}** (lower = stronger)")
        st.caption(
            "Ratings are fitted from every past result: higher attack scores more goals, lower "
            "defense concedes fewer (it's a goals-conceded multiplier). See Team Detail's "
            '"Current rating" or Predictor\'s Team Strength Leaderboard tab.'
        )

        st.markdown("**2. Expected goals (λ) for this matchup**")
        st.latex(r"\lambda_{home} = \text{attack}_{home} \times \text{defense}_{away} \times \gamma")
        st.write(
            f"= {b.home_attack:.3f} × {b.away_defense:.3f} × {b.home_advantage:.3f} (home advantage γ) "
            f"= **{b.lambda_home:.3f} expected goals**"
        )
        st.latex(r"\lambda_{away} = \text{attack}_{away} \times \text{defense}_{home}")
        st.write(f"= {b.away_attack:.3f} × {b.home_defense:.3f} = **{b.lambda_away:.3f} expected goals**")

        st.markdown("**3. Poisson probability for each possible scoreline**")
        st.latex(r"P(\text{goals} = k) = \frac{e^{-\lambda}\,\lambda^{k}}{k!}")
        st.write(
            "Applied independently to home goals (using λ_home) and away goals (using λ_away), "
            "then multiplied together for the joint probability of every home-away combination "
            "(0-0, 1-0, 0-1, 1-1, 2-0, ...)."
        )

        st.markdown("**4. Dixon-Coles low-score correction (τ)**")
        st.write(
            f"Independent Poisson under-predicts low-scoring draws in real football, so four "
            f"scorelines get nudged by ρ = **{b.rho:.3f}** (fit from this league's actual results):"
        )
        st.table(
            [
                {"Scoreline": "0-0", "τ multiplier": f"1 − ρ = {1 - b.rho:.3f}"},
                {"Scoreline": "1-0", "τ multiplier": f"1 + ρ = {1 + b.rho:.3f}"},
                {"Scoreline": "0-1", "τ multiplier": f"1 + ρ = {1 + b.rho:.3f}"},
                {"Scoreline": "1-1", "τ multiplier": f"1 − ρ = {1 - b.rho:.3f}"},
                {"Scoreline": "any other scoreline", "τ multiplier": "1.0 (no adjustment)"},
            ]
        )

        st.markdown("**5. Worked example: the model's single most likely scoreline**")
        h, a, _ = prediction.top_scorelines[0]
        p_h = float(poisson_dist.pmf(h, b.lambda_home))
        p_a = float(poisson_dist.pmf(a, b.lambda_away))
        tau_val = float(dixon_coles_tau(np.array([h]), np.array([a]), b.rho)[0])
        raw_joint = p_h * p_a * tau_val
        matrix = build_matrix(b.lambda_home, b.lambda_away, b.rho)
        st.write(f"P({home_name} scores {h}) = e^(−{b.lambda_home:.3f}) × {b.lambda_home:.3f}^{h} / {h}! = **{p_h:.4f}**")
        st.write(f"P({away_name} scores {a}) = e^(−{b.lambda_away:.3f}) × {b.lambda_away:.3f}^{a} / {a}! = **{p_a:.4f}**")
        st.write(f"τ({h}-{a}, ρ={b.rho:.3f}) = **{tau_val:.3f}**")
        st.write(f"Joint (before normalizing) = {p_h:.4f} × {p_a:.4f} × {tau_val:.3f} = {raw_joint:.5f}")
        st.write(
            f"Every scoreline from 0-0 up to 10-10 is computed the same way, then all of them are "
            f"divided by their total so the full grid sums to 1. That gives this scoreline's final "
            f"probability: **{matrix[h, a]:.3f}** - the same number shown in the table's "
            f'"Predicted score" column ({h}-{a}).'
        )

        st.markdown("**6. Match outcome / totals markets**")
        st.write(
            "Home win / Draw / Away win are that same scoreline grid summed over every cell where "
            "home > away, home == away, or home < away respectively. Over/Under 2.5 sums every cell "
            "where total goals is above/below 2.5, and Both Teams To Score sums every cell where "
            "both sides scored at least once."
        )


def scoreline_chart(prediction: MatchPrediction, home_name: str, away_name: str) -> go.Figure:
    labels = [f"{h}-{a}" for h, a, _ in prediction.top_scorelines]
    probs = [p for _, _, p in prediction.top_scorelines]
    fig = go.Figure(go.Bar(x=labels, y=probs))
    fig.update_layout(
        title=f"Most likely scorelines: {home_name} vs {away_name}",
        xaxis_title="Scoreline",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
        height=320,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig
