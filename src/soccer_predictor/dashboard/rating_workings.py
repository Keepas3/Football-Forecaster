"""The panel under the Predictor's Team Strength Leaderboard that shows how one of a
team's figures is built -- opened by the figure's link in the table, which names the
team and the figure in the `rating` URL parameter.

Each figure has its own walk-through: attack (the fitted rating from results, then the
nudge for the chances the team creates), defense (the fitted rating from goals
conceded, which nothing adjusts) and net strength (attack minus defense, beside the
clubs just above and below). The builders here are plain functions of the numbers and
write the text; render_rating_workings fetches the inputs and draws them.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from soccer_predictor.config import League
from soccer_predictor.dashboard.components import (
    LEADERBOARD_DEFENSE_COLUMN,
    RATING_PANEL_CSS,
    RATING_PARAM,
    SCROLL_TO_RATING_JS,
    TableCell,
    TableColumn,
    _escape_markdown,
    _signed_number,
    sortable_table_html,
    team_page_href,
)
from soccer_predictor.dashboard.squad_strength import TeamStrength, ordinal, team_strength
from soccer_predictor.ingest.lineups import expected_squad
from soccer_predictor.ingest.player_importance import (
    MIN_MATCHES_FOR_CURRENT_FORM,
    CurrentAttackDetail,
    resolve_current_attack_detail,
)
from soccer_predictor.ingest.understat_client import UNDERSTAT_LEAGUE_SLUG
from soccer_predictor.model.current_form_adjustment import CURRENT_FORM_IMPACT_CAP, current_attack_shift
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.model.lineup_adjustment import ExpectedPlayer
from soccer_predictor.model.standings import TeamStanding, compute_standings
from soccer_predictor.storage.db import session_scope
from soccer_predictor.storage.repository import matches_for_league

TOP_CONTRIBUTORS = 5
NEIGHBOURS_EACH_SIDE = 2


@dataclass(frozen=True)
class Step:
    heading: str
    paragraphs: tuple[str, ...]  # markdown, names already escaped
    table_html: str = ""  # shown after the paragraphs


@dataclass(frozen=True)
class RatingWorkings:
    heading: str
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class SeasonRecord:
    """A team's goals this season next to the league's, per game."""

    played: int
    scored_per_game: float
    conceded_per_game: float
    league_per_game: float  # what the average team scores (and concedes) in a game


def season_record(standings: Sequence[TeamStanding], team_id: int) -> SeasonRecord | None:
    """None if the team has not played this season (or nobody has)."""
    mine = next((s for s in standings if s.team_id == team_id), None)
    team_games = sum(s.played for s in standings)
    if mine is None or mine.played == 0 or team_games == 0:
        return None
    return SeasonRecord(
        played=mine.played,
        scored_per_game=mine.goals_for / mine.played,
        conceded_per_game=mine.goals_against / mine.played,
        league_per_game=sum(s.goals_for for s in standings) / team_games,
    )


def _percent_off_average(rating: float) -> str:
    """"34% more" or "12% fewer": how far `rating` (1.00 = an average team) is from the average."""
    share = (rating - 1) * 100
    return f"{abs(share):.0f}% {'more' if share >= 0 else 'fewer'}"


def _fit_paragraph(params: DixonColesParams) -> str:
    recency = (
        f", counting recent matches more (one about {math.log(2) / params.xi:.0f} days old counts half as much as one today)"
        if params.xi > 0
        else ""
    )
    return (
        f"The model fits every team's attack and defense together, from **{params.n_matches:,} matches** in this "
        f"competition{recency}. It finds the ratings that best explain all those scores at once, allowing for the "
        "strength of the opponents each team has faced."
    )


def _contributors_table(squad: Sequence[ExpectedPlayer]) -> str:
    """The squad's biggest sources of xG + 0.7 x xA, then everyone else added together."""
    scored = sorted(squad, key=lambda p: (-(p.xg + 0.7 * p.xa), p.name))
    top = [p for p in scored[:TOP_CONTRIBUTORS] if p.xg + 0.7 * p.xa > 0]
    if not top:
        return ""
    rest = scored[len(top) :]
    columns = [
        TableColumn("Player", left=True),
        TableColumn("xG"),
        TableColumn("xA"),
        TableColumn("xG + 0.7 × xA", hint="Expected goals plus 0.7 times expected assists"),
    ]
    rows = [
        [
            TableCell(html.escape(p.name)),
            TableCell(f"{p.xg:.1f}"),
            TableCell(f"{p.xa:.1f}"),
            TableCell(f"<b>{p.xg + 0.7 * p.xa:.1f}</b>"),
        ]
        for p in top
    ]
    if rest:
        xg, xa = sum(p.xg for p in rest), sum(p.xa for p in rest)
        rows.append(
            [
                TableCell(f"The other {len(rest)} players"),
                TableCell(f"{xg:.1f}"),
                TableCell(f"{xa:.1f}"),
                TableCell(f"<b>{xg + 0.7 * xa:.1f}</b>"),
            ]
        )
    return sortable_table_html(columns, rows, compact=True)


def attack_workings(
    team_name: str,
    strength: TeamStrength,
    params: DixonColesParams,
    record: SeasonRecord | None,
    detail: CurrentAttackDetail | None,
    chances_covered: bool,
    squad: Sequence[ExpectedPlayer] = (),
) -> RatingWorkings:
    """How the leaderboard's Attack figure is built: the fitted rating from results
    (`strength.attack_results_only`), then, when the competition's chance data gives a
    reading for this club (`detail`), the capped nudge that turns it into
    `strength.attack`. `chances_covered` says whether the competition has such data
    at all, to explain a missing `detail`."""
    name = _escape_markdown(team_name)
    base, adjusted = strength.attack_results_only, strength.attack

    results = [
        _fit_paragraph(params),
        f"A rating of 1.00 is a league-average attack. {name}'s is **{base:.2f}**, so it scores about "
        f"{_percent_off_average(base)} goals than an average team would against the same opposition. "
        "This is the *Attack (results only)* column.",
    ]
    if record is not None:
        results.append(
            f"As a check on the fit: {name} has scored **{record.scored_per_game:.2f}** goals a game in "
            f"{record.played} games this season, and the average team in the league {record.league_per_game:.2f}."
        )
    steps = [Step("1. Fitted from results", tuple(results))]

    if detail is None:
        why = (
            "There is no per-player chance data for this competition"
            if not chances_covered
            else (
                f"The chance data can't be used for {name} yet: it needs at least {MIN_MATCHES_FOR_CURRENT_FORM} "
                "matches played this season, and the club found in the data source"
            )
        )
        steps.append(
            Step(
                "2. Adjusted for chances created",
                (f"{why}, so the attack is not adjusted: it is the same as the results-only figure, **{adjusted:.2f}**.",),
            )
        )
    else:
        total = detail.team_rate * detail.matches_played
        steps.append(
            Step(
                "2. Chances created this season",
                (
                    "Goals are a slow signal, so the model also looks at the chances a team creates. Every player's "
                    "expected goals (xG) plus 0.7 × his expected assists (xA) are added up for the squad and divided by "
                    "the matches played.",
                    f"{name}: **{total:.1f}** over {detail.matches_played} matches = **{detail.team_rate:.2f} a match**. "
                    f"The average of the league's {detail.teams_compared} clubs is **{detail.league_mean_rate:.2f}**, "
                    f"so {name} creates **{detail.ratio:.2f}×** the average."
                    + (" The biggest contributors:" if squad else ""),
                ),
                _contributors_table(squad),
            )
        )
        shift = current_attack_shift(detail.ratio)
        steps.append(
            Step(
                "3. The adjustment",
                (
                    f"That ratio becomes a small nudge, capped at ±{CURRENT_FORM_IMPACT_CAP:.0%} so a hot or cold run of "
                    f"chances can't outweigh what the results say: ({detail.ratio:.2f} − 1) = {_signed_number(detail.ratio - 1)}, "
                    f"kept between −1 and +1, × {CURRENT_FORM_IMPACT_CAP:.0%} = **{_signed_number(shift * 100, 1)}%**.",
                    f"Adjusted attack = {base:.2f} × (1 {'+' if shift >= 0 else '−'} {abs(shift):.3f}) = **{adjusted:.2f}**. "
                    "This is the *Attack* column.",
                ),
            )
        )
    return RatingWorkings(f"How {name}'s attack is calculated", tuple(steps))


def defense_workings(
    team_name: str, strength: TeamStrength, params: DixonColesParams, record: SeasonRecord | None
) -> RatingWorkings:
    """How the leaderboard's Defense figure is built: the fitted rating from goals
    conceded, which nothing adjusts."""
    name = _escape_markdown(team_name)
    defense = strength.defense

    fitted = [
        _fit_paragraph(params),
        f"A rating of 1.00 is an average defense and **lower is better**. {name}'s is **{defense:.2f}**, so opponents "
        f"score about {_percent_off_average(defense)} goals against it than against an average defense.",
    ]
    if record is not None:
        fitted.append(
            f"As a check on the fit: {name} has conceded **{record.conceded_per_game:.2f}** goals a game in "
            f"{record.played} games this season, and the average team in the league {record.league_per_game:.2f}."
        )
    return RatingWorkings(
        f"How {name}'s defense is calculated",
        (
            Step("1. Fitted from goals conceded", tuple(fitted)),
            Step(
                "2. Nothing is added to it",
                (
                    "Unlike attack, defense is not adjusted for the chances a team creates: the player data (Understat, "
                    "American Soccer Analysis) measures attacking output only. The leaderboard shows the fitted rating "
                    "exactly as it is. Injuries and announced lineups change a defense only inside one match's prediction.",
                ),
            ),
            Step(
                "3. How it is used",
                (
                    f"An opponent's expected goals against {name} = its attack × **{defense:.2f}**, and × "
                    f"{params.home_advantage:.2f} (the home advantage) as well when the opponent is the home side.",
                ),
            ),
        ),
    )


def _neighbours_table(rows: list[dict], index: int, league_code: str) -> str:
    """The clubs from `NEIGHBOURS_EACH_SIDE` above `index` to as many below, with the gap in net strength."""
    start = max(0, index - NEIGHBOURS_EACH_SIDE)
    columns = [
        TableColumn("#"),
        TableColumn("Team", left=True),
        TableColumn("Attack"),
        TableColumn("Defense"),
        TableColumn("Net strength"),
        TableColumn("Gap", hint="Their net strength minus this team's"),
    ]
    table = []
    for position in range(start, min(len(rows), index + NEIGHBOURS_EACH_SIDE + 1)):
        row = rows[position]
        own = position == index
        team = (
            f'<a class="team" href="{html.escape(team_page_href(league_code, row["team_id"]))}" '
            f'target="_self">{html.escape(row["Team"])}</a>'
        )
        net = row["Net strength"]
        cells = [
            str(position + 1),
            team,
            f"{row['Attack']:.2f}",
            f"{row[LEADERBOARD_DEFENSE_COLUMN]:.2f}",
            _signed_number(net),
            "" if own else _signed_number(net - rows[index]["Net strength"]),
        ]
        table.append([TableCell(f"<b>{text}</b>" if own else text) for text in cells])
    return sortable_table_html(columns, table, compact=True, selected=index - start)


def net_workings(team_name: str, strength: TeamStrength, leaderboard: pd.DataFrame, league_code: str) -> RatingWorkings:
    """How the leaderboard's Net strength is built: attack minus defense, and where that
    puts the club among its neighbours on the board."""
    name = _escape_markdown(team_name)
    rows = leaderboard.to_dict("records")
    steps = [
        Step(
            "1. Attack minus defense",
            (
                f"Net strength = adjusted attack − defense = {strength.attack:.2f} − {strength.defense:.2f} = "
                f"**{_signed_number(strength.net)}**. A higher figure means a team that scores more and concedes less. "
                "Click an Attack or Defense figure in the table to see how each one is built.",
            ),
        )
    ]
    if strength.rank is not None:
        steps.append(
            Step(
                "2. Where that puts it",
                (
                    f"The table is ranked by net strength: {name} is **{ordinal(strength.rank)}** of {len(rows)} clubs "
                    "this season. With the clubs just above and below:",
                ),
                _neighbours_table(rows, strength.rank - 1, league_code),
            )
        )
    return RatingWorkings(f"How {name}'s net strength is calculated", tuple(steps))


def render_rating_workings(
    team_id: int, kind: str, leaderboard: pd.DataFrame, params: DixonColesParams, league: League, close_key: str
) -> None:
    """The workings of `kind` ("attack" | "defense" | "net") for `team_id`, a team on
    `leaderboard` (the current-season leaderboard of `league`, whose model is `params`),
    under the table, with a Close button that drops the `rating` parameter again."""
    strength = team_strength(leaderboard, team_id)
    if strength is None:
        return
    team_name = next(row["Team"] for row in leaderboard.to_dict("records") if row["team_id"] == team_id)

    if kind == "net":
        workings = net_workings(team_name, strength, leaderboard, league.code)
    else:
        names = dict(zip(leaderboard["team_id"], leaderboard["Team"]))
        with session_scope() as session:
            matches = matches_for_league(session, league.code)
        record = season_record(compute_standings(matches, names, str(league.seasons[-1])), team_id)
        if kind == "defense":
            workings = defense_workings(team_name, strength, params, record)
        else:
            detail = resolve_current_attack_detail(team_name, league.code)
            covered = league.code in UNDERSTAT_LEAGUE_SLUG or league.code == "MLS"
            squad = expected_squad(team_name, league) if detail is not None else []
            workings = attack_workings(team_name, strength, params, record, detail, covered, squad)

    st.html(f"<style>{RATING_PANEL_CSS}</style><script>{SCROLL_TO_RATING_JS}</script>", unsafe_allow_javascript=True)
    with st.container(key="rating_breakdown", border=True):
        st.markdown(f"#### {workings.heading}")
        if st.button("Close", key=close_key):
            del st.query_params[RATING_PARAM]
            st.rerun()
        for step in workings.steps:
            st.markdown(f"**{step.heading}**")
            for paragraph in step.paragraphs:
                st.markdown(paragraph)
            if step.table_html:
                st.markdown(step.table_html, unsafe_allow_html=True)
        st.markdown(
            f'<a href="{html.escape(team_page_href(league.code, team_id))}" target="_self">'
            f"Open {html.escape(team_name)}'s page</a> for its squad and how each player counts toward these ratings.",
            unsafe_allow_html=True,
        )
