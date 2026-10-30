"""The one module the dashboard calls into for predictions.

Reads the latest persisted Dixon-Coles fit (see prediction/training.py),
applies the (cheap) injury and form adjustments, and derives markets -- so
injury/note updates never require re-running the slower league-wide fit.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from soccer_predictor.config import League, load_leagues, load_manual_injuries
from soccer_predictor.ingest.lineups import expected_squad
from soccer_predictor.ingest.player_importance import resolve_current_attack_strength
from soccer_predictor.model.current_form_adjustment import adjust_for_current_attack_form
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.model.form_adjustment import FormNoteEntry, adjust_for_form
from soccer_predictor.model.injury_adjustment import InjuryEntry, adjust_strength
from soccer_predictor.model.lineup_adjustment import LineupAbsence, adjust_for_lineup, lineup_absences
from soccer_predictor.model.markets import MatchPrediction, PredictionBreakdown, predict_markets
from soccer_predictor.model.results_form_adjustment import (
    RECENT_MATCHES,
    ResultsForm,
    adjust_for_results_form,
    results_form,
)
from soccer_predictor.model.scoreline_matrix import build_matrix
from soccer_predictor.storage.models import FittedParams
from soccer_predictor.storage.repository import (
    form_notes_for_team,
    injuries_for_team,
    recent_results_for_team,
)


@dataclass
class TeamInjuries:
    entries: list[InjuryEntry]
    sources: list[str]  # parallel list: "manual" | "api" | "chat" per entry, for dashboard display
    expected_return_dates: list[dt.date | None]  # parallel list, for display
    notes: list[str] = field(default_factory=list)  # parallel list: e.g. "FPL: Ankle injury - Expected back 24 Oct"


def load_latest_params(session: Session, league_code: str) -> DixonColesParams | None:
    row = session.scalar(
        select(FittedParams)
        .where(FittedParams.league_code == league_code)
        .order_by(FittedParams.fitted_at.desc())
        .limit(1)
    )
    return DixonColesParams.from_json(row.params_json) if row else None


def get_injuries_for_team(
    session: Session, team_id: int, team_name: str, as_of: dt.date
) -> TeamInjuries:
    """Merges manual config/injuries.yaml with API-fetched and chat-sourced rows.

    Manual entries win: if the same player appears in more than one source,
    the manual entry's weight/position is used (manual data is the most
    trustworthy source per the plan -- free injury APIs have weak coverage).

    Only chat rows ever carry expected_return_date; manual/api rows have it
    as None and so are never filtered out here regardless of `as_of` --
    that's the point, their prior "always active" behavior is unchanged.
    """
    manual = [m for m in load_manual_injuries() if m.team == team_name]
    manual_names = {m.player.lower() for m in manual}

    other_rows = [
        row
        for row in injuries_for_team(session, team_id)
        if row.source in ("api", "chat")
        and row.player_name.lower() not in manual_names
        and (row.expected_return_date is None or row.expected_return_date >= as_of)
    ]

    entries: list[InjuryEntry] = []
    sources: list[str] = []
    expected_return_dates: list[dt.date | None] = []
    notes: list[str] = []
    for m in manual:
        entries.append(
            InjuryEntry(
                player_name=m.player,
                position=m.position,
                importance_weight=m.importance_weight,
            )
        )
        sources.append("manual")
        expected_return_dates.append(None)
        notes.append(m.note or "")
    for row in other_rows:
        entries.append(
            InjuryEntry(
                player_name=row.player_name,
                position=row.position,
                importance_weight=row.importance_weight,
            )
        )
        sources.append(row.source)
        expected_return_dates.append(row.expected_return_date)
        notes.append(row.note or "")

    return TeamInjuries(
        entries=entries, sources=sources, expected_return_dates=expected_return_dates, notes=notes
    )


def get_form_notes_for_team(session: Session, team_id: int, as_of: dt.date) -> list[FormNoteEntry]:
    active = [row for row in form_notes_for_team(session, team_id) if row.expires_on >= as_of]
    return [FormNoteEntry(magnitude=row.magnitude, affects=row.affects) for row in active]


def get_results_form(session: Session, team_id: int, league: League | None, as_of: dt.date) -> ResultsForm | None:
    """The team's last six results as a rating nudge, or None in a league that
    does not use them (see model/results_form_adjustment.py) or before there
    are six recent ones."""
    if league is None or not league.recent_results_form:
        return None
    return results_form(recent_results_for_team(session, team_id, as_of, RECENT_MATCHES), as_of)


def rate_lineup(team_name: str, league_code: str, xi_names: Sequence[str] | None) -> list[LineupAbsence] | None:
    """Who is missing from `team_name`'s usual XI given the XI named for the match,
    or None when there is no named XI or the club's season stats cannot be found
    (then the prediction stays on its injury-based estimate)."""
    if not xi_names:
        return None
    league = load_leagues().get(league_code)
    squad = expected_squad(team_name, league) if league else []
    return lineup_absences(squad, xi_names) if squad else None


def predict_fixture(
    session: Session,
    params: DixonColesParams,
    home_team_id: int,
    away_team_id: int,
    home_team_name: str,
    away_team_name: str,
    fixture_date: dt.date | None = None,
    *,
    extra_home_injuries: Sequence[InjuryEntry] = (),
    extra_away_injuries: Sequence[InjuryEntry] = (),
    extra_home_form: Sequence[FormNoteEntry] = (),
    extra_away_form: Sequence[FormNoteEntry] = (),
    home_lineup: Sequence[str] | None = None,
    away_lineup: Sequence[str] | None = None,
) -> MatchPrediction:
    """The `extra_*` entries are what-if adjustments held in memory only, added
    on top of what the database holds (the AI what-if tab passes a visitor's
    note this way). Nothing here saves them, so they cannot reach another
    visitor's prediction or a locked-in snapshot; left empty, the result is
    exactly what every other caller has always got.

    `home_lineup` / `away_lineup` are the names of the starting XI once it is
    announced. A side whose XI is given and whose club's usual XI is known is
    adjusted for who is missing from it INSTEAD of for its database injuries
    (an absent player is absent whatever the reason, and a doubtful one who
    starts is no longer a doubt); any other side is adjusted for injuries as
    before. The breakdown records which sides were rated this way."""
    if fixture_date is None:
        fixture_date = dt.date.today()

    home_absences = rate_lineup(home_team_name, params.league_code, home_lineup)
    away_absences = rate_lineup(away_team_name, params.league_code, away_lineup)

    home_injuries = [
        *(get_injuries_for_team(session, home_team_id, home_team_name, fixture_date).entries if home_absences is None else []),
        *extra_home_injuries,
    ]
    away_injuries = [
        *(get_injuries_for_team(session, away_team_id, away_team_name, fixture_date).entries if away_absences is None else []),
        *extra_away_injuries,
    ]
    home_form = [*get_form_notes_for_team(session, home_team_id, fixture_date), *extra_home_form]
    away_form = [*get_form_notes_for_team(session, away_team_id, fixture_date), *extra_away_form]
    league = load_leagues().get(params.league_code)
    home_results_form = get_results_form(session, home_team_id, league, fixture_date)
    away_results_form = get_results_form(session, away_team_id, league, fixture_date)

    base_home_attack, base_home_defense = params.attack[home_team_id], params.defense[home_team_id]
    base_away_attack, base_away_defense = params.attack[away_team_id], params.defense[away_team_id]

    home_attack, home_defense = base_home_attack, base_home_defense
    if home_absences is not None:
        home_attack, home_defense = adjust_for_lineup(home_attack, home_defense, home_absences)
    home_attack, home_defense = adjust_strength(home_attack, home_defense, home_injuries)
    home_attack, home_defense = adjust_for_form(home_attack, home_defense, home_form)
    home_attack, home_defense = adjust_for_results_form(home_attack, home_defense, home_results_form)

    away_attack, away_defense = base_away_attack, base_away_defense
    if away_absences is not None:
        away_attack, away_defense = adjust_for_lineup(away_attack, away_defense, away_absences)
    away_attack, away_defense = adjust_strength(away_attack, away_defense, away_injuries)
    away_attack, away_defense = adjust_for_form(away_attack, away_defense, away_form)
    away_attack, away_defense = adjust_for_results_form(away_attack, away_defense, away_results_form)

    home_xg_relative_strength = resolve_current_attack_strength(home_team_name, params.league_code)
    away_xg_relative_strength = resolve_current_attack_strength(away_team_name, params.league_code)
    home_attack = adjust_for_current_attack_form(home_attack, home_xg_relative_strength)
    away_attack = adjust_for_current_attack_form(away_attack, away_xg_relative_strength)

    lambda_home = home_attack * away_defense * params.home_advantage
    lambda_away = away_attack * home_defense

    matrix = build_matrix(lambda_home, lambda_away, params.rho)
    prediction = predict_markets(matrix)
    prediction.breakdown = PredictionBreakdown(
        base_home_attack=base_home_attack,
        base_home_defense=base_home_defense,
        base_away_attack=base_away_attack,
        base_away_defense=base_away_defense,
        home_attack=home_attack,
        home_defense=home_defense,
        away_attack=away_attack,
        away_defense=away_defense,
        home_advantage=params.home_advantage,
        rho=params.rho,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        n_matches=params.n_matches,
        fitted_at=params.fitted_at,
        xi=params.xi,
        home_xg_relative_strength=home_xg_relative_strength,
        away_xg_relative_strength=away_xg_relative_strength,
        home_lineup_absences=home_absences,
        away_lineup_absences=away_absences,
        home_results_form=home_results_form,
        away_results_form=away_results_form,
    )
    return prediction
