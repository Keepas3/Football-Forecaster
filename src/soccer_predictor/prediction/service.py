"""The one module the dashboard calls into for predictions.

Reads the latest persisted Dixon-Coles fit (see prediction/training.py),
applies the (cheap) injury and form adjustments, and derives markets -- so
injury/note updates never require re-running the slower league-wide fit.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from soccer_predictor.config import load_manual_injuries
from soccer_predictor.model.dixon_coles import DixonColesParams
from soccer_predictor.model.form_adjustment import FormNoteEntry, adjust_for_form
from soccer_predictor.model.injury_adjustment import InjuryEntry, adjust_strength
from soccer_predictor.model.markets import MatchPrediction, PredictionBreakdown, predict_markets
from soccer_predictor.model.scoreline_matrix import build_matrix
from soccer_predictor.storage.models import FittedParams
from soccer_predictor.storage.repository import (
    form_notes_for_team,
    injuries_for_team,
)


@dataclass
class TeamInjuries:
    entries: list[InjuryEntry]
    sources: list[str]  # parallel list: "manual" | "api" | "chat" per entry, for dashboard display
    expected_return_dates: list[dt.date | None]  # parallel list, for display


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

    return TeamInjuries(
        entries=entries, sources=sources, expected_return_dates=expected_return_dates
    )


def get_form_notes_for_team(session: Session, team_id: int, as_of: dt.date) -> list[FormNoteEntry]:
    active = [row for row in form_notes_for_team(session, team_id) if row.expires_on >= as_of]
    return [FormNoteEntry(magnitude=row.magnitude, affects=row.affects) for row in active]


def predict_fixture(
    session: Session,
    params: DixonColesParams,
    home_team_id: int,
    away_team_id: int,
    home_team_name: str,
    away_team_name: str,
    fixture_date: dt.date | None = None,
) -> MatchPrediction:
    if fixture_date is None:
        fixture_date = dt.date.today()

    home_injuries = get_injuries_for_team(session, home_team_id, home_team_name, fixture_date).entries
    away_injuries = get_injuries_for_team(session, away_team_id, away_team_name, fixture_date).entries
    home_form = get_form_notes_for_team(session, home_team_id, fixture_date)
    away_form = get_form_notes_for_team(session, away_team_id, fixture_date)

    base_home_attack, base_home_defense = params.attack[home_team_id], params.defense[home_team_id]
    base_away_attack, base_away_defense = params.attack[away_team_id], params.defense[away_team_id]

    home_attack, home_defense = adjust_strength(base_home_attack, base_home_defense, home_injuries)
    home_attack, home_defense = adjust_for_form(home_attack, home_defense, home_form)

    away_attack, away_defense = adjust_strength(base_away_attack, base_away_defense, away_injuries)
    away_attack, away_defense = adjust_for_form(away_attack, away_defense, away_form)

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
    )
    return prediction
