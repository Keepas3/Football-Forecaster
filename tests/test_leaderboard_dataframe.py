from __future__ import annotations

from soccer_predictor.dashboard.components import leaderboard_dataframe
from soccer_predictor.model.dixon_coles import DixonColesParams


def _params(team_ids: dict[str, int], attack: dict[int, float], defense: dict[int, float]) -> DixonColesParams:
    return DixonColesParams(
        league_code="EPL",
        attack=attack,
        defense=defense,
        home_advantage=1.3,
        rho=-0.1,
        xi=0.0018,
        fitted_at="2026-01-01T00:00:00",
        n_matches=100,
    )


def test_leaderboard_shows_current_form_adjusted_attack(monkeypatch):
    import soccer_predictor.dashboard.components as components

    monkeypatch.setattr(components, "resolve_current_attack_strength", lambda name, code: 1.5 if name == "Hot Team" else None)

    params = _params({}, attack={1: 1.2, 2: 1.0}, defense={1: 0.9, 2: 1.0})
    team_names = {1: "Hot Team", 2: "Average Team"}

    df = leaderboard_dataframe(params, team_names)

    hot_row = df[df["Team"] == "Hot Team"].iloc[0]
    assert hot_row["Attack (from results only)"] == 1.2
    assert hot_row["Attack"] > 1.2  # boosted by current form

    avg_row = df[df["Team"] == "Average Team"].iloc[0]
    assert avg_row["Attack"] == avg_row["Attack (from results only)"] == 1.0  # no signal -> unchanged


def test_leaderboard_sorts_by_adjusted_net_strength(monkeypatch):
    import soccer_predictor.dashboard.components as components

    # Team B has a lower base attack than Team A, but a strong enough
    # current-form boost (capped at +15%) to overtake it in net strength.
    monkeypatch.setattr(
        components,
        "resolve_current_attack_strength",
        lambda name, code: 2.0 if name == "Team B" else None,
    )

    params = _params({}, attack={1: 1.3, 2: 1.2}, defense={1: 1.0, 2: 1.0})
    team_names = {1: "Team A", 2: "Team B"}

    df = leaderboard_dataframe(params, team_names)

    assert df.iloc[0]["Team"] == "Team B"


def test_leaderboard_includes_team_id_for_row_selection(monkeypatch):
    import soccer_predictor.dashboard.components as components

    monkeypatch.setattr(components, "resolve_current_attack_strength", lambda *a, **k: None)

    df = leaderboard_dataframe(
        _params({}, attack={7: 1.0}, defense={7: 1.0}), team_names={7: "Solo Team"}
    )
    assert df.iloc[0]["team_id"] == 7
