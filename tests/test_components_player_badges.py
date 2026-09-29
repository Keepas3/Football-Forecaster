from __future__ import annotations

import soccer_predictor.dashboard.components as components
from soccer_predictor.config import ManualCaptain, ManualStarPlayer
from soccer_predictor.dashboard.components import compute_automatic_stars, render_player_badges, render_star_players
from soccer_predictor.ingest.understat_client import UnderstatPlayerStats


def test_compute_automatic_stars_picks_top_n_above_threshold():
    weights = {"Star Striker": 0.9, "Solid Starter": 0.6, "Bench Player": 0.2}
    assert compute_automatic_stars(weights, top_n=2, min_weight=0.5) == {"Star Striker", "Solid Starter"}


def test_compute_automatic_stars_no_standout_gives_empty_set():
    weights = {"Player A": 0.3, "Player B": 0.25, "Player C": 0.1}
    assert compute_automatic_stars(weights, top_n=2, min_weight=0.5) == set()


def test_compute_automatic_stars_respects_top_n_even_if_more_clear_threshold():
    weights = {"A": 0.9, "B": 0.8, "C": 0.7}
    assert compute_automatic_stars(weights, top_n=2, min_weight=0.5) == {"A", "B"}


def test_render_player_badges_captain_only(monkeypatch):
    monkeypatch.setattr(components, "load_manual_captains", lambda: [ManualCaptain(team="Arsenal", player="Saka")])
    monkeypatch.setattr(components, "load_manual_star_players", lambda: [])

    assert render_player_badges("Saka", "Arsenal") == "👑 Saka"


def test_render_player_badges_star_only(monkeypatch):
    monkeypatch.setattr(components, "load_manual_captains", lambda: [])
    monkeypatch.setattr(
        components, "load_manual_star_players", lambda: [ManualStarPlayer(team="Arsenal", player="Saka")]
    )

    assert render_player_badges("Saka", "Arsenal") == "⭐ Saka"


def test_render_player_badges_captain_and_star(monkeypatch):
    monkeypatch.setattr(components, "load_manual_captains", lambda: [ManualCaptain(team="Arsenal", player="Saka")])
    monkeypatch.setattr(
        components, "load_manual_star_players", lambda: [ManualStarPlayer(team="Arsenal", player="Saka")]
    )

    assert render_player_badges("Saka", "Arsenal") == "👑 ⭐ Saka"


def test_render_player_badges_neither(monkeypatch):
    monkeypatch.setattr(components, "load_manual_captains", lambda: [])
    monkeypatch.setattr(components, "load_manual_star_players", lambda: [])

    assert render_player_badges("Saka", "Arsenal") == "Saka"


def test_render_player_badges_matches_case_insensitively_on_player_name(monkeypatch):
    monkeypatch.setattr(components, "load_manual_captains", lambda: [ManualCaptain(team="Arsenal", player="saka")])
    monkeypatch.setattr(components, "load_manual_star_players", lambda: [])

    assert render_player_badges("Saka", "Arsenal") == "👑 Saka"


def test_render_player_badges_requires_exact_team_match(monkeypatch):
    monkeypatch.setattr(
        components, "load_manual_captains", lambda: [ManualCaptain(team="Chelsea", player="Saka")]
    )
    monkeypatch.setattr(components, "load_manual_star_players", lambda: [])

    assert render_player_badges("Saka", "Arsenal") == "Saka"


def test_render_player_badges_automatic_star_set(monkeypatch):
    monkeypatch.setattr(components, "load_manual_captains", lambda: [])
    monkeypatch.setattr(components, "load_manual_star_players", lambda: [])

    assert render_player_badges("Haaland", "Man City", automatic_stars={"Haaland"}) == "⭐ Haaland"
    assert render_player_badges("Backup", "Man City", automatic_stars={"Haaland"}) == "Backup"


def test_render_star_players_does_not_raise_for_understat_players_with_no_rating():
    # UnderstatPlayerStats has no .rating attribute at all -- render_star_players
    # used to do `if p.rating:`, which raised AttributeError for this dataclass.
    understat_roster = [
        UnderstatPlayerStats(
            name="Bukayo Saka",
            team_title="Arsenal",
            position="F",
            minutes=2200,
            goals=12,
            assists=8,
            xg=10.5,
            xa=6.2,
        )
    ]

    render_star_players(understat_roster)  # must not raise
