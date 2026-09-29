from __future__ import annotations

import soccer_predictor.dashboard.components as components
from soccer_predictor.config import ManualCaptain, ManualStarPlayer
from soccer_predictor.dashboard.components import (
    compute_top_scorer_and_assister,
    render_player_badges,
    render_star_players,
)
from soccer_predictor.ingest.understat_client import UnderstatPlayerStats


def test_compute_top_scorer_and_assister_picks_the_max_of_each():
    goals_and_assists = {"Top Scorer": (15, 2), "Top Assister": (3, 10), "Bench Player": (0, 0)}
    scorer, assister = compute_top_scorer_and_assister(goals_and_assists)
    assert scorer == "Top Scorer"
    assert assister == "Top Assister"


def test_compute_top_scorer_and_assister_same_player_can_be_both():
    goals_and_assists = {"All-Rounder": (10, 8), "Bench Player": (1, 1)}
    scorer, assister = compute_top_scorer_and_assister(goals_and_assists)
    assert scorer == "All-Rounder"
    assert assister == "All-Rounder"


def test_compute_top_scorer_and_assister_no_goals_or_assists_gives_none():
    goals_and_assists = {"Player A": (0, 0), "Player B": (0, 0)}
    scorer, assister = compute_top_scorer_and_assister(goals_and_assists)
    assert scorer is None
    assert assister is None


def test_compute_top_scorer_and_assister_empty_input():
    assert compute_top_scorer_and_assister({}) == (None, None)


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


def test_render_player_badges_top_scorer_gets_star(monkeypatch):
    monkeypatch.setattr(components, "load_manual_captains", lambda: [])
    monkeypatch.setattr(components, "load_manual_star_players", lambda: [])

    assert render_player_badges("Haaland", "Man City", top_scorer="Haaland") == "⭐ Haaland"
    assert render_player_badges("Backup", "Man City", top_scorer="Haaland") == "Backup"


def test_render_player_badges_top_assister_gets_target_symbol(monkeypatch):
    monkeypatch.setattr(components, "load_manual_captains", lambda: [])
    monkeypatch.setattr(components, "load_manual_star_players", lambda: [])

    assert render_player_badges("De Bruyne", "Man City", top_assister="De Bruyne") == "🎯 De Bruyne"
    assert render_player_badges("Backup", "Man City", top_assister="De Bruyne") == "Backup"


def test_render_player_badges_top_scorer_and_top_assister_can_be_different_players(monkeypatch):
    monkeypatch.setattr(components, "load_manual_captains", lambda: [])
    monkeypatch.setattr(components, "load_manual_star_players", lambda: [])

    result = render_player_badges("Haaland", "Man City", top_scorer="Haaland", top_assister="De Bruyne")
    assert result == "⭐ Haaland"


def test_render_player_badges_same_player_top_scorer_and_assister_gets_both_symbols(monkeypatch):
    monkeypatch.setattr(components, "load_manual_captains", lambda: [])
    monkeypatch.setattr(components, "load_manual_star_players", lambda: [])

    result = render_player_badges("All-Rounder", "Man City", top_scorer="All-Rounder", top_assister="All-Rounder")
    assert result == "⭐ 🎯 All-Rounder"


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
