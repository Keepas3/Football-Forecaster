from __future__ import annotations

import soccer_predictor.ingest.player_importance as player_importance
from soccer_predictor.ingest.player_importance import (
    DEFAULT_API_IMPORTANCE_WEIGHT,
    AsaTeamTotals,
    UnderstatTeamTotals,
    aggregate_asa_team_output,
    aggregate_understat_team_output,
    resolve_api_importance_weight,
    resolve_team_goals_and_assists,
    resolve_team_historical_stats,
    resolve_team_importance_weights,
)
from soccer_predictor.ingest.asa_client import AsaPlayerStats
from soccer_predictor.ingest.squad import SquadPlayer
from soccer_predictor.ingest.understat_client import UnderstatPlayerStats


def _understat_player(name, team="Arsenal", goals=0, assists=0, xg=0.0, xa=0.0, minutes=0, position="F"):
    return UnderstatPlayerStats(
        name=name,
        team_title=team,
        position=position,
        minutes=minutes,
        goals=goals,
        assists=assists,
        xg=xg,
        xa=xa,
    )


def _asa_player(name, goals=0, assists=0, xg=0.0, xa=0.0, minutes=0, position="ST", points_added=0.0):
    return AsaPlayerStats(
        name=name,
        position=position,
        minutes=minutes,
        goals=goals,
        assists=assists,
        xg=xg,
        xa=xa,
        shots=0,
        key_passes=0,
        points_added=points_added,
    )


def _squad_player(name, position="Offence"):
    return SquadPlayer(name=name, position=position, nationality="", date_of_birth=None)


def test_aggregate_understat_team_output_sums_xg_contribution_and_tracks_max_minutes():
    roster = [
        _understat_player("Star Winger", goals=20, assists=5, xg=18.0, xa=4.0, minutes=2800),
        _understat_player("Backup Forward", goals=2, assists=1, xg=1.5, xa=0.5, minutes=400),
    ]
    totals = aggregate_understat_team_output(roster)

    assert totals.xg_contribution_total == (18.0 + 4.0 * 0.7) + (1.5 + 0.5 * 0.7)
    assert totals.max_minutes == 2800


def test_aggregate_understat_team_output_empty_roster():
    totals = aggregate_understat_team_output([])
    assert totals.xg_contribution_total == 0
    assert totals.max_minutes == 0


def test_aggregate_asa_team_output_sums_xg_contribution_and_tracks_max_minutes():
    roster = [
        _asa_player("Star Striker", xg=12.0, xa=3.0, minutes=2000),
        _asa_player("Backup", xg=0.5, xa=0.1, minutes=300),
    ]
    totals = aggregate_asa_team_output(roster)

    assert totals.xg_contribution_total == (12.0 + 3.0 * 0.7) + (0.5 + 0.1 * 0.7)
    assert totals.max_minutes == 2000


def test_resolve_weight_uses_understat_for_matched_attacker():
    roster = [
        _understat_player("Erling Haaland", goals=25, assists=3, xg=22.0, xa=2.5, minutes=2900),
        _understat_player("Backup Forward", goals=1, assists=0, xg=0.8, xa=0.1, minutes=200),
    ]
    totals = aggregate_understat_team_output(roster)

    star_weight = resolve_api_importance_weight("Erling Haaland", "attack", roster, totals)
    backup_weight = resolve_api_importance_weight("Backup Forward", "attack", roster, totals)

    assert star_weight > backup_weight


def test_resolve_weight_matches_abbreviated_name():
    roster = [_understat_player("Mohamed Salah", goals=18, assists=10, xg=16.0, xa=8.0, minutes=2700)]
    totals = aggregate_understat_team_output(roster)

    weight = resolve_api_importance_weight("M. Salah", "attack", roster, totals)
    assert weight != DEFAULT_API_IMPORTANCE_WEIGHT


def test_resolve_weight_falls_back_to_default_when_player_not_found():
    roster = [_understat_player("Erling Haaland", goals=25, assists=3, xg=22.0, minutes=2900)]
    totals = aggregate_understat_team_output(roster)

    weight = resolve_api_importance_weight("Some New Signing", "attack", roster, totals)
    assert weight == DEFAULT_API_IMPORTANCE_WEIGHT


def test_resolve_weight_falls_back_to_default_with_no_sources_at_all():
    weight = resolve_api_importance_weight("Anyone", "attack")
    assert weight == DEFAULT_API_IMPORTANCE_WEIGHT


def test_resolve_weight_uses_defense_formula_from_understat_minutes():
    roster = [
        _understat_player("Undisputed Keeper", minutes=3000, position="GK"),
        _understat_player("Backup Keeper", minutes=90, position="GK"),
    ]
    totals = aggregate_understat_team_output(roster)

    starter_weight = resolve_api_importance_weight("Undisputed Keeper", "defense", roster, totals)
    backup_weight = resolve_api_importance_weight("Backup Keeper", "defense", roster, totals)

    assert starter_weight > backup_weight


def test_resolve_weight_uses_asa_for_mls_attacker():
    roster = [
        _asa_player("Star Striker", goals=15, assists=5, xg=12.0, xa=4.0, minutes=2000),
        _asa_player("Backup", goals=1, assists=0, xg=0.5, xa=0.1, minutes=300),
    ]
    totals = aggregate_asa_team_output(roster)

    star_weight = resolve_api_importance_weight("Star Striker", "attack", asa_players=roster, asa_totals=totals)
    backup_weight = resolve_api_importance_weight("Backup", "attack", asa_players=roster, asa_totals=totals)

    assert star_weight > backup_weight


def test_resolve_weight_uses_asa_defense_formula():
    roster = [
        _asa_player("Undisputed Keeper", minutes=3000, position="GK"),
        _asa_player("Backup Keeper", minutes=90, position="GK"),
    ]
    totals = aggregate_asa_team_output(roster)

    starter_weight = resolve_api_importance_weight(
        "Undisputed Keeper", "defense", asa_players=roster, asa_totals=totals
    )
    backup_weight = resolve_api_importance_weight(
        "Backup Keeper", "defense", asa_players=roster, asa_totals=totals
    )

    assert starter_weight > backup_weight


def test_resolve_weight_understat_and_asa_never_both_needed_at_once():
    # A player only in the ASA roster, queried with an Understat roster that
    # doesn't have them -- should fall back to ASA, not silently miss it.
    understat_roster = [_understat_player("Someone Else", goals=1, assists=0, xg=0.5, minutes=300)]
    understat_totals = aggregate_understat_team_output(understat_roster)
    asa_roster = [_asa_player("MLS Star", goals=10, assists=5, xg=8.0, xa=3.0, minutes=1800)]
    asa_totals = aggregate_asa_team_output(asa_roster)

    weight = resolve_api_importance_weight(
        "MLS Star",
        "attack",
        understat_players=understat_roster,
        understat_totals=understat_totals,
        asa_players=asa_roster,
        asa_totals=asa_totals,
    )
    assert weight != DEFAULT_API_IMPORTANCE_WEIGHT


def test_resolve_team_importance_weights_excludes_players_with_no_stats_match():
    squad = [
        _squad_player("Erling Haaland", position="Offence"),
        _squad_player("Some New Signing", position="Offence"),
    ]
    roster = [_understat_player("Erling Haaland", goals=25, assists=3, xg=22.0, minutes=2900)]
    totals = aggregate_understat_team_output(roster)

    weights = resolve_team_importance_weights(squad, understat_players=roster, understat_totals=totals)

    assert "Erling Haaland" in weights
    assert "Some New Signing" not in weights
    assert weights["Erling Haaland"] != DEFAULT_API_IMPORTANCE_WEIGHT


def test_resolve_team_importance_weights_empty_squad():
    assert resolve_team_importance_weights([]) == {}


def test_resolve_team_importance_weights_position_bucketing():
    squad = [
        _squad_player("Bukayo Saka", position="Offence"),
        _squad_player("Undisputed Keeper", position="Goalkeeper"),
    ]
    roster = [
        _understat_player("Bukayo Saka", goals=12, assists=8, xg=10.5, xa=6.2, minutes=2200),
        _understat_player("Undisputed Keeper", minutes=3000, position="GK"),
    ]
    totals = aggregate_understat_team_output(roster)

    weights = resolve_team_importance_weights(squad, understat_players=roster, understat_totals=totals)

    assert set(weights) == {"Bukayo Saka", "Undisputed Keeper"}


def test_resolve_team_historical_stats_understat_for_covered_league(monkeypatch):
    understat_roster = [_understat_player("Bukayo Saka", goals=12, assists=8, xg=10.5, xa=6.2, minutes=2200)]
    monkeypatch.setattr(player_importance, "fetch_team_season", lambda *a, **k: understat_roster)

    source, players = resolve_team_historical_stats("Arsenal", "EPL", 2026)

    assert source == "understat"
    assert players == understat_roster


def test_resolve_team_historical_stats_understat_covered_league_can_be_empty(monkeypatch):
    monkeypatch.setattr(player_importance, "fetch_team_season", lambda *a, **k: [])

    source, players = resolve_team_historical_stats("Arsenal", "EPL", 2019)

    assert source == "understat"
    assert players == []


def test_resolve_team_historical_stats_uncovered_league_returns_none_source(monkeypatch):
    def fail_if_called(*a, **k):
        raise AssertionError("Understat should not be called for a league it doesn't cover")

    monkeypatch.setattr(player_importance, "fetch_team_season", fail_if_called)

    source, players = resolve_team_historical_stats("Real Madrid", "UCL", 2024)

    assert source == "none"
    assert players == []


def test_resolve_team_historical_stats_mls_uses_asa(monkeypatch):
    asa_roster = [object()]
    monkeypatch.setattr(player_importance, "fetch_asa_team_season", lambda *a, **k: asa_roster)

    source, players = resolve_team_historical_stats("Atlanta United FC", "MLS", 2026)

    assert source == "asa"
    assert players == asa_roster


def test_resolve_team_historical_stats_mls_can_be_empty(monkeypatch):
    monkeypatch.setattr(player_importance, "fetch_asa_team_season", lambda *a, **k: [])

    source, players = resolve_team_historical_stats("Atlanta United FC", "MLS", 2026)

    assert source == "asa"
    assert players == []


def test_resolve_team_goals_and_assists_keys_by_squad_name_not_stats_name():
    # Real-world case this must handle: Understat spells this player
    # "Alexey Miranchuk", this app's ESPN-sourced squad spells him
    # "Aleksey Miranchuk" -- confirmed live earlier this session.
    squad = [_squad_player("Aleksey Miranchuk", position="Midfield")]
    understat_roster = [_understat_player("Alexey Miranchuk", goals=6, assists=6, minutes=2000)]

    result = resolve_team_goals_and_assists(squad, understat_players=understat_roster)

    assert result == {"Aleksey Miranchuk": (6, 6)}


def test_resolve_team_goals_and_assists_uses_asa_when_no_understat_given():
    squad = [_squad_player("Star Striker", position="Offence")]
    asa_roster = [_asa_player("Star Striker", goals=15, assists=3, minutes=2000)]

    result = resolve_team_goals_and_assists(squad, asa_players=asa_roster)

    assert result == {"Star Striker": (15, 3)}


def test_resolve_team_goals_and_assists_excludes_unmatched_players():
    squad = [
        _squad_player("Star Striker", position="Offence"),
        _squad_player("New Signing", position="Offence"),
    ]
    understat_roster = [_understat_player("Star Striker", goals=15, assists=3, minutes=2000)]

    result = resolve_team_goals_and_assists(squad, understat_players=understat_roster)

    assert "Star Striker" in result
    assert "New Signing" not in result


def test_resolve_team_goals_and_assists_empty_squad():
    assert resolve_team_goals_and_assists([]) == {}


def test_resolve_team_goals_and_assists_no_stats_source_gives_empty_dict():
    squad = [_squad_player("Star Striker", position="Offence")]

    assert resolve_team_goals_and_assists(squad) == {}
