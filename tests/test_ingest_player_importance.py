from __future__ import annotations

from soccer_predictor.ingest.player_importance import (
    DEFAULT_API_IMPORTANCE_WEIGHT,
    TeamOutputTotals,
    aggregate_team_output,
    aggregate_understat_team_output,
    resolve_api_importance_weight,
    resolve_team_importance_weights,
)
from soccer_predictor.ingest.player_stats import HistoricalPlayerStats
from soccer_predictor.ingest.squad import SquadPlayer
from soccer_predictor.ingest.understat_client import UnderstatPlayerStats


def _player(name, goals=0, assists=0, minutes=0, rating=None, position="Attacker"):
    return HistoricalPlayerStats(
        name=name,
        position=position,
        nationality=None,
        appearances=None,
        minutes=minutes,
        goals=goals,
        assists=assists,
        saves=None,
        tackles=None,
        yellow_cards=None,
        red_cards=None,
        rating=rating,
    )


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


def _squad_player(name, position="Offence"):
    return SquadPlayer(name=name, position=position, nationality="", date_of_birth=None)


def test_aggregate_team_output_sums_goal_contributions_and_tracks_max_minutes():
    roster = [
        _player("Star Striker", goals=20, assists=5, minutes=2800),
        _player("Backup Forward", goals=2, assists=1, minutes=400),
    ]
    totals = aggregate_team_output(roster)

    assert totals.goal_contribution_total == (20 + 5 * 0.7) + (2 + 1 * 0.7)
    assert totals.max_minutes == 2800


def test_aggregate_team_output_empty_roster():
    totals = aggregate_team_output([])
    assert totals.goal_contribution_total == 0
    assert totals.max_minutes == 0


def test_resolve_weight_uses_real_stats_for_matched_attacker():
    roster = [
        _player("Erling Haaland", goals=25, assists=3, minutes=2900),
        _player("Backup Forward", goals=1, assists=0, minutes=200),
    ]
    totals = aggregate_team_output(roster)

    star_weight = resolve_api_importance_weight("Erling Haaland", "attack", roster, totals)
    backup_weight = resolve_api_importance_weight("Backup Forward", "attack", roster, totals)

    assert star_weight > backup_weight


def test_resolve_weight_matches_abbreviated_name():
    roster = [_player("Mohamed Salah", goals=18, assists=10, minutes=2700)]
    totals = aggregate_team_output(roster)

    weight = resolve_api_importance_weight("M. Salah", "attack", roster, totals)
    assert weight != DEFAULT_API_IMPORTANCE_WEIGHT


def test_resolve_weight_falls_back_to_default_when_player_not_found():
    roster = [_player("Erling Haaland", goals=25, assists=3, minutes=2900)]
    totals = aggregate_team_output(roster)

    weight = resolve_api_importance_weight("Some New Signing", "attack", roster, totals)
    assert weight == DEFAULT_API_IMPORTANCE_WEIGHT


def test_resolve_weight_falls_back_to_default_on_empty_roster():
    weight = resolve_api_importance_weight("Anyone", "attack", [], TeamOutputTotals(0, 0))
    assert weight == DEFAULT_API_IMPORTANCE_WEIGHT


def test_resolve_weight_uses_defense_formula_for_defenders():
    roster = [
        _player("Undisputed Keeper", minutes=3000, rating=7.2, position="Goalkeeper"),
        _player("Backup Keeper", minutes=90, rating=6.0, position="Goalkeeper"),
    ]
    totals = aggregate_team_output(roster)

    starter_weight = resolve_api_importance_weight("Undisputed Keeper", "defense", roster, totals)
    backup_weight = resolve_api_importance_weight("Backup Keeper", "defense", roster, totals)

    assert starter_weight > backup_weight


def test_resolve_weight_prefers_understat_xg_when_available():
    # API-Football has nothing at all for this player -- Understat alone
    # should be enough to compute a real weight, not fall back to default.
    understat_roster = [
        _understat_player("Bukayo Saka", goals=12, assists=8, xg=10.5, xa=6.2, minutes=2200),
        _understat_player("Backup Winger", goals=1, assists=0, xg=0.8, xa=0.1, minutes=300),
    ]
    understat_totals = aggregate_understat_team_output(understat_roster)

    weight = resolve_api_importance_weight(
        "Bukayo Saka", "attack", [], TeamOutputTotals(0, 0), understat_roster, understat_totals
    )
    assert weight != DEFAULT_API_IMPORTANCE_WEIGHT


def test_resolve_weight_understat_share_ranks_players_like_api_football_does():
    understat_roster = [
        _understat_player("Star Winger", goals=15, assists=10, xg=13.0, xa=8.0, minutes=2800),
        _understat_player("Bench Player", goals=1, assists=0, xg=0.5, xa=0.1, minutes=200),
    ]
    understat_totals = aggregate_understat_team_output(understat_roster)

    star_weight = resolve_api_importance_weight(
        "Star Winger", "attack", [], TeamOutputTotals(0, 0), understat_roster, understat_totals
    )
    bench_weight = resolve_api_importance_weight(
        "Bench Player", "attack", [], TeamOutputTotals(0, 0), understat_roster, understat_totals
    )
    assert star_weight > bench_weight


def test_resolve_weight_falls_back_to_api_football_when_understat_has_no_match():
    roster = [_player("Erling Haaland", goals=25, assists=3, minutes=2900)]
    totals = aggregate_team_output(roster)
    understat_roster = [_understat_player("Someone Else", goals=1, assists=0, xg=0.5, xa=0.1, minutes=300)]
    understat_totals = aggregate_understat_team_output(understat_roster)

    weight = resolve_api_importance_weight(
        "Erling Haaland", "attack", roster, totals, understat_roster, understat_totals
    )
    assert weight != DEFAULT_API_IMPORTANCE_WEIGHT


def test_resolve_weight_ignores_understat_for_defense_positions():
    roster = [
        _player("Undisputed Keeper", minutes=3000, rating=7.2, position="Goalkeeper"),
        _player("Backup Keeper", minutes=90, rating=6.0, position="Goalkeeper"),
    ]
    totals = aggregate_team_output(roster)
    # A huge Understat attacking stat line for this "keeper" should be
    # ignored entirely -- defense stays minutes-based only.
    understat_roster = [_understat_player("Undisputed Keeper", goals=20, assists=10, xg=18.0, xa=9.0, minutes=3000)]
    understat_totals = aggregate_understat_team_output(understat_roster)

    weight = resolve_api_importance_weight(
        "Undisputed Keeper", "defense", roster, totals, understat_roster, understat_totals
    )
    expected = resolve_api_importance_weight("Undisputed Keeper", "defense", roster, totals)
    assert weight == expected


def test_resolve_team_importance_weights_excludes_players_with_no_stats_match():
    squad = [
        _squad_player("Erling Haaland", position="Offence"),
        _squad_player("Some New Signing", position="Offence"),
    ]
    roster = [_player("Erling Haaland", goals=25, assists=3, minutes=2900)]
    totals = aggregate_team_output(roster)

    weights = resolve_team_importance_weights(squad, roster, totals)

    assert "Erling Haaland" in weights
    assert "Some New Signing" not in weights
    assert weights["Erling Haaland"] != DEFAULT_API_IMPORTANCE_WEIGHT


def test_resolve_team_importance_weights_empty_squad():
    assert resolve_team_importance_weights([], [], TeamOutputTotals(0, 0)) == {}


def test_resolve_team_importance_weights_uses_understat_and_position_bucketing():
    squad = [
        _squad_player("Bukayo Saka", position="Offence"),
        _squad_player("Undisputed Keeper", position="Goalkeeper"),
    ]
    roster = [_player("Undisputed Keeper", minutes=3000, rating=7.2, position="Goalkeeper")]
    totals = aggregate_team_output(roster)
    understat_roster = [_understat_player("Bukayo Saka", goals=12, assists=8, xg=10.5, xa=6.2, minutes=2200)]
    understat_totals = aggregate_understat_team_output(understat_roster)

    weights = resolve_team_importance_weights(squad, roster, totals, understat_roster, understat_totals)

    assert set(weights) == {"Bukayo Saka", "Undisputed Keeper"}
