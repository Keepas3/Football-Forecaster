"""League picker labels name the next edition of quadrennial tournaments."""

from __future__ import annotations

from soccer_predictor.config import load_leagues
from soccer_predictor.dashboard.components import league_option_label, next_tournament_year


def test_international_tournaments_show_next_edition_year():
    leagues = load_leagues()
    assert next_tournament_year(leagues["WC"]) == "2030"
    assert next_tournament_year(leagues["EURO"]) == "2028"
    assert next_tournament_year(leagues["ASIA"]) == "2027"
    assert league_option_label(leagues["WC"]).endswith("FIFA World Cup 2030")
    assert league_option_label(leagues["EURO"]).endswith("European Championship 2028")
    assert league_option_label(leagues["ASIA"]).endswith("AFC Asian Cup 2027")


def test_annual_and_range_leagues_are_unchanged():
    leagues = load_leagues()
    for code in ("EPL", "MLS", "NL", "UCL"):
        assert next_tournament_year(leagues[code]) is None
        assert not league_option_label(leagues[code]).rstrip().split()[-1].isdigit()
