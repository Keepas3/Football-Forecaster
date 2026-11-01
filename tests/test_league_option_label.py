"""League picker labels name the next edition of quadrennial tournaments."""

from __future__ import annotations

from soccer_predictor.config import load_leagues
from soccer_predictor.dashboard.components import (
    FLAG_FONT_CSS,
    FLAG_FONT_URL,
    league_option_label,
    next_tournament_year,
)


def test_international_tournaments_show_next_edition_year():
    leagues = load_leagues()
    assert next_tournament_year(leagues["WC"]) == "2030"
    assert next_tournament_year(leagues["EURO"]) == "2028"
    assert next_tournament_year(leagues["ASIA"]) == "2027"
    assert league_option_label(leagues["WC"]).endswith("FIFA World Cup 2030")
    assert league_option_label(leagues["EURO"]).endswith("European Championship 2028")
    assert league_option_label(leagues["ASIA"]).endswith("AFC Asian Cup 2027")


ENGLAND = "\U0001F3F4" + "".join(chr(0xE0000 + ord(c)) for c in "gbeng") + "\U000E007F"


def test_the_premier_league_shows_englands_flag_not_the_letters_en():
    leagues = load_leagues()
    assert leagues["EPL"].flag_emoji == ENGLAND
    assert league_option_label(leagues["EPL"]) == f"{ENGLAND} English Premier League"


def test_every_league_flag_is_made_of_characters_the_flag_font_covers():
    # regional-indicator pairs (countries, the UN) and the black flag + tag letters (England)
    covered = [(0x1F1E6, 0x1F1FF), (0x1F3F4, 0x1F3F4), (0xE0062, 0xE007F)]
    for league in load_leagues().values():
        assert league.flag_emoji, league.code
        for char in league.flag_emoji:
            assert any(low <= ord(char) <= high for low, high in covered), (league.code, hex(ord(char)))


def test_the_flag_font_only_claims_flag_characters_and_reaches_the_league_picker():
    css = FLAG_FONT_CSS
    assert FLAG_FONT_URL in css and "@2" not in css and "@0.1.8" in FLAG_FONT_URL  # pinned, not "latest"
    assert "unicode-range:U+1F1E6-1F1FF,U+1F3F4,U+E0062-E007F" in css
    assert '[data-testid="stSelectbox"] input' in css and '[data-testid="stSelectboxVirtualDropdown"] [role="option"]' in css
    assert css.count("{") == css.count("}")


def test_annual_and_range_leagues_are_unchanged():
    leagues = load_leagues()
    for code in ("EPL", "MLS", "NL", "UCL"):
        assert next_tournament_year(leagues[code]) is None
        assert not league_option_label(leagues[code]).rstrip().split()[-1].isdigit()
