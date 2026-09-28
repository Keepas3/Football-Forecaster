from __future__ import annotations

from soccer_predictor.config import League


def test_supports_predictions_false_without_csv_code():
    league = League(code="UCL", name="UEFA Champions League", api_competition_id=2001, seasons=["2425"])
    assert league.supports_predictions is False


def test_supports_predictions_true_with_csv_code():
    league = League(
        code="EPL", name="English Premier League", api_competition_id=2021, seasons=["2425"], csv_code="E0"
    )
    assert league.supports_predictions is True


def test_api_season_year_range_format():
    league = League(code="UCL", name="UEFA Champions League", api_competition_id=2001, seasons=["2425"])
    assert league.api_season_year("2425") == 2024


def test_api_season_year_single_year_format():
    league = League(
        code="EURO",
        name="European Championship",
        api_competition_id=2018,
        seasons=["2024"],
        season_display="single_year",
    )
    assert league.api_season_year("2024") == 2024


def test_load_leagues_parses_optional_csv_code_and_season_display(tmp_path, monkeypatch):
    import soccer_predictor.config as config

    (tmp_path / "leagues.yaml").write_text(
        "leagues:\n"
        "  - code: EURO\n"
        "    name: European Championship\n"
        "    api_competition_id: 2018\n"
        '    seasons: ["2024"]\n'
        "    season_display: single_year\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    config.load_leagues.cache_clear()

    try:
        league = config.load_leagues()["EURO"]
        assert league.csv_code is None
        assert league.supports_predictions is False
        assert league.season_display == "single_year"
    finally:
        config.load_leagues.cache_clear()
