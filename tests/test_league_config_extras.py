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


def test_data_source_defaults_to_football_data_org():
    league = League(code="UCL", name="UEFA Champions League", api_competition_id=2001, seasons=["2425"])
    assert league.data_source == "football_data_org"
    assert league.espn_league_slug is None


def test_load_leagues_parses_espn_data_source_with_no_api_competition_id(tmp_path, monkeypatch):
    import soccer_predictor.config as config

    (tmp_path / "leagues.yaml").write_text(
        "leagues:\n"
        "  - code: MLS\n"
        "    name: Major League Soccer\n"
        "    data_source: espn\n"
        "    espn_league_slug: usa.1\n"
        '    seasons: ["2026"]\n'
        "    season_display: single_year\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    config.load_leagues.cache_clear()

    try:
        league = config.load_leagues()["MLS"]
        assert league.api_competition_id is None
        assert league.data_source == "espn"
        assert league.espn_league_slug == "usa.1"
    finally:
        config.load_leagues.cache_clear()


def test_trainable_defaults_to_false():
    league = League(code="UCL", name="UEFA Champions League", api_competition_id=2001, seasons=["2425"])
    assert league.trainable is False
    assert league.supports_predictions is False


def test_trainable_true_gives_supports_predictions_without_csv_code():
    league = League(
        code="MLS",
        name="Major League Soccer",
        seasons=["2026"],
        data_source="espn",
        espn_league_slug="usa.1",
        trainable=True,
    )
    assert league.csv_code is None
    assert league.supports_predictions is True


def test_real_leagues_yaml_wc_is_trainable_archive_with_historical_seasons():
    import soccer_predictor.config as config

    league = config.load_leagues()["WC"]
    assert league.trainable is True
    assert league.data_source == "archive_worldcup"
    assert "1966" in league.seasons
    assert league.supports_predictions is True


def test_real_leagues_yaml_euro_is_trainable_archive_with_historical_seasons():
    import soccer_predictor.config as config

    league = config.load_leagues()["EURO"]
    assert league.trainable is True
    assert league.data_source == "archive_euro"
    assert "1996" in league.seasons
    assert league.supports_predictions is True


def test_load_leagues_parses_trainable_flag(tmp_path, monkeypatch):
    import soccer_predictor.config as config

    (tmp_path / "leagues.yaml").write_text(
        "leagues:\n"
        "  - code: MLS\n"
        "    name: Major League Soccer\n"
        "    data_source: espn\n"
        "    espn_league_slug: usa.1\n"
        "    trainable: true\n"
        '    seasons: ["2026"]\n'
        "    season_display: single_year\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    config.load_leagues.cache_clear()

    try:
        league = config.load_leagues()["MLS"]
        assert league.trainable is True
        assert league.supports_predictions is True
    finally:
        config.load_leagues.cache_clear()


def test_news_query_defaults_to_none():
    league = League(code="UCL", name="UEFA Champions League", api_competition_id=2001, seasons=["2425"])
    assert league.news_query is None


def test_load_leagues_parses_news_query(tmp_path, monkeypatch):
    import soccer_predictor.config as config

    (tmp_path / "leagues.yaml").write_text(
        "leagues:\n"
        "  - code: EPL\n"
        "    name: English Premier League\n"
        "    news_query: Premier League\n"
        "    csv_code: E0\n"
        "    api_competition_id: 2021\n"
        '    seasons: ["2425"]\n'
        "  - code: LALIGA\n"
        "    name: La Liga\n"
        "    csv_code: SP1\n"
        "    api_competition_id: 2014\n"
        '    seasons: ["2425"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    config.load_leagues.cache_clear()

    try:
        leagues = config.load_leagues()
        assert leagues["EPL"].news_query == "Premier League"
        assert leagues["LALIGA"].news_query is None
    finally:
        config.load_leagues.cache_clear()
