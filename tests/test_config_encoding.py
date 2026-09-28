"""Regression test for a real bug: Path.read_text() without an explicit
encoding defaults to the platform's preferred encoding (cp1252 on this
Windows dev machine), not UTF-8 -- silently mangling any accented team name
(e.g. "Málaga", "Köln", "Coruña") in the YAML config files. config.py must
always pass encoding="utf-8" explicitly.
"""

from __future__ import annotations

import soccer_predictor.config as config


def test_load_leagues_decodes_utf8(tmp_path, monkeypatch):
    (tmp_path / "leagues.yaml").write_text(
        "leagues:\n"
        "  - code: TEST\n"
        "    name: Bundesliga Málaga Köln Test\n"
        "    csv_code: X0\n"
        "    api_competition_id: 1\n"
        "    seasons: [\"2425\"]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    config.load_leagues.cache_clear()

    try:
        leagues = config.load_leagues()
        assert leagues["TEST"].name == "Bundesliga Málaga Köln Test"
    finally:
        # Don't leave this test's tmp_path result cached for later tests/code
        # that expect the real config/ directory.
        config.load_leagues.cache_clear()


def test_load_team_aliases_decodes_utf8(tmp_path, monkeypatch):
    (tmp_path / "team_aliases.yaml").write_text(
        "teams:\n"
        "  - canonical_name: Deportivo La Coruna\n"
        "    csv: La Coruna\n"
        "    api: RC Deportivo La Coruña\n"
        "    league: LALIGA\n"
        "  - canonical_name: Malaga\n"
        "    csv: Malaga\n"
        "    api: Málaga CF\n"
        "    league: LALIGA\n"
        "  - canonical_name: FC Koln\n"
        "    csv: FC Koln\n"
        "    api: 1. FC Köln\n"
        "    league: BUNDESLIGA\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    config.load_team_aliases.cache_clear()

    try:
        aliases = {a.canonical_name: a for a in config.load_team_aliases()}
        assert aliases["Deportivo La Coruna"].api_name == "RC Deportivo La Coruña"
        assert aliases["Malaga"].api_name == "Málaga CF"
        assert aliases["FC Koln"].api_name == "1. FC Köln"
    finally:
        config.load_team_aliases.cache_clear()


def test_load_manual_injuries_decodes_utf8(tmp_path, monkeypatch):
    (tmp_path / "injuries.yaml").write_text(
        "injuries:\n"
        "  - team: Málaga\n"
        "    player: José García\n"
        "    position: attack\n"
        "    importance_weight: 0.5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    injuries = config.load_manual_injuries()
    assert injuries[0].team == "Málaga"
    assert injuries[0].player == "José García"
