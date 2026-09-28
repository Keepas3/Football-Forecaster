from __future__ import annotations

import soccer_predictor.config as config


def test_load_manual_captains_decodes_utf8(tmp_path, monkeypatch):
    (tmp_path / "captains.yaml").write_text(
        "captains:\n  - team: Arsenal\n    player: Martin Ødegaard\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    captains = config.load_manual_captains()
    assert captains[0].team == "Arsenal"
    assert captains[0].player == "Martin Ødegaard"


def test_load_manual_captains_empty_file(tmp_path, monkeypatch):
    (tmp_path / "captains.yaml").write_text("captains: []\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    assert config.load_manual_captains() == []


def test_load_manual_star_players_decodes_utf8_and_note(tmp_path, monkeypatch):
    (tmp_path / "star_players.yaml").write_text(
        "star_players:\n"
        "  - team: Real Madrid\n"
        "    player: Jude Bellingham\n"
        '    note: "Manual entry"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    stars = config.load_manual_star_players()
    assert stars[0].team == "Real Madrid"
    assert stars[0].player == "Jude Bellingham"
    assert stars[0].note == "Manual entry"


def test_load_manual_star_players_note_defaults_empty(tmp_path, monkeypatch):
    (tmp_path / "star_players.yaml").write_text(
        "star_players:\n  - team: Real Madrid\n    player: Jude Bellingham\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    assert config.load_manual_star_players()[0].note == ""


def test_load_manual_star_players_empty_file(tmp_path, monkeypatch):
    (tmp_path / "star_players.yaml").write_text("star_players: []\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    assert config.load_manual_star_players() == []
