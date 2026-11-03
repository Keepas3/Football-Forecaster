"""The Streamlit settings the deployed site depends on."""

from __future__ import annotations

import tomllib
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"


def test_the_source_file_watcher_is_off():
    """Each visitor's session would otherwise resolve the path of every loaded module (about 4,000) on the
    server's event loop -- seconds of CPU per page load, which is what froze the site on Community Cloud."""
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["server"]["fileWatcherType"] == "none"
