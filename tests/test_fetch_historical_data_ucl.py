"""scripts/fetch_historical_data.py: the Champions League's training-only seasons come from
the archive, the newer ones are left for football-data.org."""

from __future__ import annotations

import contextlib
import datetime as dt
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from soccer_predictor.config import League
from soccer_predictor.ingest import team_mapper
from soccer_predictor.ingest.team_mapper import seed_teams_and_aliases
from soccer_predictor.storage.models import Base
from soccer_predictor.storage.repository import matches_for_league

UCL = League(
    code="UCL",
    name="UEFA Champions League",
    api_competition_id=2001,
    seasons=["2425"],
    history_seasons=["1819", "1920", "2324"],
    trainable=True,
)


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_historical_data.py"
    spec = importlib.util.spec_from_file_location("fetch_historical_data_ucl_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def session(script, monkeypatch, tmp_path):
    monkeypatch.setattr(team_mapper, "UNMATCHED_LOG", tmp_path / "unmatched.log")
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    s = Session(engine)

    @contextlib.contextmanager
    def scope():
        yield s
        s.flush()

    monkeypatch.setattr(script, "session_scope", scope)
    seed_teams_and_aliases(s, "UCL")
    s.commit()
    yield s
    s.close()


def _frame(*rows):
    return pd.DataFrame(rows, columns=["date", "home_team_name", "away_team_name", "home_goals", "away_goals"])


def test_archive_seasons_are_stored_and_the_rest_are_left_to_football_data_org(script, session, monkeypatch, capsys):
    frames = {
        "1819": _frame((dt.date(2019, 5, 7), "Liverpool FC", "FC Barcelona", 4, 0)),
        "1920": _frame((dt.date(2020, 8, 7), "Manchester City", "Real Madrid", 2, 1)),
    }
    monkeypatch.setattr(script.ucl_archive, "matches_frame", lambda season: frames[season])

    remaining = script._sync_history_seasons(UCL)

    assert remaining == ["2324"]
    stored = matches_for_league(session, "UCL")
    assert sorted(zip(stored["season"], stored["home_goals"], stored["away_goals"])) == [("1819", 4, 0), ("1920", 2, 1)]
    assert "1819: 1 matches ingested from the archive, 0 skipped" in capsys.readouterr().out


def test_an_unknown_club_is_skipped_and_counted_not_guessed(script, session, monkeypatch, capsys):
    monkeypatch.setattr(
        script.ucl_archive,
        "matches_frame",
        lambda season: _frame((dt.date(2019, 5, 7), "Liverpool FC", "Totally Unknown FC", 1, 0)),
    )
    script._sync_history_seasons(UCL)
    assert matches_for_league(session, "UCL").empty
    assert "0 matches ingested from the archive, 1 skipped" in capsys.readouterr().out


def test_a_failed_download_skips_that_season_and_carries_on(script, session, monkeypatch, capsys):
    def frames(season):
        if season == "1819":
            raise requests.ConnectionError("offline")
        return _frame((dt.date(2020, 8, 7), "Manchester City", "Real Madrid", 2, 1))

    monkeypatch.setattr(script.ucl_archive, "matches_frame", frames)

    remaining = script._sync_history_seasons(UCL)

    assert remaining == ["2324"]
    assert list(matches_for_league(session, "UCL")["season"]) == ["1920"]
    assert "season 1819: skipped" in capsys.readouterr().out


def test_another_league_never_reads_the_champions_league_archive(script, session, monkeypatch):
    monkeypatch.setattr(script.ucl_archive, "matches_frame", lambda season: pytest.fail("archive used"))
    other = League(code="XYZ", name="Other", seasons=["2425"], history_seasons=["1819"], trainable=True)
    assert script._sync_history_seasons(other) == ["1819"]
