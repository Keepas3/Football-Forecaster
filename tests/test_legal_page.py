"""The Privacy & Terms page, its contact settings, and its place in the navigation."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
import yaml
from streamlit.testing.v1 import AppTest

import soccer_predictor.config as config
from soccer_predictor.config import CONFIG_DIR, SiteInfo
from soccer_predictor.dashboard import components, navigation
from soccer_predictor.dashboard.views import legal


def _page(monkeypatch, site=SiteInfo()):
    monkeypatch.setattr(legal, "load_site_info", lambda: site)

    def app():
        from soccer_predictor.dashboard.views import legal as view

        view.render()

    at = AppTest.from_function(app, default_timeout=30).run()
    assert not at.exception
    return at


def _text(at) -> str:
    return "\n".join(m.value for m in at.markdown) + "\n" + "\n".join(c.value for c in at.caption)


def test_the_page_covers_what_the_site_does_with_a_visitors_data(monkeypatch):
    at = _page(monkeypatch)
    text = _text(at)

    assert [t.value for t in at.title] == ["Privacy & Terms"]
    for must_say in (
        "not betting or financial advice",
        "Gemini",  # the what-if note leaves the site
        "Google News",  # so does a news search
        "Streamlit Community Cloud",
        "Last updated",
        "no accounts",
    ):
        assert must_say.lower() in text.lower(), must_say


def test_every_site_a_visitors_browser_loads_files_from_is_named(monkeypatch):
    text = _text(_page(monkeypatch))
    hosts = {urlparse(components.FLAG_FONT_URL).netloc}
    for league in yaml.safe_load((CONFIG_DIR / "leagues.yaml").read_text(encoding="utf-8"))["leagues"]:
        if league.get("flag_url"):
            hosts.add(urlparse(league["flag_url"]).netloc)

    assert hosts, "no hosts found: the check itself is broken"
    for host in hosts | {"crests.football-data.org", "a.espncdn.com", "upload.wikimedia.org"}:
        assert host in text, f"{host} is fetched by visitors' browsers but the page does not name it"


def test_no_contact_section_until_an_operator_or_email_is_set(monkeypatch):
    at = _page(monkeypatch, SiteInfo())
    assert "Contact" not in [s.value for s in at.subheader]


def test_the_contact_section_names_the_operator_and_links_the_email(monkeypatch):
    at = _page(monkeypatch, SiteInfo(operator="A. Person", contact_email="hello@example.org"))

    assert "Contact" in [s.value for s in at.subheader]
    text = _text(at)
    assert "run by A. Person" in text
    assert "[hello@example.org](mailto:hello@example.org)" in text


def test_an_operator_name_cannot_inject_markup_into_the_page(monkeypatch):
    monkeypatch.setattr(legal, "load_site_info", lambda: SiteInfo(operator="<script>alert(1)</script>"))

    assert "<script>" not in legal.contact_markdown()


# --- config/site.yaml ----------------------------------------------------------------------------


def test_the_shipped_site_file_is_valid_and_blank_so_nothing_personal_is_published_by_default():
    raw = yaml.safe_load((CONFIG_DIR / "site.yaml").read_text(encoding="utf-8"))
    assert raw == {"operator": "", "contact_email": ""}


def test_site_info_is_empty_when_the_file_is_missing_or_blank(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    assert config.load_site_info() == SiteInfo()  # no file

    (tmp_path / "site.yaml").write_text("# nothing yet\n", encoding="utf-8")
    assert config.load_site_info() == SiteInfo()  # a comment only parses to nothing


def test_site_info_reads_utf8_values_and_trims_them(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    (tmp_path / "site.yaml").write_text('operator: "  José Núñez  "\ncontact_email: " me@example.org "\n', encoding="utf-8")

    assert config.load_site_info() == SiteInfo(operator="José Núñez", contact_email="me@example.org")


# --- navigation ----------------------------------------------------------------------------------


def test_the_whole_app_lists_the_page_in_its_navigation(monkeypatch):
    registered = []
    real_navigation = st.navigation

    def spy(pages, *args, **kwargs):
        page = real_navigation(pages, *args, **kwargs)
        registered.extend((p.url_path, p.visibility) for p in pages)
        return page

    monkeypatch.setattr(st, "navigation", spy)
    app_file = Path(navigation.__file__).with_name("app.py")

    at = AppTest.from_file(str(app_file), default_timeout=120).run()

    assert not at.exception
    assert ("privacy", "visible") in registered  # a tab in the sidebar, not a hidden page


def test_a_trimmed_status_note_still_explains_early_and_locked():
    note = components.PREDICTION_STATUS_NOTE
    assert "Early" in note and "Locked" in note and "Track Record" in note
    assert len(note) < 200  # the full explanation is on the Privacy & Terms page
    assert "Early" in legal.FORECASTS and "Locked" in legal.FORECASTS
