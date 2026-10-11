"""scripts/keep_streamlit_alive.py must survive the page reloading under it.

Right after a deploy (or while an app wakes up) Streamlit Cloud redirects and
reloads, so a frame can vanish between listing it and querying it: Playwright
raises "Frame was detached". That failed the scheduled job even though the app
was fine. Playwright isn't a project dependency, so it is stubbed here."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


class FakePlaywrightError(Exception):
    pass


@pytest.fixture
def script(monkeypatch):
    stub = types.ModuleType("playwright.sync_api")
    stub.Error = FakePlaywrightError
    stub.Page = object
    stub.sync_playwright = lambda: None
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", stub)
    path = Path(__file__).resolve().parents[1] / "scripts" / "keep_streamlit_alive.py"
    spec = importlib.util.spec_from_file_location("keep_streamlit_alive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Locator:
    def __init__(self, count=0, raises=False):
        self._count, self._raises = count, raises

    def count(self):
        if self._raises:
            raise FakePlaywrightError("Locator.count: Frame was detached")
        return self._count


class Frame:
    def __init__(self, ready=False, detached=False):
        self.ready, self.detached = ready, detached

    def locator(self, selector):
        return Locator(count=1 if self.ready else 0, raises=self.detached)

    def get_by_role(self, *a, **k):
        return Locator(raises=self.detached)


class Page:
    def __init__(self, frames, title_raises=False):
        self.frames, self._title_raises = frames, title_raises

    def title(self):
        if self._title_raises:
            raise FakePlaywrightError("Execution context was destroyed")
        return "Football Match Predictor"


def test_a_detached_frame_is_skipped_not_fatal(script):
    page = Page([Frame(detached=True), Frame(ready=True)])
    assert script._app_is_ready(page) is True


def test_only_detached_frames_just_means_not_ready_yet(script):
    assert script._app_is_ready(Page([Frame(detached=True)])) is False


def test_a_ready_page_with_no_detachment_still_works(script):
    assert script._app_is_ready(Page([Frame(ready=True)])) is True
    assert script._app_is_ready(Page([Frame(ready=False)])) is False


def test_wake_button_search_survives_a_detached_frame(script):
    assert script._click_wake_button_if_present(Page([Frame(detached=True)])) is False


def test_title_lookup_survives_a_reload(script):
    assert script._page_title(Page([], title_raises=True)) == "(page was reloading)"
    assert script._page_title(Page([])) == "Football Match Predictor"


def test_page_load_is_retried_then_succeeds(script, monkeypatch):
    monkeypatch.setattr(script.time, "sleep", lambda s: None)
    calls = []

    class Navigating:
        def goto(self, url, **kw):
            calls.append(url)
            if len(calls) < 3:
                raise FakePlaywrightError("net::ERR_ABORTED\nnavigation interrupted")

    script._open(Navigating(), "https://example.test/")
    assert len(calls) == 3


def test_page_load_gives_up_after_the_last_attempt(script, monkeypatch):
    monkeypatch.setattr(script.time, "sleep", lambda s: None)

    class Broken:
        def goto(self, url, **kw):
            raise FakePlaywrightError("net::ERR_NAME_NOT_RESOLVED")

    with pytest.raises(FakePlaywrightError):
        script._open(Broken(), "https://example.test/")
