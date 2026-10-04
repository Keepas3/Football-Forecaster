"""Keeps the deployed Streamlit app awake by visiting it in a real browser.

Streamlit Community Cloud puts an app to sleep after ~12 hours with no
visitors, and only counts a visit once a browser opens the app's websocket
session -- a plain HTTP request (the old `curl` ping) just receives the
static HTML shell, returns 200 whether the app is asleep or not, and never
touches that timer. This loads the page in headless Chrome, clicks "Yes, get
this app back up!" if the app is already asleep, then waits for the app's own
UI to render (proof the script actually ran) and stays connected a few
seconds before leaving.

Exits non-zero if the app never comes up, so the scheduled workflow goes red
instead of silently "succeeding".

Usage (needs `pip install playwright` and Google Chrome; the GitHub Actions
ubuntu runners already have Chrome):
    python scripts/keep_streamlit_alive.py [URL]
"""

from __future__ import annotations

import os
import sys
import time

from playwright.sync_api import Page, sync_playwright

DEFAULT_URL = "https://football-forecaster.streamlit.app/"
# A cold start (installing dependencies, loading the DB) can take a minute or
# two after the app has been asleep.
APP_READY_TIMEOUT_SECONDS = 180
# Stay on the page this long once it's up, so the session registers as a real visit.
LINGER_SECONDS = 15
WAKE_BUTTON_TEXT = "get this app back up"
# The app's own sidebar navigation -- only present once Streamlit has run the
# app script (the sleeping page and the "Please wait" shell don't have it).
APP_READY_SELECTOR = '[data-testid="stSidebarNav"], [data-testid="stSidebar"]'


def _frames(page: Page):
    # Community Cloud may serve the app in an iframe inside its own wrapper.
    return page.frames


def _click_wake_button_if_present(page: Page) -> bool:
    for frame in _frames(page):
        button = frame.get_by_role("button", name=WAKE_BUTTON_TEXT, exact=False)
        if button.count() and button.first.is_visible():
            button.first.click()
            return True
    return False


def _app_is_ready(page: Page) -> bool:
    return any(frame.locator(APP_READY_SELECTOR).count() > 0 for frame in _frames(page))


def keep_alive(url: str) -> bool:
    with sync_playwright() as p:
        # Installed Chrome, so there's no browser download step.
        browser = p.chromium.launch(channel="chrome", headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)

            woke_it = False
            deadline = time.monotonic() + APP_READY_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if _app_is_ready(page):
                    print(f"App is up{' (it was asleep; woke it)' if woke_it else ''}: {page.title()!r}")
                    time.sleep(LINGER_SECONDS)
                    return True
                if not woke_it and _click_wake_button_if_present(page):
                    woke_it = True
                    print("App was asleep -- clicked the wake-up button, waiting for it to start...")
                time.sleep(2)

            print(f"App did not come up within {APP_READY_TIMEOUT_SECONDS}s. Page title: {page.title()!r}")
            return False
        finally:
            browser.close()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("STREAMLIT_APP_URL", DEFAULT_URL)
    sys.exit(0 if keep_alive(target) else 1)
