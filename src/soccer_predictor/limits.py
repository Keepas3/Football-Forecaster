"""A small sliding-window counter shared by the things that must not be hammered:
password gates (failed attempts, across every visitor at once) and the news
search (uncached trips to Google). Per-session limits can't stop someone who
simply opens a new session, so these count for the whole server process.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class SlidingWindow:
    """At most `max_events` events in any `window_seconds`."""

    def __init__(self, max_events: int, window_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._clock = clock
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        while self._events and now - self._events[0] >= self.window_seconds:
            self._events.popleft()

    def blocked(self) -> bool:
        """True while the window is full (nothing is recorded)."""
        with self._lock:
            self._expire(self._clock())
            return len(self._events) >= self.max_events

    def record(self) -> None:
        with self._lock:
            now = self._clock()
            self._expire(now)
            self._events.append(now)

    def allow(self) -> bool:
        """Records an event and returns True if the window had room; returns
        False (recording nothing) when it is full."""
        with self._lock:
            now = self._clock()
            self._expire(now)
            if len(self._events) >= self.max_events:
                return False
            self._events.append(now)
            return True
