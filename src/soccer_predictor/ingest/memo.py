"""A small in-process cache with an expiry, for results that are expensive to
rebuild but only change slowly.

The API clients already cache responses on disk, but reading and JSON-parsing
those files is itself slow when it happens over and over: the Leagues page
predicts every upcoming fixture, each prediction needs both teams' current-form
numbers, and each of those used to re-read ~30 cached files (about 5 seconds a
page render for MLS). Keeping the parsed result in memory for a few minutes
makes the repeats free. The result is shared between callers, so treat it as
read-only.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_registry: list[dict] = []


def ttl_memo(seconds: float) -> Callable[[F], F]:
    """Memoizes a function of hashable arguments for `seconds`."""

    def decorator(func: F) -> F:
        cache: dict[tuple, tuple[float, Any]] = {}
        lock = threading.Lock()
        _registry.append(cache)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            with lock:
                hit = cache.get(key)
                if hit is not None and now - hit[0] < seconds:
                    return hit[1]
            value = func(*args, **kwargs)  # computed outside the lock: two threads may both compute, harmlessly
            with lock:
                cache[key] = (time.monotonic(), value)
            return value

        return wrapper  # type: ignore[return-value]

    return decorator


def clear_all() -> None:
    """Empties every ttl_memo cache (tests, and anything that must see fresh data)."""
    for cache in _registry:
        cache.clear()
