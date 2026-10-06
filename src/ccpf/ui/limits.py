"""Per-session usage limits for the public chat app.

The shared narration budget (config.yaml `narration.budget_usd`) is a global
cap; without a per-session limit, one visitor could drain it for everyone.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Callable

MAX_MESSAGE_CHARS = 1000

DISCLAIMER = (
    "Informational only — not legal advice. Figures summarize past NCDRC judgments; "
    "your outcome depends on the facts of your case. Consult a lawyer."
)


class SessionRateLimiter:
    """Sliding-window limiter: at most `max_requests` per `window_s` seconds."""

    def __init__(self, max_requests: int = 10, window_s: float = 600.0, clock: Callable[[], float] = time.monotonic):
        self.max_requests = max_requests
        self.window_s = window_s
        self._clock = clock
        self._hits: deque[float] = deque()

    def _prune(self) -> None:
        cutoff = self._clock() - self.window_s
        while self._hits and self._hits[0] <= cutoff:
            self._hits.popleft()

    def allow(self) -> bool:
        """Record a request if under the limit; return whether it was allowed."""
        self._prune()
        if len(self._hits) >= self.max_requests:
            return False
        self._hits.append(self._clock())
        return True

    def retry_after_s(self) -> float:
        self._prune()
        if len(self._hits) < self.max_requests:
            return 0.0
        return max(0.0, self._hits[0] + self.window_s - self._clock())
