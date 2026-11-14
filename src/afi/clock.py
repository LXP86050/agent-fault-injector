from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> float: ...

    def sleep_ms(self, duration_ms: float) -> None: ...


class SystemClock:
    def now_ms(self) -> float:
        return time.monotonic() * 1000.0

    def sleep_ms(self, duration_ms: float) -> None:
        if duration_ms < 0:
            raise ValueError("duration_ms must be >= 0")
        if duration_ms > 0:
            time.sleep(duration_ms / 1000.0)


class ManualClock:
    """Experiment clock. ``sleep_ms`` advances time and does not block.

    Scenarios stay reproducible because latency and backoff move this clock
    instead of the wall clock. Pass ``--realtime`` when you want real sleeps.
    """

    def __init__(self, start_ms: float = 0.0) -> None:
        self._now = start_ms

    def now_ms(self) -> float:
        return self._now

    def sleep_ms(self, duration_ms: float) -> None:
        if duration_ms < 0:
            raise ValueError("duration_ms must be >= 0")
        self._now += duration_ms
