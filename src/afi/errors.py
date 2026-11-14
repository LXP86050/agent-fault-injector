from __future__ import annotations


class FaultError(Exception):
    """A failure synthesized by the injector, or raised by a call guard."""


class ToolTimeout(FaultError, TimeoutError):
    def __init__(self, target: str, delay_ms: float) -> None:
        self.target = target
        self.delay_ms = delay_ms
        super().__init__(f"{target} timed out after {delay_ms:.0f}ms")


class ToolHTTPError(FaultError):
    def __init__(self, target: str, status: int, body: str = "") -> None:
        self.target = target
        self.status = status
        self.body = body
        super().__init__(f"{target} returned HTTP {status}")


class CircuitOpen(FaultError):
    def __init__(self, target: str) -> None:
        self.target = target
        super().__init__(f"circuit open for {target}")


class Retryable(FaultError):
    """The dependency answered, but the client should try again."""


# Statuses a client can treat as transient. 429 is included on purpose:
# rate limits clear, and a short backoff is the usual response.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
