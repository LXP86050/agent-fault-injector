from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from afi.clock import Clock
from afi.errors import RETRYABLE_STATUS, CircuitOpen, Retryable, ToolHTTPError, ToolTimeout

T = TypeVar("T")

# Phrases that show up when a document is trying to steer the model.
# This is a heuristic, not a detector. It catches the fault this package injects
# and the obvious cousins of it.
INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous",
    "reply exactly:",
    "disregard the user",
)

_KEY_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_KEY_NOISE = re.compile(r"[^a-z0-9]+")


@dataclass
class RetryPolicy:
    """Exponential backoff with symmetric jitter.

    ``attempts`` counts the first try. The delay after failure ``n`` (starting
    at 0) is ``min(cap, base * 2^n)``, shifted by at most ``jitter`` of that
    delay in either direction. Jitter is drawn from the guard's seeded RNG.
    """

    attempts: int = 3
    base_ms: float = 20.0
    cap_ms: float = 200.0
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.base_ms < 0 or self.cap_ms < 0:
            raise ValueError("backoff delays must be >= 0")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be between 0 and 1")

    def delay_ms(self, retry_index: int, rng: random.Random) -> float:
        growth = self.base_ms * float(2**retry_index)
        base = self.cap_ms if growth > self.cap_ms else growth
        if self.jitter == 0 or base == 0:
            return base
        delta = base * self.jitter
        shift = float(rng.uniform(-delta, delta))
        return 0.0 if base + shift < 0 else base + shift


@dataclass
class CircuitBreaker:
    """Opens after ``failure_threshold`` consecutive failures.

    While open, calls fail immediately until ``reset_ms`` has passed on the
    experiment clock. The next call is a single half-open trial.
    """

    failure_threshold: int = 3
    reset_ms: float = 1_000.0

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.reset_ms < 0:
            raise ValueError("reset_ms must be >= 0")


@dataclass
class _Circuit:
    failures: int = 0
    state: str = "closed"
    opened_at: float = 0.0


@dataclass
class CallGuard:
    """Retry budget and a per-target circuit breaker sharing one clock.

    Validation failures raised as ``Retryable`` count the same as a timeout:
    the dependency did not give us something we can use. Non-retryable HTTP
    statuses and ordinary exceptions propagate on the first attempt and still
    count as a failure for the breaker.
    """

    clock: Clock
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._circuits: dict[str, _Circuit] = {}
        self.calls: dict[str, int] = {}
        self.short_circuits: dict[str, int] = {}

    def call(
        self,
        name: str,
        fn: Callable[..., T],
        /,
        *args: Any,
        validate: Callable[[T], T] | None = None,
        **kwargs: Any,
    ) -> T:
        last_error: BaseException | None = None
        for attempt in range(self.retry.attempts):
            self._enter(name)
            self.calls[name] = self.calls.get(name, 0) + 1
            try:
                value = fn(*args, **kwargs)
                if validate is not None:
                    value = validate(value)
            except Retryable as exc:
                self._record_failure(name)
                last_error = exc
            except ToolTimeout as exc:
                self._record_failure(name)
                last_error = exc
            except ToolHTTPError as exc:
                self._record_failure(name)
                last_error = exc
                if exc.status not in RETRYABLE_STATUS:
                    raise
            except Exception:
                self._record_failure(name)
                raise
            else:
                self._record_success(name)
                return value
            if self._circuits[name].state == "open":
                break
            if attempt + 1 < self.retry.attempts:
                self.clock.sleep_ms(self.retry.delay_ms(attempt, self._rng))
        assert last_error is not None
        raise last_error

    def _circuit(self, name: str) -> _Circuit:
        circuit = self._circuits.get(name)
        if circuit is None:
            circuit = _Circuit()
            self._circuits[name] = circuit
        return circuit

    def _enter(self, name: str) -> None:
        circuit = self._circuit(name)
        if circuit.state != "open":
            return
        if self.clock.now_ms() - circuit.opened_at >= self.breaker.reset_ms:
            circuit.state = "half_open"
            return
        self.short_circuits[name] = self.short_circuits.get(name, 0) + 1
        raise CircuitOpen(name)

    def _record_success(self, name: str) -> None:
        circuit = self._circuit(name)
        circuit.state = "closed"
        circuit.failures = 0

    def _record_failure(self, name: str) -> None:
        circuit = self._circuit(name)
        circuit.failures += 1
        if circuit.state == "half_open" or circuit.failures >= self.breaker.failure_threshold:
            circuit.state = "open"
            circuit.opened_at = self.clock.now_ms()


def looks_injected(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in INJECTION_MARKERS)


def as_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise Retryable("malformed json") from exc
    if not isinstance(value, dict):
        raise Retryable(f"expected a JSON object, got {type(value).__name__}")
    return value


def key_form(key: str) -> str:
    spaced = _KEY_BOUNDARY.sub(r"\g<1>_\g<2>", key).lower()
    return _KEY_NOISE.sub("_", spaced).strip("_")


def select_fields(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    """Pull ``keys`` out of an object, allowing drifted names.

    ``capital`` matches ``capital`` and ``capital_city``, not ``recapitalize``.
    A selected string that looks like an instruction is a retry, not data.
    """

    obj = as_object(value)
    forms = [(key, key_form(key)) for key in obj]
    picked: dict[str, Any] = {}
    for key in keys:
        want = key_form(key)
        exact = [original for original, form in forms if form == want]
        prefixed = [original for original, form in forms if form.startswith(want + "_")]
        if exact:
            chosen = exact[0]
        elif prefixed:
            chosen = min(prefixed, key=len)
        else:
            continue
        raw = obj[chosen]
        if isinstance(raw, str) and looks_injected(raw):
            raise Retryable(f"{chosen} contains instruction-shaped text")
        picked[key] = raw
    missing = [key for key in keys if key not in picked]
    if missing:
        raise Retryable("missing " + ", ".join(missing))
    return picked


def require_fresh(
    payload: dict[str, Any],
    *,
    now_ms: float,
    max_age_ms: float,
    time_field: str = "as_of_ms",
) -> dict[str, Any]:
    stamp = payload.get(time_field)
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        raise Retryable(f"missing {time_field}")
    age = now_ms - float(stamp)
    if age > max_age_ms:
        raise Retryable(f"stale by {age:.0f}ms")
    return payload


def collect_partial(
    guard: CallGuard,
    name: str,
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    list_field: str = "items",
    total_field: str = "total",
    id_field: str = "sku",
) -> dict[str, Any]:
    """Retry a list read and union rows until ``total`` distinct ids are held.

    Each attempt may return a different slice. Rows already collected stay.
    The guard's breaker still applies, so a dead dependency stops the merge.
    """

    bag: dict[Any, dict[str, Any]] = {}
    reported = 0

    def validate(payload: Any) -> dict[str, Any]:
        nonlocal reported
        obj = as_object(payload)
        items = obj.get(list_field)
        if not isinstance(items, list):
            raise Retryable(f"missing {list_field}")
        total = obj.get(total_field)
        if isinstance(total, int) and not isinstance(total, bool):
            reported = total
        for item in items:
            if not isinstance(item, dict) or id_field not in item:
                raise Retryable(f"{list_field} row has no {id_field}")
            bag[item[id_field]] = item
        if reported and len(bag) < reported:
            raise Retryable(f"have {len(bag)} of {reported}")
        return {list_field: list(bag.values()), total_field: reported or len(bag)}

    return guard.call(name, fn, validate=validate, **kwargs)
