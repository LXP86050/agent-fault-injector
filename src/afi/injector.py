from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import is_dataclass
from typing import Any, TypeVar, cast

from afi.clock import Clock
from afi.faults import FaultContext, Invocation, apply_fault
from afi.models import Event, LLMClient, LLMResponse, Message, Phase, ToolSpec
from afi.scenario import FaultSpec, Scenario

T = TypeVar("T")


def snapshot(value: Any) -> Any:
    """Copy a return value into plain JSON-ish data for the trace."""

    if is_dataclass(value) and not isinstance(value, type):
        return {name: snapshot(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class _LLMProxy:
    def __init__(self, injector: Injector, client: LLMClient) -> None:
        self._injector = injector
        self._client = client

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        return self._injector.invoke(
            "llm",
            lambda: self._client.complete(messages, tools),
            (),
            {},
        )


class Injector:
    """Wraps one scenario around tool calls and the LLM client.

    The first armed fault whose probability roll succeeds is the one that
    runs. Later faults in the file do not stack on that call. A fault that is
    not armed yet (``call_index <= after_call``) does not consume a random
    draw. Call indexes keep climbing for the whole scenario, including retries,
    so ``after_call`` is a warm-up for the dependency rather than for one task.
    """

    def __init__(self, scenario: Scenario, *, clock: Clock, rng: random.Random) -> None:
        self.scenario = scenario
        self.clock = clock
        self.rng = rng
        self.trace: list[Event] = []
        self._counts: dict[str, int] = {}
        self._cache: dict[str, Any] = {}
        self._seq = 0

    def reset_trace(self) -> None:
        self.trace.clear()

    def tool(self, name: str, fn: Callable[..., T]) -> Callable[..., T]:
        def wrapped(*args: Any, **kwargs: Any) -> T:
            return self.invoke(name, fn, args, kwargs)

        wrapped.__name__ = name
        return wrapped

    def llm(self, client: LLMClient) -> LLMClient:
        return _LLMProxy(self, client)

    def invoke(
        self,
        target: str,
        fn: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> T:
        index = self._counts.get(target, 0) + 1
        self._counts[target] = index
        self._event("before", target, {"call_index": index, **_brief_args(args, kwargs)})
        spec = self._pick(target, index)
        if spec is None:
            return self._direct(target, fn, args, kwargs)

        # Stamp the fault before applying it so the sleep counts as recovery time.
        event = self._event("fault", target, {"fault": spec.kind, "call_index": index})
        inv = Invocation(target=target, call_index=index, args=args, kwargs=kwargs, fn=fn)
        ctx = FaultContext(clock=self.clock, cache=self._cache)
        try:
            outcome = apply_fault(spec.kind, inv, spec.params, ctx)
        except Exception as exc:
            self._event("error", target, {"type": type(exc).__name__, "message": str(exc)})
            raise
        event.detail.update(outcome.detail)
        if outcome.ran:
            self._cache[target] = snapshot(outcome.produced)
        if outcome.error is not None:
            err = outcome.error
            self._event("error", target, {"type": type(err).__name__, "message": str(err)})
            raise err
        self._event("result", target, {"value": snapshot(outcome.value)})
        return cast(T, outcome.value)

    def _direct(
        self,
        target: str,
        fn: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> T:
        try:
            value = fn(*args, **kwargs)
        except Exception as exc:
            self._event("error", target, {"type": type(exc).__name__, "message": str(exc)})
            raise
        self._cache[target] = snapshot(value)
        self._event("result", target, {"value": snapshot(value)})
        return value

    def _pick(self, target: str, index: int) -> FaultSpec | None:
        for spec in self.scenario.faults:
            if not _in_radius(spec.target, target, self.scenario.blast_radius):
                continue
            if index <= spec.after_call:
                continue
            if self.rng.random() < spec.probability:
                return spec
        return None

    def _event(self, phase: Phase, target: str, detail: dict[str, Any]) -> Event:
        self._seq += 1
        event = Event(
            seq=self._seq,
            t_ms=self.clock.now_ms(),
            target=target,
            phase=phase,
            detail=detail,
        )
        self.trace.append(event)
        return event


def _in_radius(spec_target: str, target: str, radius: frozenset[str]) -> bool:
    if spec_target == "*":
        return target in radius
    return spec_target == target


def _brief_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    if args:
        detail["args"] = snapshot(args)
    if kwargs:
        detail["kwargs"] = snapshot(kwargs)
    return detail
