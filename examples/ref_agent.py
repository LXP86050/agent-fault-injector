"""Reference agent.

Wraps each tool with retries, backoff, and a circuit breaker. Payloads are
treated as data: unknown fields are dropped, drifted keys are matched on word
boundaries, partial lists are merged until the reported total is in hand, and
prices older than a second on the experiment clock are refused. If the tool
never returns something usable, the agent abstains instead of guessing.
"""

from __future__ import annotations

import json
from typing import Any

from afi.models import Message, ToolCall, Toolkit
from afi.resilience import (
    CallGuard,
    CircuitBreaker,
    RetryPolicy,
    collect_partial,
    require_fresh,
    select_fields,
)
from afi.tasks import Task

# Four attempts is enough for the shipped seeds, where a transient fault
# misses at least once inside the first three draws. The breaker opens on the
# third consecutive failure so a hard outage does not burn the whole budget,
# and the next task sees the open circuit immediately.
RETRY = RetryPolicy(attempts=4, base_ms=10, cap_ms=80, jitter=0.2)
BREAKER = CircuitBreaker(failure_threshold=3, reset_ms=500)
PRICE_MAX_AGE_MS = 1000

_FIELDS: dict[str, tuple[str, ...]] = {
    "lookup_country": ("name", "capital"),
    "get_ticket": ("id", "status"),
    "healthcheck": ("status",),
}


class ReferenceAgent:
    def __init__(self, toolkit: Toolkit) -> None:
        self.toolkit = toolkit
        self.guard = CallGuard(
            toolkit.clock,
            retry=RETRY,
            breaker=BREAKER,
            seed=toolkit.seed,
        )

    def run(self, task: Task) -> str:
        messages: list[Message] = [Message(role="user", content=task.prompt)]
        first = self.toolkit.llm.complete(messages, self.toolkit.specs)
        if not first.tool_calls:
            return first.content or "ABSTAIN"
        call = first.tool_calls[0]
        try:
            cleaned = self._call_tool(call)
        except Exception:
            # A down dependency is not a prompt to invent the fact.
            return "ABSTAIN"
        messages.append(
            Message(role="assistant", content=first.content or "", tool_calls=first.tool_calls)
        )
        messages.append(
            Message(
                role="tool",
                content=json.dumps(cleaned),
                name=call.name,
                tool_call_id=call.id,
            )
        )
        second = self.toolkit.llm.complete(messages, self.toolkit.specs)
        return second.content or "ABSTAIN"

    def _call_tool(self, call: ToolCall) -> dict[str, Any]:
        fn = self.toolkit.tools[call.name]
        if call.name == "list_stock":
            return collect_partial(self.guard, call.name, fn, dict(call.arguments))
        if call.name == "get_price":
            return self.guard.call(
                call.name,
                fn,
                validate=self._fresh_price,
                **call.arguments,
            )
        fields = _FIELDS[call.name]
        return self.guard.call(
            call.name,
            fn,
            validate=lambda value: select_fields(value, fields),
            **call.arguments,
        )

    def _fresh_price(self, value: Any) -> dict[str, Any]:
        picked = select_fields(value, ("symbol", "price", "as_of_ms"))
        return require_fresh(
            picked,
            now_ms=self.toolkit.clock.now_ms(),
            max_age_ms=PRICE_MAX_AGE_MS,
        )


def build(toolkit: Toolkit) -> ReferenceAgent:
    return ReferenceAgent(toolkit)
