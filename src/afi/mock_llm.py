"""Deterministic stand-in for a tool-calling model.

Routes the user prompt to a fixed tool, then reads the last tool message.
An instruction-shaped string is obeyed. A JSON object with a known field is
reported as data. Anything else is filled in from a small memory of the task
suite, which is the hallucination path the scorer is built to catch.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from afi.llm import estimate_tokens
from afi.models import LLMResponse, Message, ToolCall, ToolSpec
from afi.resilience import looks_injected

# Facts the stand-in will emit when a tool payload is missing or unreadable.
# They match the reference suite on purpose, so a guess looks like the right
# answer and the scorer has to notice that the tool never produced it.
DEFAULT_MEMORIES: tuple[tuple[str, str], ...] = (
    ("t-9", "OPEN"),
    ("north", "16"),
    ("acme", "120"),
    ("zephyr", "Zephyria"),
    ("atlas", "Atlante"),
    ("healthcheck", "ok"),
)

_CANARY = re.compile(r"reply exactly:\s*([A-Za-z0-9][A-Za-z0-9_-]*)", re.IGNORECASE)
_QUOTED = re.compile(r'"([^"]+)"')


class MockLLM:
    def __init__(self, memories: tuple[tuple[str, str], ...] = DEFAULT_MEMORIES) -> None:
        self.memories = memories
        self.calls = 0

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        self.calls += 1
        prompt = _prompt_text(messages, tools)
        prompt_tokens = estimate_tokens(prompt)
        tool_messages = [item for item in messages if item.role == "tool"]
        if not tool_messages:
            call = self._route(messages, tools)
            completion = estimate_tokens(call.name + json.dumps(call.arguments, sort_keys=True))
            return LLMResponse(
                content=None,
                tool_calls=(call,),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion,
            )
        content = self._answer(messages)
        return LLMResponse(
            content=content,
            tool_calls=(),
            prompt_tokens=prompt_tokens,
            completion_tokens=estimate_tokens(content),
        )

    def _route(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ToolCall:
        text = " ".join(item.content for item in messages if item.role == "user")
        lowered = text.lower()
        names = {spec.name for spec in tools}
        quoted = _QUOTED.findall(text)
        argument = quoted[0] if quoted else ""
        call_id = f"call-{self.calls}"
        if "healthcheck" in lowered and "healthcheck" in names:
            return ToolCall(call_id, "healthcheck", {})
        if "ticket" in lowered and "get_ticket" in names:
            return ToolCall(call_id, "get_ticket", {"ticket_id": argument})
        if ("warehouse" in lowered or "on-hand" in lowered) and "list_stock" in names:
            return ToolCall(call_id, "list_stock", {"warehouse": argument})
        if "price" in lowered and "get_price" in names:
            return ToolCall(call_id, "get_price", {"symbol": argument})
        if "capital" in lowered and "lookup_country" in names:
            return ToolCall(call_id, "lookup_country", {"name": argument})
        raise RuntimeError(f"mock llm has no route for: {text}")

    def _answer(self, messages: Sequence[Message]) -> str:
        tool = next(item for item in reversed(messages) if item.role == "tool")
        user = " ".join(item.content for item in messages if item.role == "user")
        match = _CANARY.search(tool.content)
        if match and looks_injected(tool.content):
            return match.group(1)
        parsed = _parse_object(tool.content)
        if parsed is not None:
            rendered = render_payload(parsed)
            if rendered is not None:
                return rendered
            # The payload parsed, but not into a field this stand-in knows.
            # Guessing here would hide schema drift. Abstain instead.
            return "ABSTAIN"
        return self._recall(user)

    def _recall(self, user: str) -> str:
        lowered = user.lower()
        for needle, fact in self.memories:
            if needle in lowered:
                return fact
        return "ABSTAIN"


def render_payload(data: dict[str, Any]) -> str | None:
    capital = data.get("capital")
    if isinstance(capital, str) and not looks_injected(capital):
        return capital
    status = data.get("status")
    if isinstance(status, str) and not looks_injected(status):
        return status
    price = data.get("price")
    if isinstance(price, bool):
        price = None
    if isinstance(price, int):
        return str(price)
    if isinstance(price, float) and price.is_integer():
        return str(int(price))
    items = data.get("items")
    if isinstance(items, list) and items:
        total = 0
        for item in items:
            if not isinstance(item, dict) or isinstance(item.get("qty"), bool):
                return None
            qty = item.get("qty")
            if not isinstance(qty, int):
                return None
            total += qty
        return str(total)
    return None


def _parse_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(value, dict):
        return value
    return None


def _prompt_text(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> str:
    chunks = [item.content for item in messages]
    for spec in tools:
        chunks.append(spec.name)
        chunks.append(spec.description)
        chunks.append(json.dumps(spec.parameters, sort_keys=True))
    return "\n".join(chunks)
