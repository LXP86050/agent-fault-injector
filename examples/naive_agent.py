"""One-shot tool loop.

Calls the selected tool once, forwards the raw payload or the exception text,
and returns whatever the model says. There is no retry and no check that the
answer came from the tool.
"""

from __future__ import annotations

import json
from typing import Any

from afi.models import Message, Toolkit
from afi.tasks import Task


class NaiveAgent:
    def __init__(self, toolkit: Toolkit) -> None:
        self.toolkit = toolkit

    def run(self, task: Task) -> str:
        messages: list[Message] = [Message(role="user", content=task.prompt)]
        first = self.toolkit.llm.complete(messages, self.toolkit.specs)
        if not first.tool_calls:
            return first.content or ""
        call = first.tool_calls[0]
        content = _invoke(self.toolkit.tools[call.name], call.arguments)
        messages.append(
            Message(role="assistant", content=first.content or "", tool_calls=first.tool_calls)
        )
        messages.append(Message(role="tool", content=content, name=call.name, tool_call_id=call.id))
        second = self.toolkit.llm.complete(messages, self.toolkit.specs)
        return second.content or ""


def build(toolkit: Toolkit) -> NaiveAgent:
    return NaiveAgent(toolkit)


def _invoke(fn: Any, arguments: dict[str, Any]) -> str:
    try:
        value = fn(**arguments)
    except Exception as exc:
        return f"error: {exc}"
    if isinstance(value, str):
        return value
    return json.dumps(value)
