from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from afi.errors import ToolHTTPError, ToolTimeout
from afi.models import LLMResponse, Message, ToolCall, ToolSpec


def estimate_tokens(text: str) -> int:
    """Rough stand-in for a tokenizer. Stable, and good enough to compare runs."""

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class OpenAICompatible:
    """Minimal chat-completions client for an OpenAI-compatible endpoint.

    The API key is taken by the constructor. Pass it from the environment at
    the call site so it does not land in a process listing via the CLI.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": [_message(item) for item in messages],
            "tools": [_tool(item) for item in tools],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise ToolHTTPError("llm", exc.code, detail) from exc
        except TimeoutError as exc:
            raise ToolTimeout("llm", self.timeout_s * 1000) from exc
        if not isinstance(payload, dict):
            raise ToolHTTPError("llm", 502, "response was not a JSON object")
        return _response(payload)


def _message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id or "",
        }
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ],
        }
    return {"role": message.role, "content": message.content}


def _tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _response(payload: dict[str, Any]) -> LLMResponse:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ToolHTTPError("llm", 502, "response missing choices[0].message") from exc
    calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ToolHTTPError("llm", 502, "tool call arguments were not JSON") from exc
        elif isinstance(arguments, dict):
            parsed = arguments
        else:
            raise ToolHTTPError("llm", 502, "tool call arguments were not an object")
        calls.append(
            ToolCall(
                id=str(raw.get("id", "")),
                name=str(function.get("name", "")),
                arguments=parsed,
            )
        )
    usage = payload.get("usage") or {}
    content = message.get("content")
    return LLMResponse(
        content=content if isinstance(content, str) or content is None else str(content),
        tool_calls=tuple(calls),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )
