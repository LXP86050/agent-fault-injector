from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from afi.clock import Clock
from afi.tasks import Task

Phase = Literal["before", "fault", "result", "error"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    prompt_tokens: int
    completion_tokens: int


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse: ...


@dataclass
class Event:
    seq: int
    t_ms: float
    target: str
    phase: Phase
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class World:
    """The dependencies an experiment can reach, before faults are wrapped on."""

    tools: dict[str, Callable[..., Any]]
    specs: tuple[ToolSpec, ...]


@dataclass
class Toolkit:
    """What an agent is handed for one scenario: already-instrumented dependencies."""

    tools: Mapping[str, Callable[..., Any]]
    specs: tuple[ToolSpec, ...]
    llm: LLMClient
    clock: Clock
    seed: int


class Agent(Protocol):
    def run(self, task: Task) -> str: ...
