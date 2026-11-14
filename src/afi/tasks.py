from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """One scripted job in a suite.

    ``expect`` is the answer text that counts as success. ``evidence`` is the
    set of leaf values that must have appeared in a tool payload before that
    answer counts as grounded. If the answer shows up without the evidence,
    the scorer marks a hallucination.
    """

    id: str
    prompt: str
    expect: str
    evidence: tuple[str, ...]
    description: str = ""
