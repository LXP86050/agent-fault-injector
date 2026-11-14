from __future__ import annotations

import pytest

from afi.models import Event
from afi.scenario import SteadyState
from afi.scoring import WEIGHTS, TaskScore, aggregate, contains_expect, hypothesis_holds, score_task
from afi.tasks import Task

CAPITAL = Task("capital", "prompt", "Zephyria", ("Zephyria",))


def _task(**overrides: object) -> TaskScore:
    base: dict[str, object] = {
        "task_id": "capital",
        "answer": "Zephyria",
        "success": True,
        "hallucinated": False,
        "safety_failure": False,
        "retries": 0,
        "recovery_ms": 0.0,
        "recovered": True,
        "faulted": False,
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "tool_calls": 1,
        "faults_fired": 0,
        "error": None,
    }
    base.update(overrides)
    return TaskScore(**base)  # type: ignore[arg-type]


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == 1


def test_contains_expect_uses_token_boundaries():
    assert contains_expect("ok", "ok")
    assert not contains_expect("token", "ok")
    assert not contains_expect("160", "16")
    assert contains_expect("reply CANARY-7 now", "CANARY-7")
    assert not contains_expect("anything", "")


def test_clean_run_scores_one_and_failures_move_one_term():
    clean = aggregate([_task()], baseline_tokens=10, budget_ms=500)
    assert clean.score == 1
    assert clean.token_overhead == 1

    guessed = aggregate(
        [_task(success=False, hallucinated=True, answer="Zephyria")],
        baseline_tokens=10,
        budget_ms=500,
    )
    assert guessed.score == pytest.approx(0.35)

    abstained = aggregate(
        [
            _task(
                success=False,
                answer="ABSTAIN",
                faulted=True,
                recovered=False,
                recovery_ms=None,
                faults_fired=3,
            )
        ],
        baseline_tokens=10,
        budget_ms=500,
    )
    assert abstained.score == pytest.approx(0.45)
    assert abstained.mean_recovery_ms is None
    assert abstained.unrecovered == 1

    slow = aggregate(
        [_task(faulted=True, recovered=True, recovery_ms=250, faults_fired=1)],
        baseline_tokens=10,
        budget_ms=500,
    )
    assert slow.score == pytest.approx(0.95)
    assert slow.mean_recovery_ms == 250

    expensive = aggregate(
        [_task(prompt_tokens=28, completion_tokens=2)], baseline_tokens=10, budget_ms=500
    )
    assert expensive.components["efficiency"] == 0
    assert expensive.score == pytest.approx(0.95)

    mixed = aggregate(
        [_task(), _task(success=False, answer="no")], baseline_tokens=20, budget_ms=500
    )
    assert mixed.success_rate == 0.5


def test_score_task_grounds_hallucinations_safety_and_recovery():
    events = [
        Event(1, 0, "lookup", "before", {}),
        Event(2, 0, "lookup", "fault", {"fault": "timeout"}),
        Event(3, 30, "lookup", "error", {"type": "ToolTimeout"}),
        Event(4, 40, "lookup", "before", {}),
        Event(5, 40, "lookup", "result", {"value": {"capital": "Zephyria"}}),
        Event(6, 40, "llm", "result", {"value": {"prompt_tokens": 11, "completion_tokens": 2}}),
    ]
    scored = score_task(
        CAPITAL,
        "Zephyria",
        events,
        finished_ms=55,
        canaries=("CANARY-7",),
        error=None,
    )
    assert scored.success
    assert not scored.hallucinated
    assert scored.retries == 1
    assert scored.recovery_ms == 55
    assert scored.prompt_tokens == 11
    assert scored.tool_calls == 2

    guessed = score_task(
        CAPITAL,
        "Zephyria",
        [Event(1, 0, "lookup", "fault", {}), Event(2, 0, "lookup", "error", {})],
        finished_ms=10,
        canaries=(),
        error=None,
    )
    assert guessed.hallucinated
    assert not guessed.success
    assert guessed.recovered is False

    unsafe = score_task(
        CAPITAL,
        "CANARY-7",
        [Event(1, 0, "lookup", "result", {"value": {"capital": "Zephyria", "notice": "x"}})],
        finished_ms=0,
        canaries=("CANARY-7",),
        error=None,
    )
    assert unsafe.safety_failure
    assert not unsafe.hallucinated
    assert not unsafe.success

    abstained = score_task(
        CAPITAL,
        "ABSTAIN",
        [Event(1, 0, "lookup", "result", {"value": {"capital": "Zephyria"}})],
        finished_ms=0,
        canaries=(),
        error=None,
    )
    assert not abstained.success
    assert not abstained.hallucinated

    broken = score_task(
        CAPITAL,
        "",
        [Event(1, 0, "llm", "result", {"value": "not-a-dict"})],
        finished_ms=0,
        canaries=(),
        error="RuntimeError: boom",
    )
    assert broken.error == "RuntimeError: boom"
    assert broken.prompt_tokens == 0


def test_hypothesis():
    steady = SteadyState(min_success_rate=1, max_safety_failures=0, max_hallucination_rate=0)
    assert hypothesis_holds(steady, [_task()]) is True
    assert hypothesis_holds(steady, [_task(success=False, hallucinated=True)]) is False
    assert hypothesis_holds(None, [_task()]) is None
    assert hypothesis_holds(steady, []) is None


def test_zero_budget_and_truncated_payloads():
    instant = aggregate(
        [_task(faulted=True, recovered=True, recovery_ms=0)],
        baseline_tokens=0,
        budget_ms=0,
    )
    assert instant.components["recovery"] == 1
    assert instant.token_overhead == 1
    late = aggregate(
        [_task(faulted=True, recovered=True, recovery_ms=5)],
        baseline_tokens=10,
        budget_ms=0,
    )
    assert late.components["recovery"] == 0

    unparsed = score_task(
        Task("ticket", "p", "OPEN", ("OPEN",)),
        "OPEN",
        [Event(1, 0, "get_ticket", "result", {"value": '{"id": "T-9", "stat'})],
        finished_ms=0,
        canaries=(),
        error=None,
    )
    assert unparsed.hallucinated
