from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from afi._version import __version__
from afi.clock import Clock, ManualClock, SystemClock
from afi.injector import Injector
from afi.mock_llm import MockLLM
from afi.models import Agent, LLMClient, Toolkit, World
from afi.report import ExperimentReport, ScenarioScore
from afi.scenario import FaultSpec, Scenario
from afi.scoring import WEIGHTS, TaskScore, aggregate, hypothesis_holds, score_task
from afi.tasks import Task


def run_experiment(
    scenarios: Sequence[Scenario],
    agent: Callable[[Toolkit], Agent],
    tasks: Sequence[Task],
    world: Callable[[Clock], World],
    *,
    agent_name: str,
    llm: LLMClient | None = None,
    realtime: bool = False,
) -> ExperimentReport:
    """Run ``agent`` on ``tasks`` under each scenario.

    A no-fault pass runs first and supplies the token baseline. That pass uses
    the same agent and the same client, so a remote model is called for the
    baseline as well as for the scenarios. Call indexes and circuit-breaker
    state reset between scenarios and stay live across tasks inside one.
    """

    if not scenarios:
        raise ValueError("no scenarios")
    index = _index_tasks(tasks)
    selected = {scenario.name: _select(scenario, index) for scenario in scenarios}
    client = llm if llm is not None else MockLLM()
    needed: list[Task] = []
    seen: set[str] = set()
    for scenario in scenarios:
        for task in selected[scenario.name]:
            if task.id not in seen:
                seen.add(task.id)
                needed.append(task)
    baseline = _baseline_tokens(
        needed,
        agent=agent,
        world=world,
        llm=client,
        realtime=realtime,
    )
    scored: list[ScenarioScore] = []
    for scenario in scenarios:
        scored.append(
            _run_scenario(
                scenario,
                selected[scenario.name],
                agent=agent,
                world=world,
                llm=client,
                realtime=realtime,
                baseline=baseline,
            )
        )
    faulted = [item for item in scored if item.fault_count > 0]
    controls = [item for item in scored if item.fault_count == 0]
    pool = faulted or scored
    overall = sum(item.score for item in pool) / len(pool)
    ordered = controls + faulted
    return ExperimentReport(
        version=1,
        library_version=__version__,
        agent=agent_name,
        clock="realtime" if realtime else "simulated",
        overall_score=overall,
        fault_scenario_count=len(faulted) if faulted else len(scored),
        control_excluded=bool(faulted and controls),
        weights=dict(WEIGHTS),
        scenarios=ordered,
    )


def _run_scenario(
    scenario: Scenario,
    tasks: Sequence[Task],
    *,
    agent: Callable[[Toolkit], Agent],
    world: Callable[[Clock], World],
    llm: LLMClient,
    realtime: bool,
    baseline: dict[str, int],
) -> ScenarioScore:
    injector, bound = _bind(scenario, world, agent, llm, realtime)
    task_scores: list[TaskScore] = []
    for task in tasks:
        injector.reset_trace()
        answer, error = _execute(bound, task)
        task_scores.append(
            score_task(
                task,
                answer,
                injector.trace,
                finished_ms=injector.clock.now_ms(),
                canaries=scenario.canaries(),
                error=error,
            )
        )
    numbers = aggregate(
        task_scores,
        baseline_tokens=sum(baseline[task.id] for task in tasks),
        budget_ms=scenario.recovery_budget_ms,
    )
    return ScenarioScore(
        name=scenario.name,
        description=scenario.description,
        seed=scenario.seed,
        fault_labels=[_fault_label(spec) for spec in scenario.faults],
        fault_count=len(scenario.faults),
        score=numbers.score,
        components=numbers.components,
        hypothesis_held=hypothesis_holds(scenario.steady_state, task_scores),
        success_rate=numbers.success_rate,
        hallucination_rate=numbers.hallucination_rate,
        safety_failure_rate=numbers.safety_failure_rate,
        mean_retries=numbers.mean_retries,
        mean_recovery_ms=numbers.mean_recovery_ms,
        unrecovered=numbers.unrecovered,
        tokens=numbers.tokens,
        baseline_tokens=numbers.baseline_tokens,
        token_overhead=numbers.token_overhead,
        tool_calls=numbers.tool_calls,
        faults_fired=numbers.faults_fired,
        recovery_budget_ms=scenario.recovery_budget_ms,
        tasks=task_scores,
    )


def _baseline_tokens(
    tasks: Sequence[Task],
    *,
    agent: Callable[[Toolkit], Agent],
    world: Callable[[Clock], World],
    llm: LLMClient,
    realtime: bool,
) -> dict[str, int]:
    scenario = Scenario(
        name="baseline",
        seed=0,
        description="",
        faults=(),
        blast_radius=frozenset(),
        steady_state=None,
        tasks=None,
        recovery_budget_ms=500,
    )
    injector, bound = _bind(scenario, world, agent, llm, realtime)
    totals: dict[str, int] = {}
    for task in tasks:
        injector.reset_trace()
        _execute(bound, task)
        score = score_task(
            task,
            "",
            injector.trace,
            finished_ms=injector.clock.now_ms(),
            canaries=(),
            error=None,
        )
        totals[task.id] = score.prompt_tokens + score.completion_tokens
    return totals


def _bind(
    scenario: Scenario,
    world: Callable[[Clock], World],
    agent: Callable[[Toolkit], Agent],
    llm: LLMClient,
    realtime: bool,
) -> tuple[Injector, Agent]:
    clock: Clock = SystemClock() if realtime else ManualClock()
    built = world(clock)
    if not isinstance(built, World):
        raise TypeError("world factory must return a World")
    injector = Injector(scenario, clock=clock, rng=random.Random(scenario.seed))
    tools = {name: injector.tool(name, fn) for name, fn in built.tools.items()}
    toolkit = Toolkit(
        tools=tools,
        specs=built.specs,
        llm=injector.llm(llm),
        clock=clock,
        seed=scenario.seed,
    )
    bound = agent(toolkit)
    if not callable(getattr(bound, "run", None)):
        raise TypeError("agent factory must return an object with run()")
    return injector, bound


def _execute(agent: Agent, task: Task) -> tuple[str, str | None]:
    try:
        return agent.run(task), None
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _index_tasks(tasks: Sequence[Task]) -> dict[str, Task]:
    index: dict[str, Task] = {}
    for task in tasks:
        if task.id in index:
            raise ValueError(f"duplicate task id {task.id}")
        index[task.id] = task
    if not index:
        raise ValueError("no tasks")
    return index


def _select(scenario: Scenario, index: dict[str, Task]) -> list[Task]:
    if scenario.tasks is None:
        return list(index.values())
    missing = [name for name in scenario.tasks if name not in index]
    if missing:
        raise ValueError(f"{scenario.name}: unknown task {', '.join(missing)}")
    chosen = [index[name] for name in scenario.tasks]
    if not chosen:
        raise ValueError(f"{scenario.name}: no tasks selected")
    return chosen


def _fault_label(spec: FaultSpec) -> str:
    if spec.kind == "http_error":
        return f"http_error:{spec.params.get('status', 500)}"
    return spec.kind
