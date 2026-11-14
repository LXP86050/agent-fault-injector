from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.text import Text

from afi.scoring import WEIGHTS, TaskScore


@dataclass
class ScenarioScore:
    name: str
    description: str
    seed: int
    fault_labels: list[str]
    fault_count: int
    score: float
    components: dict[str, float]
    hypothesis_held: bool | None
    success_rate: float
    hallucination_rate: float
    safety_failure_rate: float
    mean_retries: float
    mean_recovery_ms: float | None
    unrecovered: int
    tokens: int
    baseline_tokens: int
    token_overhead: float
    tool_calls: int
    faults_fired: int
    recovery_budget_ms: float
    tasks: list[TaskScore]


@dataclass
class ExperimentReport:
    version: int
    library_version: str
    agent: str
    clock: str
    overall_score: float
    fault_scenario_count: int
    control_excluded: bool
    weights: dict[str, float]
    scenarios: list[ScenarioScore]


def dumps(report: ExperimentReport) -> str:
    return json.dumps(_report_dict(report), indent=2) + "\n"


def loads(text: str) -> ExperimentReport:
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("report must be a JSON object")
    version = raw.get("version")
    if version != 1:
        raise ValueError(f"unsupported report version {version!r}")
    return _report_from(raw)


def render_terminal(report: ExperimentReport, console: Console | None = None) -> None:
    out = console or Console(highlight=False)
    out.print("[bold]agent-fault-injector[/]")
    out.print(f"agent   {report.agent}")
    out.print(f"clock   {report.clock}")
    scope = f"mean of {report.fault_scenario_count} fault scenario"
    if report.fault_scenario_count != 1:
        scope += "s"
    if report.control_excluded:
        scope += ", control excluded"
    out.print(f"overall {report.overall_score:.3f}   {scope}")
    out.print()
    out.print(
        f"{'scenario':<18}{'score':>6}  {'succ':>4}  {'hall':>4}  "
        f"{'safe':>4}  {'rtry':>4}  {'recovery':<12}  hyp",
        style="bold",
    )
    for scenario in report.scenarios:
        out.print(_summary_line(scenario))

    for scenario in report.scenarios:
        faults = ", ".join(scenario.fault_labels) if scenario.fault_labels else "none"
        out.print()
        out.print(Text(scenario.name, style="bold"), f"  seed {scenario.seed}  faults {faults}")
        if scenario.description:
            out.print(Text(scenario.description, style="dim"))
        for task in scenario.tasks:
            out.print(
                f"  {task.task_id:<10} {_task_mark(task):<12} "
                f"retries {task.retries:<2} recovery {_task_recovery(task):<14} "
                f"tools {task.tool_calls}"
            )
    out.print()
    out.print(_formula(report.weights))
    out.print(
        "Recovery is experiment-clock time from the first fault in a task to a "
        "grounded answer. Unrecovered tasks are left out of the mean and score 0."
    )


def render_html(report: ExperimentReport) -> str:
    rows = "\n".join(_scenario_row(scenario) for scenario in report.scenarios)
    sections = "\n".join(_scenario_section(scenario) for scenario in report.scenarios)
    scope = f"Mean of {report.fault_scenario_count} fault scenario"
    if report.fault_scenario_count != 1:
        scope += "s"
    if report.control_excluded:
        scope += ". The fault-free control is shown and left out of the mean."
    title = html.escape(f"Resilience report — {report.agent}")
    meta = (
        f"agent {html.escape(report.agent)} · clock {html.escape(report.clock)} · "
        f"afi {html.escape(report.library_version)}"
    )
    note = (
        "Recovery is experiment-clock time from the first fault in a task "
        "to a grounded answer. Unrecovered tasks are left out of the mean and score 0. "
        "Token overhead is LLM tokens against a no-fault pass of the same tasks. "
        "Tool retries show up in the retry column, not the token bill."
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
:root {{
  --ink: #1f1a17;
  --muted: #5e564e;
  --paper: #f4f1ea;
  --card: #fbf9f5;
  --line: #d9d2c5;
  --accent: #8f2d2a;
  --ok: #1f6b4a;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 15px/1.45 Georgia, "Iowan Old Style", Palatino, serif;
}}
main {{ max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }}
h1 {{ font-size: 28px; font-weight: 600; margin: 0 0 8px; }}
h2 {{ font-size: 18px; font-weight: 600; margin: 28px 0 8px; }}
p {{ margin: 0 0 10px; }}
.meta {{ color: var(--muted); margin-bottom: 18px; }}
.overall {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 22px;
  margin: 8px 0 18px;
}}
table {{ width: 100%; border-collapse: collapse; background: var(--card); }}
th, td {{
  text-align: left;
  padding: 7px 8px;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
  font-variant-numeric: tabular-nums;
}}
th {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.02em;
}}
.num, .bar-cell {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
}}
.bar {{
  height: 8px;
  background: var(--line);
  width: 88px;
  display: inline-block;
  vertical-align: middle;
  margin-right: 6px;
}}
.bar > span {{ display: block; height: 100%; background: var(--accent); }}
.pass {{ color: var(--ok); }}
.fail {{ color: var(--accent); }}
.na {{ color: var(--muted); }}
section {{
  margin-top: 22px;
  padding-top: 8px;
  border-top: 1px solid var(--line);
}}
.desc {{ color: var(--muted); }}
footer {{ margin-top: 28px; color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
<main>
  <h1>Resilience report</h1>
  <p class="meta">{meta}</p>
  <p class="overall">{report.overall_score:.3f}</p>
  <p>{html.escape(scope)}</p>
  <table>
    <thead>
      <tr>
        <th>scenario</th>
        <th>score</th>
        <th>success</th>
        <th>halluc.</th>
        <th>safety</th>
        <th>retries</th>
        <th>recovery</th>
        <th>overhead</th>
        <th>hypothesis</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  {sections}
  <footer>
    <p>{html.escape(_formula(report.weights))}</p>
    <p>{note}</p>
  </footer>
</main>
</body>
</html>
"""


def _scenario_row(scenario: ScenarioScore) -> str:
    width = max(0.0, min(100.0, scenario.score * 100))
    return (
        "<tr>"
        f"<td>{html.escape(scenario.name)}</td>"
        f'<td class="bar-cell"><span class="bar"><span style="width: {width:.1f}%"></span></span>'
        f"{scenario.score:.3f}</td>"
        f'<td class="num">{scenario.success_rate:.2f}</td>'
        f'<td class="num">{scenario.hallucination_rate:.2f}</td>'
        f'<td class="num">{scenario.safety_failure_rate:.2f}</td>'
        f'<td class="num">{scenario.mean_retries:.2f}</td>'
        f'<td class="num">{html.escape(_recovery_plain(scenario))}</td>'
        f'<td class="num">{scenario.token_overhead:.2f}</td>'
        f"<td>{_hypothesis_html(scenario.hypothesis_held)}</td>"
        "</tr>"
    )


def _scenario_section(scenario: ScenarioScore) -> str:
    faults = ", ".join(scenario.fault_labels) if scenario.fault_labels else "none"
    task_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(task.task_id)}</td>"
        f"<td>{html.escape(_task_mark(task))}</td>"
        f"<td>{html.escape(task.answer) or '—'}</td>"
        f'<td class="num">{task.retries}</td>'
        f'<td class="num">{html.escape(_task_recovery(task))}</td>'
        f'<td class="num">{task.tool_calls}</td>'
        f'<td class="num">{task.prompt_tokens + task.completion_tokens}</td>'
        "</tr>"
        for task in scenario.tasks
    )
    description = ""
    if scenario.description:
        description = f'<p class="desc">{html.escape(scenario.description)}</p>'
    return (
        "<section>"
        f"<h2>{html.escape(scenario.name)}</h2>"
        f"{description}"
        f"<p>seed {scenario.seed} · faults {html.escape(faults)} · "
        f"budget {scenario.recovery_budget_ms:.0f}ms · "
        f"fired {scenario.faults_fired} · unrecovered {scenario.unrecovered}</p>"
        "<table><thead><tr>"
        "<th>task</th><th>result</th><th>answer</th><th>retries</th>"
        "<th>recovery</th><th>tools</th><th>tokens</th>"
        "</tr></thead><tbody>"
        f"{task_rows}</tbody></table></section>"
    )


def _report_dict(report: ExperimentReport) -> dict[str, Any]:
    return {
        "version": report.version,
        "library_version": report.library_version,
        "agent": report.agent,
        "clock": report.clock,
        "overall_score": report.overall_score,
        "fault_scenario_count": report.fault_scenario_count,
        "control_excluded": report.control_excluded,
        "weights": report.weights,
        "scenarios": [_scenario_dict(scenario) for scenario in report.scenarios],
    }


def _scenario_dict(scenario: ScenarioScore) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "description": scenario.description,
        "seed": scenario.seed,
        "fault_labels": scenario.fault_labels,
        "fault_count": scenario.fault_count,
        "score": scenario.score,
        "components": scenario.components,
        "hypothesis_held": scenario.hypothesis_held,
        "success_rate": scenario.success_rate,
        "hallucination_rate": scenario.hallucination_rate,
        "safety_failure_rate": scenario.safety_failure_rate,
        "mean_retries": scenario.mean_retries,
        "mean_recovery_ms": scenario.mean_recovery_ms,
        "unrecovered": scenario.unrecovered,
        "tokens": scenario.tokens,
        "baseline_tokens": scenario.baseline_tokens,
        "token_overhead": scenario.token_overhead,
        "tool_calls": scenario.tool_calls,
        "faults_fired": scenario.faults_fired,
        "recovery_budget_ms": scenario.recovery_budget_ms,
        "tasks": [_task_dict(task) for task in scenario.tasks],
    }


def _task_dict(task: TaskScore) -> dict[str, Any]:
    return {
        "id": task.task_id,
        "answer": task.answer,
        "success": task.success,
        "hallucinated": task.hallucinated,
        "safety_failure": task.safety_failure,
        "retries": task.retries,
        "recovery_ms": task.recovery_ms,
        "recovered": task.recovered,
        "faulted": task.faulted,
        "prompt_tokens": task.prompt_tokens,
        "completion_tokens": task.completion_tokens,
        "tool_calls": task.tool_calls,
        "faults_fired": task.faults_fired,
        "error": task.error,
    }


def _report_from(raw: dict[str, Any]) -> ExperimentReport:
    scenarios_raw = raw.get("scenarios")
    if not isinstance(scenarios_raw, list):
        raise ValueError("report.scenarios must be a list")
    weights = raw.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("report.weights must be an object")
    return ExperimentReport(
        version=1,
        library_version=str(raw.get("library_version", "")),
        agent=str(raw.get("agent", "")),
        clock=str(raw.get("clock", "")),
        overall_score=float(raw.get("overall_score", 0)),
        fault_scenario_count=int(raw.get("fault_scenario_count", 0)),
        control_excluded=bool(raw.get("control_excluded", False)),
        weights={str(key): float(value) for key, value in weights.items()},
        scenarios=[_scenario_from(item) for item in scenarios_raw],
    )


def _scenario_from(raw: object) -> ScenarioScore:
    if not isinstance(raw, dict):
        raise ValueError("scenario must be an object")
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list):
        raise ValueError("scenario.tasks must be a list")
    components = raw.get("components")
    if not isinstance(components, dict):
        raise ValueError("scenario.components must be an object")
    labels = raw.get("fault_labels")
    if not isinstance(labels, list):
        raise ValueError("scenario.fault_labels must be a list")
    mean_recovery = raw.get("mean_recovery_ms")
    return ScenarioScore(
        name=str(raw.get("name", "")),
        description=str(raw.get("description", "")),
        seed=int(raw.get("seed", 0)),
        fault_labels=[str(item) for item in labels],
        fault_count=int(raw.get("fault_count", 0)),
        score=float(raw.get("score", 0)),
        components={str(key): float(value) for key, value in components.items()},
        hypothesis_held=_optional_bool(raw.get("hypothesis_held")),
        success_rate=float(raw.get("success_rate", 0)),
        hallucination_rate=float(raw.get("hallucination_rate", 0)),
        safety_failure_rate=float(raw.get("safety_failure_rate", 0)),
        mean_retries=float(raw.get("mean_retries", 0)),
        mean_recovery_ms=None if mean_recovery is None else float(mean_recovery),
        unrecovered=int(raw.get("unrecovered", 0)),
        tokens=int(raw.get("tokens", 0)),
        baseline_tokens=int(raw.get("baseline_tokens", 0)),
        token_overhead=float(raw.get("token_overhead", 0)),
        tool_calls=int(raw.get("tool_calls", 0)),
        faults_fired=int(raw.get("faults_fired", 0)),
        recovery_budget_ms=float(raw.get("recovery_budget_ms", 0)),
        tasks=[_task_from(item) for item in tasks_raw],
    )


def _task_from(raw: object) -> TaskScore:
    if not isinstance(raw, dict):
        raise ValueError("task must be an object")
    recovery = raw.get("recovery_ms")
    error = raw.get("error")
    return TaskScore(
        task_id=str(raw.get("id", "")),
        answer=str(raw.get("answer", "")),
        success=bool(raw.get("success", False)),
        hallucinated=bool(raw.get("hallucinated", False)),
        safety_failure=bool(raw.get("safety_failure", False)),
        retries=int(raw.get("retries", 0)),
        recovery_ms=None if recovery is None else float(recovery),
        recovered=bool(raw.get("recovered", False)),
        faulted=bool(raw.get("faulted", False)),
        prompt_tokens=int(raw.get("prompt_tokens", 0)),
        completion_tokens=int(raw.get("completion_tokens", 0)),
        tool_calls=int(raw.get("tool_calls", 0)),
        faults_fired=int(raw.get("faults_fired", 0)),
        error=None if error is None else str(error),
    )


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError("hypothesis_held must be a bool or null")


def _formula(weights: dict[str, float]) -> str:
    order = ("success", "safety", "honesty", "recovery", "efficiency")
    used = weights or WEIGHTS
    parts = [f"{used[name]:.2f}*{name}" for name in order if name in used]
    return "score = " + " + ".join(parts)


def _hypothesis_text(held: bool | None) -> Text:
    if held is True:
        return Text("pass", style="green")
    if held is False:
        return Text("fail", style="red")
    return Text("n/a")


def _hypothesis_html(held: bool | None) -> str:
    if held is True:
        return '<span class="pass">pass</span>'
    if held is False:
        return '<span class="fail">fail</span>'
    return '<span class="na">n/a</span>'


def _summary_line(scenario: ScenarioScore) -> Text:
    line = Text()
    line.append(f"{scenario.name:<18}")
    line.append(
        f"{scenario.score:6.3f}  {scenario.success_rate:4.2f}  "
        f"{scenario.hallucination_rate:4.2f}  {scenario.safety_failure_rate:4.2f}  "
        f"{scenario.mean_retries:4.2f}  {_short_recovery(scenario):<12}  "
    )
    line.append_text(_hypothesis_text(scenario.hypothesis_held))
    return line


def _short_recovery(scenario: ScenarioScore) -> str:
    if scenario.fault_count == 0 or scenario.mean_recovery_ms is None:
        if scenario.unrecovered:
            return "unrecovered"
        return "n/a"
    return f"{scenario.mean_recovery_ms:.0f}ms"


def _recovery_plain(scenario: ScenarioScore) -> str:
    if scenario.mean_recovery_ms is None:
        if scenario.unrecovered:
            return "unrecovered"
        return "n/a"
    text = f"{scenario.mean_recovery_ms:.0f}ms"
    if scenario.unrecovered:
        text += f" ({scenario.unrecovered} unrecovered)"
    return text


def _task_mark(task: TaskScore) -> str:
    if task.safety_failure:
        return "unsafe"
    if task.hallucinated:
        return "hallucinated"
    if task.success:
        return "ok"
    return "fail"


def _task_recovery(task: TaskScore) -> str:
    if not task.faulted:
        return "n/a"
    if not task.recovered or task.recovery_ms is None:
        return "unrecovered"
    return f"{task.recovery_ms:.0f}ms"
