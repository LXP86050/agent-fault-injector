from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from afi.clock import Clock
from afi.faults import FAULTS
from afi.models import Agent, Toolkit, World
from afi.report import dumps, loads, render_html, render_terminal
from afi.runner import run_experiment
from afi.scenario import ScenarioError, load_scenarios
from afi.tasks import Task


class CLIError(Exception):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="afi")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run scenarios and score the agent")
    run.add_argument("scenarios", type=Path, help="scenario file or directory")
    run.add_argument("--agent", required=True, help="module:factory, called with a Toolkit")
    run.add_argument(
        "--tasks",
        default="examples.suite:TASKS",
        help="module:attr sequence of Tasks",
    )
    run.add_argument(
        "--world",
        default="examples.suite:build_world",
        help="module:factory called with the experiment clock",
    )
    run.add_argument("--out", type=Path, help="write the JSON report here")
    run.add_argument("--html", type=Path, help="write the HTML report here")
    run.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit 1 when the overall score is below this value",
    )
    run.add_argument(
        "--realtime",
        action="store_true",
        help="sleep on the wall clock instead of the simulated experiment clock",
    )
    run.add_argument("--quiet", action="store_true", help="skip the terminal report")

    report = sub.add_parser("report", help="print a JSON report again")
    report.add_argument("path", type=Path)
    report.add_argument("--html", type=Path)

    sub.add_parser("list-faults", help="list fault kinds")

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "list-faults":
            return _list_faults()
        if args.command == "report":
            return _report(args.path, args.html)
        return _run(args)
    except (CLIError, ScenarioError, ValueError, TypeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.scenarios)
    agent = _callable(args.agent)
    tasks = _tasks(args.tasks)
    world = cast(Callable[[Clock], World], _callable(args.world))
    report = run_experiment(
        scenarios,
        cast(Callable[[Toolkit], Agent], agent),
        tasks,
        world,
        agent_name=args.agent,
        realtime=args.realtime,
    )
    if args.out is not None:
        args.out.write_text(dumps(report), encoding="utf-8")
    if args.html is not None:
        args.html.write_text(render_html(report), encoding="utf-8")
    if not args.quiet:
        render_terminal(report)
    if args.fail_under is not None and report.overall_score < args.fail_under:
        print(
            f"overall {report.overall_score:.3f} is below {args.fail_under:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


def _report(path: Path, html: Path | None) -> int:
    report = loads(path.read_text(encoding="utf-8"))
    render_terminal(report)
    if html is not None:
        html.write_text(render_html(report), encoding="utf-8")
    return 0


def _list_faults() -> int:
    width = max(len(info.kind) for info in FAULTS)
    for info in FAULTS:
        print(f"{info.kind:<{width}}  {info.summary}")
    return 0


def _load(spec: str) -> Any:
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise CLIError(f"expected module:attr, got {spec!r}")
    root = str(Path.cwd())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise CLIError(f"cannot import {module_name}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise CLIError(f"{spec} was not found") from exc


def _callable(spec: str) -> Callable[..., Any]:
    obj = _load(spec)
    if not callable(obj):
        raise CLIError(f"{spec} is not callable")
    return cast(Callable[..., Any], obj)


def _tasks(spec: str) -> list[Task]:
    obj = _load(spec)
    if isinstance(obj, Task):
        items: Sequence[object] = (obj,)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        items = obj
    else:
        raise CLIError(f"{spec} must be a sequence of Task")
    tasks = list(items)
    if not tasks or not all(isinstance(item, Task) for item in tasks):
        raise CLIError(f"{spec} must be a sequence of Task")
    return cast(list[Task], tasks)
