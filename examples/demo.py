"""Run the naive loop and the reference agent on the shipped scenarios."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(_ROOT / "src"), str(_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from afi.report import render_terminal  # noqa: E402
from afi.runner import run_experiment  # noqa: E402
from afi.scenario import load_scenarios  # noqa: E402
from examples.naive_agent import build as build_naive  # noqa: E402
from examples.ref_agent import build as build_ref  # noqa: E402
from examples.suite import TASKS, build_world  # noqa: E402


def main() -> int:
    scenarios = load_scenarios(_ROOT / "examples" / "scenarios")
    naive = run_experiment(
        scenarios,
        build_naive,
        TASKS,
        build_world,
        agent_name="examples.naive_agent:build",
    )
    ref = run_experiment(
        scenarios,
        build_ref,
        TASKS,
        build_world,
        agent_name="examples.ref_agent:build",
    )
    render_terminal(naive)
    print()
    render_terminal(ref)
    print()
    print(f"naive      {naive.overall_score:.3f}")
    print(f"reference  {ref.overall_score:.3f}")
    print(f"delta      {ref.overall_score - naive.overall_score:+.3f}")
    return 0 if ref.overall_score > naive.overall_score else 1


if __name__ == "__main__":
    raise SystemExit(main())
