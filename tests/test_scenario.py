from __future__ import annotations

from pathlib import Path

import pytest

from afi.scenario import ScenarioError, load_scenario, load_scenarios

ROOT = Path("examples/scenarios")

GOOD = """
name: demo
seed: 7
description: a fault
recovery_budget_ms: 250
tasks:
  - capital
steady_state:
  min_success_rate: 0.5
  max_safety_failures: 0
  max_hallucination_rate: 0
blast_radius:
  targets:
    - lookup
faults:
  - fault: timeout
    target: lookup
    probability: 0.5
    after_call: 2
    params:
      delay_ms: 10
"""


def _write(tmp_path: Path, text: str, name: str = "demo.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_a_scenario_and_the_shipped_directory():
    scenarios = load_scenarios(ROOT)
    names = [item.name for item in scenarios]
    assert names[0] == "latency" or "steady" in names
    assert len(names) == len(set(names))
    assert "timeouts" in names
    timeouts = next(item for item in scenarios if item.name == "timeouts")
    assert timeouts.seed == 7
    assert timeouts.faults[0].kind == "timeout"
    assert timeouts.faults[0].after_call == 0
    assert timeouts.blast_radius == frozenset({"lookup_country"})
    steady = next(item for item in scenarios if item.name == "steady")
    assert steady.faults == ()
    injection = next(item for item in scenarios if item.name == "prompt_injection")
    assert injection.canaries() == ("CANARY-7",)


def test_round_trip_fields(tmp_path: Path):
    scenario = load_scenario(_write(tmp_path, GOOD))
    assert scenario.name == "demo"
    assert scenario.recovery_budget_ms == 250
    assert scenario.tasks == ("capital",)
    assert scenario.steady_state is not None
    assert scenario.steady_state.min_success_rate == 0.5
    assert scenario.faults[0].after_call == 2


@pytest.mark.parametrize(
    "text",
    [
        "name: [1, 2]\n",
        "[]\n",
        "name: demo\nseed: true\nfaults: []\n",
        "name: demo\nseed: 1\nnope: 1\nfaults: []\n",
        "name: demo\nseed: 1\nrecovery_budget_ms: -1\nfaults: []\n",
        "name: ' '\nseed: 1\nfaults: []\n",
        "name: demo\nseed: 1\nblast_radius:\n  targets: []\nfaults:\n  - fault: timeout\n    target: lookup\n    probability: 1\n",
        "name: demo\nseed: 1\nblast_radius:\n  targets: [other]\nfaults:\n  - fault: timeout\n    target: lookup\n    probability: 1\n",
        "name: demo\nseed: 1\nblast_radius:\n  targets: [lookup]\nfaults:\n  - fault: nope\n    target: lookup\n    probability: 1\n",
        "name: demo\nseed: 1\nblast_radius:\n  targets: [lookup]\nfaults:\n  - fault: timeout\n    target: lookup\n    probability: 2\n",
        "name: demo\nseed: 1\nblast_radius:\n  targets: [lookup]\nfaults:\n  - fault: timeout\n    target: lookup\n    probability: 1\n    params:\n      nope: 1\n",
        "name: demo\nseed: 1\nblast_radius:\n  targets: [lookup]\nfaults:\n  - fault: http_error\n    target: lookup\n    probability: 1\n    params:\n      status: 99\n",
        "name: demo\nseed: 1\nblast_radius:\n  targets: [lookup]\nfaults:\n  - fault: prompt_injection\n    target: lookup\n    probability: 1\n    params:\n      mode: sideways\n",
        "name: demo\nseed: 1\nblast_radius:\n  targets: [lookup]\nfaults:\n  - fault: schema_drift\n    target: lookup\n    probability: 1\n    params:\n      rename:\n        capital: 1\n",
        "name: demo\nseed: 1\ntasks: [capital, capital]\nfaults: []\n",
        "name: demo\nseed: 1\nsteady_state:\n  max_safety_failures: -1\nfaults: []\n",
        ": [\n",
    ],
)
def test_rejects_bad_scenarios(tmp_path: Path, text: str):
    with pytest.raises(ScenarioError):
        load_scenario(_write(tmp_path, text))


def test_duplicate_names_and_empty_directory(tmp_path: Path):
    _write(tmp_path, "name: demo\nseed: 1\nfaults: []\n", "a.yaml")
    _write(tmp_path, "name: demo\nseed: 2\nfaults: []\n", "b.yml")
    with pytest.raises(ScenarioError, match="duplicate"):
        load_scenarios(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ScenarioError, match="no scenarios"):
        load_scenarios(empty)
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ScenarioError):
        load_scenario(missing)


def test_star_target_is_allowed_when_the_radius_is_set(tmp_path: Path):
    text = """
name: star
seed: 1
blast_radius:
  targets: [lookup]
faults:
  - fault: latency
    target: "*"
    probability: 1
    params:
      delay_ms: 0
"""
    scenario = load_scenario(_write(tmp_path, text))
    assert scenario.faults[0].target == "*"
    assert scenario.steady_state is None
    assert scenario.canaries() == ()
