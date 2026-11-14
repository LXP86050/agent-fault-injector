from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from afi.errors import FaultError
from afi.faults import DEFAULT_CANARY, FAULTS, fault_info

_FAULT_PARAMS = {info.kind: info.params for info in FAULTS}

_SCENARIO_KEYS = frozenset(
    {
        "name",
        "seed",
        "description",
        "faults",
        "blast_radius",
        "steady_state",
        "tasks",
        "recovery_budget_ms",
    }
)
_FAULT_KEYS = frozenset({"fault", "target", "probability", "after_call", "params"})
_STEADY_KEYS = frozenset({"min_success_rate", "max_safety_failures", "max_hallucination_rate"})
_RADIUS_KEYS = frozenset({"targets"})


class ScenarioError(ValueError):
    """The scenario file is not a valid experiment."""


@dataclass(frozen=True)
class FaultSpec:
    kind: str
    target: str
    probability: float
    after_call: int
    params: dict[str, Any]


@dataclass(frozen=True)
class SteadyState:
    """Hypothesis that should still hold while faults are firing.

    This is the chaos-experiment check, not a benchmark. ``min_success_rate``
    is the fraction of tasks that must still succeed. Safety failures are a
    count because one followed injection is already a broken run.
    """

    min_success_rate: float = 0.0
    max_safety_failures: int = 0
    max_hallucination_rate: float = 1.0


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    description: str
    faults: tuple[FaultSpec, ...]
    blast_radius: frozenset[str]
    steady_state: SteadyState | None
    tasks: tuple[str, ...] | None
    recovery_budget_ms: float = 500.0

    def canaries(self) -> tuple[str, ...]:
        found: list[str] = []
        for spec in self.faults:
            if spec.kind != "prompt_injection":
                continue
            found.append(str(spec.params.get("canary", DEFAULT_CANARY)))
        return tuple(dict.fromkeys(found))


def load_scenarios(path: Path) -> list[Scenario]:
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix in {".yaml", ".yml"} and p.is_file())
        if not files:
            raise ScenarioError(f"no scenarios in {path}")
        scenarios = [load_scenario(item) for item in files]
        names = [item.name for item in scenarios]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ScenarioError(f"duplicate scenario name: {', '.join(duplicates)}")
        return scenarios
    return [load_scenario(path)]


def load_scenario(path: Path) -> Scenario:
    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ScenarioError(f"{path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ScenarioError(f"{path}: scenario must be a mapping")
    try:
        return _scenario(loaded, where=str(path))
    except ScenarioError as exc:
        raise ScenarioError(f"{path}: {exc}") from exc


def _scenario(data: Mapping[str, Any], *, where: str) -> Scenario:
    _check_keys(data, _SCENARIO_KEYS, where)
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError("name must be a non-empty string")
    seed = _as_int(data.get("seed", 0), "seed")
    description = data.get("description", "")
    if not isinstance(description, str):
        raise ScenarioError("description must be a string")
    budget = data.get("recovery_budget_ms", 500)
    recovery_budget = _as_non_negative_number(budget, "recovery_budget_ms")
    faults = tuple(_fault(item) for item in _as_list(data.get("faults", []), "faults"))
    radius = _radius(data.get("blast_radius"), faults)
    steady = _steady(data.get("steady_state"))
    tasks = _tasks(data.get("tasks"))
    return Scenario(
        name=name.strip(),
        seed=seed,
        description=description.strip(),
        faults=faults,
        blast_radius=radius,
        steady_state=steady,
        tasks=tasks,
        recovery_budget_ms=recovery_budget,
    )


def _fault(value: object) -> FaultSpec:
    data = _as_mapping(value, "fault")
    _check_keys(data, _FAULT_KEYS, "fault")
    kind = data.get("fault")
    if not isinstance(kind, str):
        raise ScenarioError("fault must name a fault kind")
    try:
        fault_info(kind)
    except FaultError as exc:
        raise ScenarioError(str(exc)) from exc
    target = data.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ScenarioError(f"{kind}: target must be a non-empty string")
    probability = _as_unit(data.get("probability"), f"{kind}.probability")
    after_call = _as_int(data.get("after_call", 0), f"{kind}.after_call")
    if after_call < 0:
        raise ScenarioError(f"{kind}.after_call must be >= 0")
    params = _params(kind, data.get("params", {}))
    return FaultSpec(
        kind=kind,
        target=target.strip(),
        probability=probability,
        after_call=after_call,
        params=params,
    )


def _params(kind: str, value: object) -> dict[str, Any]:
    data = _as_mapping(value, f"{kind}.params")
    allowed = _FAULT_PARAMS[kind]
    extra = sorted(set(data) - allowed)
    if extra:
        raise ScenarioError(f"{kind}: unknown params: {', '.join(extra)}")
    if kind in {"timeout", "latency"} and "delay_ms" in data:
        _as_non_negative_number(data["delay_ms"], f"{kind}.delay_ms")
    if kind == "http_error" and "status" in data:
        status = _as_int(data["status"], "http_error.status")
        if not 100 <= status <= 599:
            raise ScenarioError("http_error.status must be an HTTP status code")
    if kind == "http_error" and "body" in data and not isinstance(data["body"], str):
        raise ScenarioError("http_error.body must be a string")
    if kind == "truncated_json" and "keep_chars" in data:
        keep = _as_int(data["keep_chars"], "truncated_json.keep_chars")
        if keep < 0:
            raise ScenarioError("truncated_json.keep_chars must be >= 0")
    if kind == "schema_drift":
        _schema_params(data)
    if kind == "partial_results" and "keep" in data:
        keep = _as_int(data["keep"], "partial_results.keep")
        if keep < 0:
            raise ScenarioError("partial_results.keep must be >= 0")
    list_field = data.get("list_field")
    if kind == "partial_results" and "list_field" in data and not isinstance(list_field, str):
        raise ScenarioError("partial_results.list_field must be a string")
    if kind == "prompt_injection":
        _injection_params(data)
    if kind == "stale_data" and "overrides" in data and not isinstance(data["overrides"], dict):
        raise ScenarioError("stale_data.overrides must be a mapping")
    return dict(data)


def _schema_params(data: Mapping[str, Any]) -> None:
    rename = data.get("rename", {})
    if not isinstance(rename, dict) or not all(
        isinstance(key, str) and isinstance(val, str) for key, val in rename.items()
    ):
        raise ScenarioError("schema_drift.rename must map strings to strings")
    drop = data.get("drop", [])
    if not isinstance(drop, list) or not all(isinstance(item, str) for item in drop):
        raise ScenarioError("schema_drift.drop must be a list of strings")


def _injection_params(data: Mapping[str, Any]) -> None:
    mode = data.get("mode", "channel")
    if mode not in {"channel", "value"}:
        raise ScenarioError("prompt_injection.mode must be 'channel' or 'value'")
    for key in ("canary", "text", "field"):
        if key in data and not isinstance(data[key], str):
            raise ScenarioError(f"prompt_injection.{key} must be a string")


def _radius(value: object, faults: tuple[FaultSpec, ...]) -> frozenset[str]:
    if value is None:
        if faults:
            raise ScenarioError("blast_radius is required when faults are declared")
        return frozenset()
    data = _as_mapping(value, "blast_radius")
    _check_keys(data, _RADIUS_KEYS, "blast_radius")
    targets = data.get("targets", [])
    names_ok = isinstance(targets, list) and all(
        isinstance(item, str) and item.strip() for item in targets
    )
    if not names_ok:
        raise ScenarioError("blast_radius.targets must be a list of names")
    radius = frozenset(item.strip() for item in targets)
    if faults and not radius:
        raise ScenarioError("blast_radius.targets must name at least one dependency")
    for spec in faults:
        if spec.target == "*":
            continue
        if spec.target not in radius:
            raise ScenarioError(
                f"{spec.kind} targets {spec.target}, which is outside the blast radius"
            )
    return radius


def _steady(value: object) -> SteadyState | None:
    if value is None:
        return None
    data = _as_mapping(value, "steady_state")
    _check_keys(data, _STEADY_KEYS, "steady_state")
    rate = _as_unit(data.get("min_success_rate", 0), "steady_state.min_success_rate")
    failures = _as_int(data.get("max_safety_failures", 0), "steady_state.max_safety_failures")
    if failures < 0:
        raise ScenarioError("steady_state.max_safety_failures must be >= 0")
    hallucinations = _as_unit(
        data.get("max_hallucination_rate", 1), "steady_state.max_hallucination_rate"
    )
    return SteadyState(
        min_success_rate=rate,
        max_safety_failures=failures,
        max_hallucination_rate=hallucinations,
    )


def _tasks(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = _as_list(value, "tasks")
    names: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ScenarioError("tasks must be a list of task ids")
        names.append(item.strip())
    if len(names) != len(set(names)):
        raise ScenarioError("tasks contains a duplicate id")
    return tuple(names)


def _check_keys(data: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    extra = sorted(set(data) - allowed)
    if extra:
        raise ScenarioError(f"unknown {where} field: {', '.join(extra)}")


def _as_mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScenarioError(f"{where} must be a mapping")
    return value


def _as_list(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise ScenarioError(f"{where} must be a list")
    return list(value)


def _as_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenarioError(f"{where} must be an integer")
    return value


def _as_non_negative_number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(f"{where} must be a number")
    number = float(value)
    if number < 0:
        raise ScenarioError(f"{where} must be >= 0")
    return number


def _as_unit(value: object, where: str) -> float:
    number = _as_non_negative_number(value, where)
    if number > 1:
        raise ScenarioError(f"{where} must be between 0 and 1")
    return number
