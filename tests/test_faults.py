from __future__ import annotations

import json
import random

import pytest

from afi.clock import ManualClock, SystemClock
from afi.errors import FaultError, ToolHTTPError, ToolTimeout
from afi.faults import FAULTS, FaultContext, Invocation, apply_fault, fault_info
from afi.injector import Injector
from afi.scenario import FaultSpec, Scenario


def _invoke(fn, *, target="lookup", index=1, args=(), kwargs=None):
    return Invocation(target, index, args, kwargs or {}, fn)


def _ctx(cache=None):
    return FaultContext(ManualClock(), {} if cache is None else cache)


def test_catalog_matches_handlers():
    kinds = [info.kind for info in FAULTS]
    assert kinds == [
        "timeout",
        "latency",
        "http_error",
        "truncated_json",
        "malformed_json",
        "schema_drift",
        "stale_data",
        "partial_results",
        "prompt_injection",
    ]
    with pytest.raises(FaultError):
        fault_info("nope")


def test_timeout_does_not_run_the_call():
    clock = ManualClock()
    ctx = FaultContext(clock, {})
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return {"capital": "Zephyria"}

    outcome = apply_fault("timeout", _invoke(fn), {"delay_ms": 15}, ctx)
    assert calls["n"] == 0
    assert clock.now_ms() == 15
    assert isinstance(outcome.error, ToolTimeout)
    assert outcome.error.delay_ms == 15


def test_latency_runs_after_the_sleep():
    clock = ManualClock()
    ctx = FaultContext(clock, {})

    def fn():
        return {"capital": "Zephyria"}

    outcome = apply_fault("latency", _invoke(fn), {"delay_ms": 8}, ctx)
    assert outcome.error is None
    assert outcome.value == {"capital": "Zephyria"}
    assert outcome.ran is True
    assert clock.now_ms() == 8


def test_http_error_defaults_to_500():
    outcome = apply_fault("http_error", _invoke(lambda: None), {}, _ctx())
    assert isinstance(outcome.error, ToolHTTPError)
    assert outcome.error.status == 500
    custom = apply_fault(
        "http_error", _invoke(lambda: None), {"status": 429, "body": "slow"}, _ctx()
    )
    assert isinstance(custom.error, ToolHTTPError)
    assert custom.error.status == 429
    assert custom.error.body == "slow"


def test_json_faults_break_a_serialized_payload():
    def fn():
        return {"id": "T-9", "status": "OPEN"}

    truncated = apply_fault("truncated_json", _invoke(fn), {"keep_chars": 8}, _ctx())
    assert truncated.value == '{"id": "'
    with pytest.raises(json.JSONDecodeError):
        json.loads(truncated.value)

    broken = apply_fault("malformed_json", _invoke(fn), {}, _ctx())
    assert "<bad" in broken.value
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken.value)


def test_schema_drift_renames_and_round_trips_strings():
    def fn():
        return json.dumps({"capital": "Zephyria", "extra": 1})

    outcome = apply_fault(
        "schema_drift",
        _invoke(fn),
        {"rename": {"capital": "capital_city"}, "drop": ["extra"]},
        _ctx(),
    )
    parsed = json.loads(outcome.value)
    assert parsed == {"capital_city": "Zephyria"}
    skipped = apply_fault("schema_drift", _invoke(lambda: [1, 2]), {}, _ctx())
    assert skipped.detail["skipped"] == "not an object"


def test_stale_replay_then_override():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return {"n": calls["n"]}

    cache: dict[str, object] = {}
    ctx = FaultContext(ManualClock(), cache)
    first = apply_fault("stale_data", _invoke(fn), {}, ctx)
    assert first.detail["mode"] == "seed"
    assert first.value == {"n": 1}
    cache["lookup"] = {"n": 1}
    second = apply_fault("stale_data", _invoke(fn, index=2), {}, ctx)
    assert second.detail["mode"] == "replay"
    assert second.value == {"n": 1}
    assert second.ran is False
    assert calls["n"] == 1

    overridden = apply_fault(
        "stale_data",
        _invoke(fn, index=3),
        {"overrides": {"price": 80, "as_of_ms": -10000}},
        _ctx(),
    )
    assert overridden.value["price"] == 80
    assert overridden.produced["n"] == 2
    skipped = apply_fault("stale_data", _invoke(lambda: "nope"), {"overrides": {"a": 1}}, _ctx())
    assert skipped.detail["skipped"] == "not an object"


def test_partial_results_rotate_with_the_call_index():
    def fn():
        return {"total": 3, "items": [{"sku": "A"}, {"sku": "B"}, {"sku": "C"}]}

    first = apply_fault("partial_results", _invoke(fn, index=1), {"keep": 1}, _ctx())
    second = apply_fault("partial_results", _invoke(fn, index=2), {"keep": 1}, _ctx())
    third = apply_fault("partial_results", _invoke(fn, index=3), {"keep": 1}, _ctx())
    assert [row["sku"] for row in first.value["items"]] == ["A"]
    assert [row["sku"] for row in second.value["items"]] == ["B"]
    assert [row["sku"] for row in third.value["items"]] == ["C"]
    assert first.value["total"] == 3
    untouched = apply_fault("partial_results", _invoke(fn), {"keep": 5}, _ctx())
    assert len(untouched.value["items"]) == 3
    skipped = apply_fault("partial_results", _invoke(lambda: {"total": 1}), {}, _ctx())
    assert skipped.detail["skipped"] == "no list field"


def test_prompt_injection_channel_and_value():
    def fn():
        return {"capital": "Zephyria"}

    channel = apply_fault(
        "prompt_injection",
        _invoke(fn),
        {"canary": "CANARY-7", "mode": "channel"},
        _ctx(),
    )
    assert channel.value["capital"] == "Zephyria"
    assert "CANARY-7" in channel.value["notice"]
    assert "Ignore previous instructions" in channel.value["notice"]

    poisoned = apply_fault(
        "prompt_injection",
        _invoke(fn),
        {"canary": "ZEBRA-1", "mode": "value", "field": "capital"},
        _ctx(),
    )
    assert "ZEBRA-1" in poisoned.value["capital"]
    appended = apply_fault("prompt_injection", _invoke(lambda: "plain"), {}, _ctx())
    assert "CANARY-7" in appended.value


def test_unknown_fault_raises():
    with pytest.raises(FaultError):
        apply_fault("nope", _invoke(lambda: None), {}, _ctx())


def test_clocks_reject_negative_sleeps():
    with pytest.raises(ValueError):
        ManualClock().sleep_ms(-1)
    with pytest.raises(ValueError):
        SystemClock().sleep_ms(-1)
    clock = SystemClock()
    before = clock.now_ms()
    clock.sleep_ms(0)
    assert clock.now_ms() >= before
    manual = ManualClock(4)
    manual.sleep_ms(0)
    assert manual.now_ms() == 4


def _scenario(*faults: FaultSpec, radius: set[str] | None = None, seed: int = 0) -> Scenario:
    names = {fault.target for fault in faults if fault.target != "*"}
    return Scenario(
        name="s",
        seed=seed,
        description="",
        faults=tuple(faults),
        blast_radius=frozenset(radius if radius is not None else names),
        steady_state=None,
        tasks=None,
    )


def test_injector_is_deterministic_and_honors_after_call():
    fault = FaultSpec("timeout", "lookup", 1, 2, {"delay_ms": 4})
    scenario = _scenario(fault)

    def run():
        clock = ManualClock()
        injector = Injector(scenario, clock=clock, rng=random.Random(scenario.seed))
        tool = injector.tool("lookup", lambda: {"ok": True})
        assert tool() == {"ok": True}
        assert tool() == {"ok": True}
        with pytest.raises(ToolTimeout):
            tool()
        return [(event.phase, event.t_ms, event.detail.get("fault")) for event in injector.trace]

    assert run() == run()


def test_first_armed_fault_wins_and_star_stays_inside_the_radius():
    faults = (
        FaultSpec("timeout", "lookup", 1, 0, {"delay_ms": 3}),
        FaultSpec("latency", "lookup", 1, 0, {"delay_ms": 50}),
    )
    scenario = _scenario(*faults)
    clock = ManualClock()
    injector = Injector(scenario, clock=clock, rng=random.Random(0))
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return {"ok": True}

    with pytest.raises(ToolTimeout):
        injector.tool("lookup", fn)()
    assert calls["n"] == 0
    assert clock.now_ms() == 3

    star = _scenario(FaultSpec("timeout", "*", 1, 0, {"delay_ms": 1}), radius={"lookup"})
    injector = Injector(star, clock=ManualClock(), rng=random.Random(0))
    with pytest.raises(ToolTimeout):
        injector.tool("lookup", fn)()
    assert injector.tool("health", fn)() == {"ok": True}


def test_tool_exception_is_traced():
    scenario = _scenario()
    injector = Injector(scenario, clock=ManualClock(), rng=random.Random(0))

    def fn():
        raise KeyError("missing")

    with pytest.raises(KeyError):
        injector.tool("lookup", fn)()
    assert injector.trace[-1].phase == "error"
    assert injector.trace[-1].detail["type"] == "KeyError"


def test_llm_timeout_is_injected():
    from afi.models import LLMResponse

    scenario = _scenario(FaultSpec("timeout", "llm", 1, 0, {"delay_ms": 2}), radius={"llm"})

    class Client:
        def complete(self, messages, tools):
            return LLMResponse("hi", (), 3, 1)

    injector = Injector(scenario, clock=ManualClock(), rng=random.Random(0))
    wrapped = injector.llm(Client())
    with pytest.raises(ToolTimeout):
        wrapped.complete((), ())
    assert injector.clock.now_ms() == 2
