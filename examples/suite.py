from __future__ import annotations

import json
from typing import Any

from afi.clock import Clock
from afi.models import ToolSpec, World
from afi.tasks import Task

# The suite is deliberately small and fake. The capitals are not real, so a
# correct answer has to come from the tool. A model that "knows" geography
# cannot stumble into Zephyria.

TASKS: tuple[Task, ...] = (
    Task(
        id="capital",
        prompt='Look up the capital of "zephyr".',
        expect="Zephyria",
        evidence=("Zephyria",),
        description="Country lookup.",
    ),
    Task(
        id="atlas",
        prompt='Look up the capital of "atlas".',
        expect="Atlante",
        evidence=("Atlante",),
        description="Second country, used by the control run.",
    ),
    Task(
        id="ticket",
        prompt='Read ticket "T-9" and report its status.',
        expect="OPEN",
        evidence=("OPEN",),
        description="Status comes back as a JSON string.",
    ),
    Task(
        id="stock",
        prompt='Sum on-hand quantities for warehouse "north".',
        expect="16",
        evidence=("A-1", "B-2", "C-3"),
        description="Three stock rows, total 16.",
    ),
    Task(
        id="price",
        prompt='Report the fresh spot price for "acme".',
        expect="120",
        evidence=("120",),
        description="Quote stamped with as_of_ms.",
    ),
    Task(
        id="health",
        prompt="Run the healthcheck.",
        expect="ok",
        evidence=("ok",),
        description="Dependency outside the usual blast radius.",
    ),
)


def build_world(clock: Clock) -> World:
    def lookup_country(name: str) -> dict[str, str]:
        capitals = {"zephyr": "Zephyria", "atlas": "Atlante"}
        try:
            capital = capitals[name.lower()]
        except KeyError as exc:
            raise KeyError(f"unknown country {name}") from exc
        return {"name": name.lower(), "capital": capital}

    def get_ticket(ticket_id: str) -> str:
        return json.dumps({"id": ticket_id, "status": "OPEN"})

    def list_stock(warehouse: str) -> dict[str, Any]:
        items = [
            {"sku": "A-1", "qty": 2},
            {"sku": "B-2", "qty": 5},
            {"sku": "C-3", "qty": 9},
        ]
        return {"warehouse": warehouse, "total": len(items), "items": items}

    def get_price(symbol: str) -> dict[str, Any]:
        return {"symbol": symbol, "price": 120, "as_of_ms": int(clock.now_ms())}

    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    tools: dict[str, Any] = {
        "lookup_country": lookup_country,
        "get_ticket": get_ticket,
        "list_stock": list_stock,
        "get_price": get_price,
        "healthcheck": healthcheck,
    }
    specs = (
        ToolSpec(
            "lookup_country",
            "Look up a country's capital by name.",
            _object({"name": {"type": "string"}}, ["name"]),
        ),
        ToolSpec(
            "get_ticket",
            "Read a ticket and return its JSON document.",
            _object({"ticket_id": {"type": "string"}}, ["ticket_id"]),
        ),
        ToolSpec(
            "list_stock",
            "List on-hand units for a warehouse.",
            _object({"warehouse": {"type": "string"}}, ["warehouse"]),
        ),
        ToolSpec(
            "get_price",
            "Read the current spot price for a symbol.",
            _object({"symbol": {"type": "string"}}, ["symbol"]),
        ),
        ToolSpec("healthcheck", "Report whether the process is up.", _object({}, [])),
    )
    return World(tools=tools, specs=specs)


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}
