"""Fixed synthetic complete-rotation scenarios for offline policy comparison."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from time import perf_counter

from .engine import schedule
from .validation import validate_plan


def _iso(value):
    return value.isoformat(timespec="seconds")


def build_scenarios():
    zone = timezone(timedelta(hours=8))
    start = datetime(2025, 5, 2, 23, 0, tzinfo=zone)
    flights = []
    for index in range(8):
        tail = f"B-SYN-{index + 1:02d}"
        first = start + timedelta(minutes=5 + index)
        legs = [
            (1, 0, "ZGGG", "ZBAA"), (2, 95, "ZBAA", "ZGGG"),
            (3, 190, "ZGGG", "ZBAA"),
        ]
        if index >= 4:
            legs.append((4, 285, "ZBAA", "ZGGG"))
        for leg, minutes, origin, destination in legs:
            departure = first + timedelta(minutes=minutes)
            arrival = departure + timedelta(minutes=60)
            flights.append({"flight_id": f"SYN-{index + 1:02d}-{leg}", "tail": tail,
                            "origin": origin, "destination": destination,
                            "planned_departure": _iso(departure), "planned_arrival": _iso(arrival),
                            "predicted_departure": _iso(departure), "predicted_arrival": _iso(arrival)})
    base = {"airport": "ZGGG", "horizon_start": _iso(start),
            "horizon_end": _iso(start + timedelta(hours=8)), "slot_minutes": 15,
            "departure_capacity": 2, "arrival_capacity": 2, "mtt_minutes": 30,
            "closures": []}
    tight = {**base, "departure_capacity": 1}
    closed = {**base, "closures": [{"kind": "arrival",
                                     "start": _iso(start + timedelta(hours=2, minutes=30)),
                                     "end": _iso(start + timedelta(hours=3))}]}
    short = {**base, "horizon_end": _iso(start + timedelta(hours=4))}
    return [{"name": name, "config": config, "flights": flights}
            for name, config in (("base", base), ("tight_departure", tight),
                                 ("arrival_closure", closed), ("short_horizon", short))]


def compare_policies(cases):
    if not cases:
        raise ValueError("scenario set must be nonempty")
    identity = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    results = []
    for case in cases:
        item = {"name": case["name"], "flight_count": len(case["flights"])}
        for policy in ("earliest", "rotation_urgency"):
            start = perf_counter()
            plan = schedule(case["flights"], case["config"], priority_policy=policy)
            duration = perf_counter() - start
            errors = validate_plan(case["flights"], case["config"], plan)
            delays = [assignment["delay_minutes"] for assignment in plan["assignments"]]
            item[policy] = {"validation_errors": errors, "status": plan["status"],
                            "assigned_count": len(plan["assignments"]),
                            "unassigned_count": len(plan["unassigned"]),
                            "total_delay_minutes": sum(delays),
                            "max_delay_minutes": max(delays, default=0),
                            "elapsed_seconds": duration}
        results.append(item)
    return {"scenario_origin": "fixed_synthetic_complete_rotations",
            "scenario_sha256": identity, "scenarios": results}
