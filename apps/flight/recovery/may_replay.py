"""Offline May flight-plan replay with explicitly hypothetical airport capacity."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from .benchmark import compare_policies
from training.baselines import HistoricalMedianBaseline
from training.prompt import reconstruct_times
from training.temporal_audit import audit_temporal_rows, eligible_training_rows
from training.zggg_dataset import verify_zggg_dataset


ZONE = ZoneInfo("Asia/Shanghai")


def _time(value):
    timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("flight time must include timezone")
    return timestamp.astimezone(ZONE)


def _scheduler_time(value):
    timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("scheduler time must include timezone")
    return timestamp.isoformat(timespec="seconds")


def select_rotations(rows, date, *, max_tails=8):
    if not 1 <= max_tails <= 100:
        raise ValueError("invalid tail limit")
    day = datetime.fromisoformat(date).date()
    groups = defaultdict(list)
    for row in rows:
        feature = row["features"]
        departure = _time(feature["planned_off_block"])
        arrival = _time(feature["planned_on_block"])
        if departure.date() == day and arrival.date() == day and arrival > departure:
            groups[feature["tail_number"]].append(row)
    candidates = []
    for tail in sorted(groups):
        current, runs = [], []
        for row in sorted(groups[tail], key=lambda item: (item["features"]["planned_off_block"], item["record_id"])):
            feature = row["features"]
            if current:
                previous = current[-1]["features"]
                gap = _time(feature["planned_off_block"]) - _time(previous["planned_on_block"])
                if (previous["arrival_airport"] != feature["departure_airport"] or
                        gap < timedelta(0) or gap > timedelta(hours=6)):
                    if len(current) >= 3:
                        runs.append(current)
                    current = []
            current.append(row)
        if len(current) >= 3:
            runs.append(current)
        if runs:
            chosen = min(runs, key=lambda run: (-len(run), run[0]["features"]["planned_off_block"]))
            candidates.append(chosen[:4])
    return [row for run in candidates[:max_tails] for row in run]


def build_may_replay(rows, date, *, max_tails=8, capacity=2):
    if not 1 <= capacity <= 200:
        raise ValueError("invalid hypothetical capacity")
    start = datetime.fromisoformat(date).replace(tzinfo=ZONE)
    end = start + timedelta(days=1)
    boundary = start.isoformat()
    past_rows = [row for row in rows if _time(row["features"]["planned_off_block"]) < start]
    training = eligible_training_rows(past_rows, boundary)
    selected = select_rotations(rows, date, max_tails=max_tails)
    if not training or not selected:
        raise ValueError("replay needs past completed labels and complete future rotations")
    model = HistoricalMedianBaseline().fit(training)
    predictions = {item["record_id"]: item for item in model.predict(selected)}
    flights = []
    for row in selected:
        feature = row["features"]
        prediction = predictions[row["record_id"]]
        predicted_off, _, _, predicted_on = reconstruct_times(feature["planned_off_block"], prediction)
        flights.append({"flight_id": row["record_id"], "tail": feature["tail_number"],
                        "origin": feature["departure_airport"], "destination": feature["arrival_airport"],
                        "planned_departure": _scheduler_time(feature["planned_off_block"]),
                        "planned_arrival": _scheduler_time(feature["planned_on_block"]),
                        "predicted_departure": _scheduler_time(predicted_off),
                        "predicted_arrival": _scheduler_time(predicted_on)})
    config = {"airport": "ZGGG", "horizon_start": start.isoformat(),
              "horizon_end": end.isoformat(), "slot_minutes": 15,
              "departure_capacity": capacity, "arrival_capacity": capacity,
              "mtt_minutes": 30, "closures": []}
    return {"source": "may_planned_flights_hypothetical_capacity", "date": date,
            "training_rows": len(training), "training_audit": audit_temporal_rows(past_rows, boundary),
            "selected_tails": len({flight["tail"] for flight in flights}),
            "flights": flights, "config": config}


def evaluate_may_replay(dataset_dir, dates, capacities=(1, 2), *, max_tails=8):
    dataset_dir = Path(dataset_dir)
    manifest = verify_zggg_dataset(dataset_dir)
    rows = [json.loads(line) for line in (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    if not dates or not capacities:
        raise ValueError("dates and capacities must be nonempty")
    cases, audits = [], []
    for capacity in capacities:
        for date in dates:
            replay = build_may_replay(rows, date, max_tails=max_tails, capacity=capacity)
            name = f"{date}-capacity-{capacity}"
            cases.append({"name": name, "flights": replay["flights"], "config": replay["config"]})
            audits.append({"name": name, "training_rows": replay["training_rows"],
                           "selected_tails": replay["selected_tails"],
                           "selected_flights": len(replay["flights"]),
                           "training_audit": replay["training_audit"]})
    result = compare_policies(cases)
    result.update({"scenario_origin": "may_planned_flights_hypothetical_capacity",
                   "dataset_manifest_hash": manifest["manifest_hash"],
                   "training_audits": audits,
                   "limitations": ["airport capacity and MTT are hypothetical",
                                   "flight-plan publication times are unknown",
                                   "label receipt times are unknown"]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", default=["2025-05-10", "2025-05-13", "2025-05-16", "2025-05-19"])
    parser.add_argument("--capacities", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("replay output must be a new file")
    report = evaluate_may_replay(args.dataset, args.dates, args.capacities)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scenarios": len(report["scenarios"]),
                      "scenario_sha256": report["scenario_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
