"""Paired, retrospective error diagnostics for verified ZGGG flight predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .schema import LABEL_FIELDS
from .zggg_dataset import verify_zggg_dataset
from .zggg_evaluate import _peak, _severe


def _index(rows, expected, name):
    ids = [row.get("record_id") for row in rows]
    if (len(ids) != len(set(ids)) or len(ids) != len(expected) or set(ids) != expected):
        raise ValueError(f"{name} prediction IDs do not match labels")
    return {row["record_id"]: row for row in rows}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("prediction and label values must be finite numbers")
    return float(value)


def _metric(pairs):
    count = len(pairs)
    baseline = sum(value[0] for value in pairs) / count
    candidate = sum(value[1] for value in pairs) / count
    return {
        "baseline_mae": baseline,
        "candidate_mae": candidate,
        "paired_delta_mae": candidate - baseline,
        "candidate_better_rows": sum(value[1] < value[0] for value in pairs),
        "candidate_worse_rows": sum(value[1] > value[0] for value in pairs),
        "tied_rows": sum(value[1] == value[0] for value in pairs),
    }


def compare_flight_predictions(rows, baseline, candidate, dataset_manifest_hash):
    if not isinstance(dataset_manifest_hash, str) or not dataset_manifest_hash:
        raise ValueError("dataset manifest hash is required")
    ids = [row.get("record_id") for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("label IDs must be unique and nonempty")
    expected = set(ids)
    baseline_by_id = _index(baseline, expected, "baseline")
    candidate_by_id = _index(candidate, expected, "candidate")
    metrics = {name: [] for name in (*LABEL_FIELDS, "off_block", "on_block")}
    groups = {}
    for row in rows:
        features = row["features"]
        direction = features.get("airport_direction")
        if direction not in {"DEPARTURE", "ARRIVAL"}:
            raise ValueError("invalid ZGGG direction")
        record_id = row["record_id"]
        truth = [_number(row["labels"].get(name)) for name in LABEL_FIELDS]
        predictions = []
        for lookup in (baseline_by_id, candidate_by_id):
            values = [_number(lookup[record_id].get(name)) for name in LABEL_FIELDS]
            if any(value < 0 for value in values[1:]):
                raise ValueError("prediction duration must be nonnegative")
            predictions.append(values)
        errors = {
            name: (abs(predictions[0][index] - truth[index]),
                   abs(predictions[1][index] - truth[index]))
            for index, name in enumerate(LABEL_FIELDS)
        }
        errors["off_block"] = errors["off_block_delay_min"]
        errors["on_block"] = (abs(sum(predictions[0]) - sum(truth)),
                              abs(sum(predictions[1]) - sum(truth)))
        for name, pair in errors.items():
            metrics[name].append(pair)
        names = [f"direction={direction}",
                 "actual_off_block_delay_ge_30m" if truth[0] >= 30 else "actual_off_block_delay_lt_30m",
                 f"peak_period={str(_peak(features)).lower()}",
                 f"severe_weather={str(_severe(features)).lower()}"]
        for group_name in names:
            group = groups.setdefault(group_name, {name: [] for name in metrics})
            for name, pair in errors.items():
                group[name].append(pair)
    return {
        "dataset_manifest_hash": dataset_manifest_hash,
        "evaluation_scope": "retrospective_paired_same_records",
        "samples": len(rows),
        "overall": {name: _metric(pairs) for name, pairs in metrics.items()},
        "groups": {name: {"samples": len(values["on_block"]),
                          **{metric: _metric(pairs) for metric, pairs in values.items()}}
                   for name, values in sorted(groups.items())},
    }


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("diagnostic output must be a new file")
    manifest = verify_zggg_dataset(args.dataset)
    report = compare_flight_predictions(
        _jsonl(args.dataset / f"{args.split}.jsonl"), _jsonl(args.baseline),
        _jsonl(args.candidate), manifest["manifest_hash"])
    report["split"] = args.split
    report["baseline_file"] = str(args.baseline.resolve())
    report["candidate_file"] = str(args.candidate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"samples": report["samples"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
