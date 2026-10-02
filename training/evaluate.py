"""Unified metrics and release gates for flight prediction candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .schema import LABEL_FIELDS


@dataclass(frozen=True)
class EvaluationReport:
    sample_count: int
    components: dict[str, dict[str, float]]
    absolute_times: dict[str, dict[str, float]]
    time_order_rate: float
    structured_success_rate: float
    groups: dict[str, dict[str, float | int]]


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]


def _metrics(errors):
    absolute = np.abs(np.asarray(errors, dtype=float))
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "p90_absolute_error": float(np.percentile(absolute, 90)),
        "within_15_min": float(np.mean(absolute <= 15)),
        "within_30_min": float(np.mean(absolute <= 30)),
    }


def evaluate_predictions(labels, predictions, *, label_manifest_hash, prediction_label_hash,
                         group_fields=(), minimum_group_size=20, failed_record_ids=()):
    if not label_manifest_hash or label_manifest_hash != prediction_label_hash:
        raise ValueError("label manifest hash mismatch")
    label_ids = [row.get("record_id") for row in labels]
    prediction_ids = [row.get("record_id") for row in predictions]
    failed_ids = list(failed_record_ids)
    if (len(label_ids) != len(set(label_ids)) or len(prediction_ids) != len(set(prediction_ids)) or
            len(failed_ids) != len(set(failed_ids))):
        raise ValueError("record IDs must be unique")
    if set(prediction_ids) & set(failed_ids) or set(label_ids) != set(prediction_ids) | set(failed_ids) or not label_ids:
        raise ValueError("prediction coverage does not match labels")
    if not prediction_ids:
        raise ValueError("no successful predictions to evaluate")
    by_id = {row["record_id"]: row for row in predictions}
    component_errors = {name: [] for name in LABEL_FIELDS}
    cumulative_errors = {name: [] for name in ("off_block", "takeoff", "landing", "on_block")}
    valid = 0
    group_errors = {}
    for truth in labels:
        if truth["record_id"] in set(failed_ids):
            continue
        prediction = by_id[truth["record_id"]]
        errors = []
        values = []
        for name in LABEL_FIELDS:
            value = prediction.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("prediction values must be finite numbers")
            values.append(float(value))
            error = float(value) - float(truth["labels"][name])
            errors.append(error)
            component_errors[name].append(error)
        running = 0.0
        for name, error in zip(cumulative_errors, errors):
            running += error
            cumulative_errors[name].append(running)
        valid += int(values[1] >= 0 and values[2] >= 0 and values[3] >= 0)
        for field in group_fields:
            value = truth.get("features", {}).get(field)
            if value not in (None, ""):
                group_errors.setdefault(f"{field}={value}", []).append(abs(running))
    return EvaluationReport(
        sample_count=len(labels),
        components={name: _metrics(errors) for name, errors in component_errors.items()},
        absolute_times={name: _metrics(errors) for name, errors in cumulative_errors.items()},
        time_order_rate=valid / len(predictions), structured_success_rate=len(predictions) / len(labels),
        groups={key: {"samples": len(errors), "arrival_mae": float(np.mean(errors))}
                for key, errors in sorted(group_errors.items()) if len(errors) >= minimum_group_size},
    )


def render_markdown(report: EvaluationReport) -> str:
    lines = ["# Flight prediction evaluation", "", f"Samples: {report.sample_count}", "", "## Absolute-time metrics", ""]
    for name, metrics in report.absolute_times.items():
        lines.append(f"- {name}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, P90={metrics['p90_absolute_error']:.3f}")
    if report.groups:
        lines.extend(("", "## Groups", ""))
        for name, values in report.groups.items():
            lines.append(f"- {name}: samples={values['samples']}, arrival_MAE={values['arrival_mae']:.3f}")
    return "\n".join(lines) + "\n"


def check_release_gate(candidate, schedule) -> GateResult:
    failures = []
    candidate_domains = {
        "sample_count": (0, None), "departure_mae": (0, None), "arrival_mae": (0, None),
        "structured_success_rate": (0, 1), "time_order_rate": (0, 1),
        "maximum_subgroup_regression": (0, None),
    }
    schedule_domains = {"departure_mae": (0, None), "arrival_mae": (0, None)}
    for values, domains in ((candidate, candidate_domains), (schedule, schedule_domains)):
        for field, (lower, upper) in domains.items():
            value = values.get(field)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < lower or
                    (upper is not None and value > upper)):
                return GateResult(False, ("invalid_metrics",))
    if schedule["departure_mae"] == 0 or schedule["arrival_mae"] == 0:
        return GateResult(False, ("invalid_metrics",))
    if candidate["sample_count"] < 500: failures.append("insufficient_samples")
    if candidate["departure_mae"] >= schedule["departure_mae"]: failures.append("departure_not_better")
    if candidate["arrival_mae"] >= schedule["arrival_mae"]: failures.append("arrival_not_better")
    relative = max(
        (schedule["departure_mae"] - candidate["departure_mae"]) / schedule["departure_mae"],
        (schedule["arrival_mae"] - candidate["arrival_mae"]) / schedule["arrival_mae"],
    )
    if relative < 0.05: failures.append("improvement_below_five_percent")
    if candidate["structured_success_rate"] < 0.99: failures.append("structured_success_below_99_percent")
    if candidate["time_order_rate"] != 1.0: failures.append("time_order_not_100_percent")
    if candidate["maximum_subgroup_regression"] > 0.10: failures.append("subgroup_regression")
    return GateResult(not failures, tuple(failures))
