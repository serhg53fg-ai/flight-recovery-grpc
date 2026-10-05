"""ZGGG-only flight and airport-flow evaluation and release gates."""

from __future__ import annotations

import math
from zoneinfo import ZoneInfo

import numpy as np

from .flow_model import FLOW_LABELS
from .schema import LABEL_FIELDS, ZGGG_AIRPORT


ZGGG_GATE_THRESHOLDS = {
    "minimum_structured_success_rate": 0.99,
    "required_time_order_rate": 1.0,
    "maximum_flight_regression": 0.0,
    "maximum_flow_regression": 0.0,
    "maximum_group_regression": 0.10,
}


def _metrics(errors):
    values = np.asarray(errors, dtype=float)
    absolute = np.abs(values)
    return {"mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(values))))}


def _prediction_map(labels, predictions, name):
    label_ids = [row.get("record_id") for row in labels]
    prediction_ids = [row.get("record_id") for row in predictions]
    if (not label_ids or len(label_ids) != len(set(label_ids)) or
            len(prediction_ids) != len(set(prediction_ids)) or
            set(label_ids) != set(prediction_ids)):
        raise ValueError(f"{name} prediction IDs do not match labels")
    return {row["record_id"]: row for row in predictions}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("prediction values must be finite numbers")
    return float(value)


def _severe(features):
    for side in ("departure_weather", "arrival_weather"):
        report = features.get(side, {}).get("metar", {})
        values = report.get("features", {}) if not report.get("missing", True) else {}
        if values.get("precipitation") or values.get("thunderstorm"):
            return True
        visibility, ceiling = values.get("visibility_m"), values.get("ceiling_ft")
        if visibility is not None and visibility < 3000:
            return True
        if ceiling is not None and ceiling < 500:
            return True
    return False


def _peak(features):
    from datetime import datetime
    timestamp = datetime.fromisoformat(features["planned_off_block"].replace("Z", "+00:00"))
    hour = timestamp.astimezone(ZoneInfo("Asia/Shanghai")).hour
    return 7 <= hour < 10 or 17 <= hour < 20


def evaluate_zggg_predictions(flight_labels, flight_predictions, flow_labels,
                              flow_predictions, label_manifest_hash,
                              prediction_manifest_hash):
    if not label_manifest_hash or label_manifest_hash != prediction_manifest_hash:
        raise ValueError("label manifest hash mismatch")
    flight_by_id = _prediction_map(flight_labels, flight_predictions, "flight")
    flow_by_id = _prediction_map(flow_labels, flow_predictions, "flow")
    component_errors = {name: [] for name in LABEL_FIELDS}
    departure_errors, arrival_errors = [], []
    group_errors: dict[str, list[float]] = {}
    valid_order = 0
    for truth in flight_labels:
        features = truth.get("features", {})
        departure, arrival = features.get("departure_airport"), features.get("arrival_airport")
        if (departure == ZGGG_AIRPORT) == (arrival == ZGGG_AIRPORT):
            raise ValueError("flight record is outside ZGGG scope")
        expected_direction = "DEPARTURE" if departure == ZGGG_AIRPORT else "ARRIVAL"
        if features.get("airport_direction") != expected_direction:
            raise ValueError("flight record has invalid ZGGG direction")
        prediction = flight_by_id[truth["record_id"]]
        errors, values = [], []
        for name in LABEL_FIELDS:
            value = _number(prediction.get(name))
            values.append(value)
            error = value - float(truth["labels"][name])
            component_errors[name].append(error)
            errors.append(error)
        valid_order += int(all(value >= 0 for value in values[1:]))
        cumulative = sum(errors)
        if expected_direction == "DEPARTURE":
            departure_errors.append(errors[0])
        else:
            arrival_errors.append(cumulative)
        for key, enabled in ((f"direction={expected_direction}", True),
                             ("peak_period=true", _peak(features)),
                             ("severe_weather=true", _severe(features))):
            if enabled:
                group_errors.setdefault(key, []).append(cumulative)
    if not departure_errors or not arrival_errors:
        raise ValueError("both ZGGG directions are required")
    flow_errors = {name: [] for name in FLOW_LABELS}
    for truth in flow_labels:
        if truth.get("features", {}).get("airport") != ZGGG_AIRPORT:
            raise ValueError("flow record is outside ZGGG scope")
        prediction = flow_by_id[truth["record_id"]]
        for name in FLOW_LABELS:
            flow_errors[name].append(_number(prediction.get(name)) - float(truth["labels"][name]))
    return {
        "dataset_manifest_hash": label_manifest_hash,
        "flight": {
            "samples": len(flight_labels),
            "components": {name: _metrics(errors) for name, errors in component_errors.items()},
            "departure_off_block": _metrics(departure_errors),
            "arrival_on_block": _metrics(arrival_errors),
            "structured_success_rate": len(flight_predictions) / len(flight_labels),
            "time_order_rate": valid_order / len(flight_predictions),
        },
        "flow": {name: _metrics(errors) for name, errors in flow_errors.items()},
        "groups": {key: {"samples": len(errors), "on_block_mae": _metrics(errors)["mae"]}
                   for key, errors in sorted(group_errors.items())},
    }


def _mean_mae(metrics):
    return float(np.mean([value["mae"] for value in metrics.values()]))


def gate_zggg_release(candidate, reference, thresholds=None):
    limits = dict(ZGGG_GATE_THRESHOLDS if thresholds is None else thresholds)
    if candidate.get("dataset_manifest_hash") != reference.get("dataset_manifest_hash"):
        raise ValueError("candidate/reference manifest mismatch")
    failures = []
    flight_candidate = candidate["flight"]
    flight_reference = reference["flight"]
    candidate_flight_mae = np.mean([
        flight_candidate["departure_off_block"]["mae"],
        flight_candidate["arrival_on_block"]["mae"],
        *[value["mae"] for value in flight_candidate["components"].values()],
    ])
    reference_flight_mae = np.mean([
        flight_reference["departure_off_block"]["mae"],
        flight_reference["arrival_on_block"]["mae"],
        *[value["mae"] for value in flight_reference["components"].values()],
    ])
    if candidate_flight_mae - reference_flight_mae > limits["maximum_flight_regression"]:
        failures.append("flight_regression")
    if _mean_mae(candidate["flow"]) - _mean_mae(reference["flow"]) > limits["maximum_flow_regression"]:
        failures.append("flow_regression")
    if flight_candidate["structured_success_rate"] < limits["minimum_structured_success_rate"]:
        failures.append("structured_success_below_threshold")
    if flight_candidate["time_order_rate"] < limits["required_time_order_rate"]:
        failures.append("time_order_below_threshold")
    for key, values in reference.get("groups", {}).items():
        if key not in candidate.get("groups", {}):
            failures.append(f"missing_group:{key}")
            continue
        baseline = values["on_block_mae"]
        actual = candidate["groups"][key]["on_block_mae"]
        regression = (actual - baseline) / baseline if baseline else (0 if actual == 0 else 1)
        if regression > limits["maximum_group_regression"]:
            failures.append(f"group_regression:{key}")
    return {"passed": not failures, "failures": failures, "thresholds": limits,
            "candidate_flight_mae": float(candidate_flight_mae),
            "reference_flight_mae": float(reference_flight_mae),
            "candidate_flow_mae": _mean_mae(candidate["flow"]),
            "reference_flow_mae": _mean_mae(reference["flow"])}
