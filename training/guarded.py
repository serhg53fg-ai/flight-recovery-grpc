"""Learn and apply an airport-level fallback policy from validation evidence."""

from __future__ import annotations

from collections import defaultdict
import math

from .schema import LABEL_FIELDS


def _prediction_index(rows, source):
    result = {}
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in result:
            raise ValueError(f"{source} contains invalid or duplicate record_id")
        values = {}
        for field in LABEL_FIELDS:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{source} contains invalid {field}")
            values[field] = float(value)
        result[record_id] = values
    return result


def _labels(rows):
    result = {}
    for row in rows:
        record_id = row.get("record_id")
        features, labels = row.get("features", {}), row.get("labels", {})
        if not isinstance(record_id, str) or not record_id or record_id in result:
            raise ValueError("labels contain invalid or duplicate record_id")
        departure, arrival = features.get("departure_airport"), features.get("arrival_airport")
        if not departure or not arrival:
            raise ValueError("labels contain missing airport")
        try:
            total = sum(float(labels[field]) for field in LABEL_FIELDS)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("labels contain invalid components") from error
        if not math.isfinite(total):
            raise ValueError("labels contain non-finite components")
        result[record_id] = (str(departure), str(arrival), total)
    return result


def fit_policy(labels, candidate, baseline, minimum_group_size=20, maximum_regression=0.0):
    """Allow candidate use only for airport groups supported by validation data."""

    if not isinstance(minimum_group_size, int) or minimum_group_size < 1:
        raise ValueError("minimum_group_size must be positive")
    if not 0 <= float(maximum_regression) <= 1:
        raise ValueError("maximum_regression must be between zero and one")
    truths = _labels(labels)
    candidate_index = _prediction_index(candidate, "candidate")
    baseline_index = _prediction_index(baseline, "baseline")
    if not truths or truths.keys() != candidate_index.keys() or truths.keys() != baseline_index.keys():
        raise ValueError("label and prediction record_id sets must match")

    groups = {"departure": defaultdict(list), "arrival": defaultdict(list)}
    for record_id, (departure, arrival, truth) in truths.items():
        candidate_error = abs(sum(candidate_index[record_id].values()) - truth)
        baseline_error = abs(sum(baseline_index[record_id].values()) - truth)
        groups["departure"][departure].append((candidate_error, baseline_error))
        groups["arrival"][arrival].append((candidate_error, baseline_error))

    evidence = {}
    eligible = {}
    for dimension, values in groups.items():
        evidence[dimension] = {}
        eligible[dimension] = []
        for airport, errors in sorted(values.items()):
            candidate_mae = sum(item[0] for item in errors) / len(errors)
            baseline_mae = sum(item[1] for item in errors) / len(errors)
            accepted = (len(errors) >= minimum_group_size and
                        candidate_mae <= baseline_mae * (1 + maximum_regression))
            evidence[dimension][airport] = {
                "samples": len(errors), "candidate_arrival_mae": candidate_mae,
                "baseline_arrival_mae": baseline_mae, "eligible": accepted,
            }
            if accepted:
                eligible[dimension].append(airport)
    return {
        "policy_version": "airport-fallback-v1",
        "validation_rows": len(truths),
        "minimum_group_size": minimum_group_size,
        "maximum_regression": float(maximum_regression),
        "eligible_departure_airports": eligible["departure"],
        "eligible_arrival_airports": eligible["arrival"],
        "evidence": evidence,
    }


def apply_policy(rows, candidate, baseline, policy):
    """Choose candidate only when both endpoint airport groups are eligible."""

    records = {}
    for row in rows:
        record_id = row.get("record_id")
        features = row.get("features", {})
        departure, arrival = features.get("departure_airport"), features.get("arrival_airport")
        if (not isinstance(record_id, str) or not record_id or record_id in records or
                not departure or not arrival):
            raise ValueError("rows contain invalid record_id or airport")
        records[record_id] = (str(departure), str(arrival))
    candidate_index = _prediction_index(candidate, "candidate")
    baseline_index = _prediction_index(baseline, "baseline")
    if records.keys() != candidate_index.keys() or records.keys() != baseline_index.keys():
        raise ValueError("row and prediction record_id sets must match")
    departures = set(policy.get("eligible_departure_airports", ()))
    arrivals = set(policy.get("eligible_arrival_airports", ()))
    result = []
    for row in rows:
        record_id = row["record_id"]
        departure, arrival = records[record_id]
        source = candidate_index if departure in departures and arrival in arrivals else baseline_index
        result.append({"record_id": record_id, **source[record_id]})
    return result
