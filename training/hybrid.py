"""Compose a prediction from independently evaluated component predictors."""

from __future__ import annotations

import math

from .schema import LABEL_FIELDS


def _index(rows, source):
    indexed = {}
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{source} contains an invalid record_id")
        if record_id in indexed:
            raise ValueError(f"{source} contains duplicate record_id: {record_id}")
        values = {}
        for field in LABEL_FIELDS:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{source} field {field} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{source} field {field} must be finite")
            values[field] = value
        indexed[record_id] = values
    if not indexed:
        raise ValueError(f"{source} predictions must not be empty")
    return indexed


def compose_predictions(candidate, baseline, baseline_fields=("off_block_delay_min",)):
    """Replace selected candidate components with values from a baseline.

    Records are joined by ``record_id`` and emitted in candidate order.  Strict
    validation prevents silent row loss or malformed values from reaching the
    evaluation and release-gate stages.
    """

    fields = tuple(baseline_fields)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("baseline component fields must be non-empty and unique")
    invalid = set(fields) - set(LABEL_FIELDS)
    if invalid:
        raise ValueError(f"unknown prediction component: {sorted(invalid)}")

    candidate_index = _index(candidate, "candidate")
    baseline_index = _index(baseline, "baseline")
    if candidate_index.keys() != baseline_index.keys():
        missing = sorted(candidate_index.keys() - baseline_index.keys())
        extra = sorted(baseline_index.keys() - candidate_index.keys())
        raise ValueError(f"prediction record_id mismatch: missing={missing[:5]}, extra={extra[:5]}")

    result = []
    for row in candidate:
        record_id = row["record_id"]
        result.append({
            "record_id": record_id,
            **{
                field: (baseline_index[record_id][field]
                        if field in fields else candidate_index[record_id][field])
                for field in LABEL_FIELDS
            },
        })
    return result
