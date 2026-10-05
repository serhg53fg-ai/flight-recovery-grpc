"""Resumable validation/test prediction export through the real gRPC gateway."""

from __future__ import annotations

from datetime import timezone
import json
from pathlib import Path
import time
from uuid import uuid4

from flight.v1 import prediction_pb2 as pb

from apps.flight.domain.prediction import normalize_flight
from .schema import LABEL_FIELDS


def _minutes(later, earlier):
    return (later - earlier).total_seconds() / 60.0


def _components(record_id, planned, response):
    prediction = response.prediction
    values = [getattr(prediction, field).ToDatetime(tzinfo=timezone.utc) for field in
              ("off_block", "takeoff", "landing", "on_block")]
    result = {
        "record_id": record_id,
        "off_block_delay_min": _minutes(values[0], planned),
        "taxi_out_min": _minutes(values[1], values[0]),
        "airborne_min": _minutes(values[2], values[1]),
        "taxi_in_min": _minutes(values[3], values[2]),
    }
    if any(not isinstance(result[field], float) for field in LABEL_FIELDS):
        raise ValueError("invalid duration prediction")
    return result


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def export_rpc_predictions(source, output, client, timeout=20, resume=False):
    """Export a split via gRPC; resume retains successes and retries failures."""

    source, output = Path(source), Path(output)
    if output.exists() and not resume:
        raise ValueError("prediction output must be a new directory")
    if not 0 < float(timeout) <= 300:
        raise ValueError("invalid RPC timeout")
    rows = _read_jsonl(source)
    if not rows:
        raise ValueError("prediction split must not be empty")
    if not output.exists():
        output.mkdir(parents=True)
    prediction_path, failure_path = output / "predictions.jsonl", output / "failures.jsonl"
    completed_rows = _read_jsonl(prediction_path) if resume else []
    completed_ids = {row["record_id"] for row in completed_rows}
    if len(completed_ids) != len(completed_rows):
        raise ValueError("existing predictions contain duplicate record_id")

    # A previous failure is retried. Rebuild this file with failures from this run.
    failure_path.write_text("", encoding="utf-8")
    with prediction_path.open("a", encoding="utf-8") as predictions, \
            failure_path.open("a", encoding="utf-8") as failures:
        for row in rows:
            record_id = row.get("record_id")
            if record_id in completed_ids:
                continue
            started = time.monotonic()
            try:
                flight = normalize_flight(row["features"], "UTC")
                flight.flight_id = record_id
                response = client.predict(
                    pb.PredictRequest(trace_id=str(uuid4()), flight=flight), timeout=timeout,
                )
                planned = flight.planned_off_block.ToDatetime(tzinfo=timezone.utc)
                prediction = _components(record_id, planned, response)
            except Exception as error:
                failures.write(json.dumps({
                    "record_id": record_id,
                    "error_type": type(error).__name__,
                    "elapsed_ms": (time.monotonic() - started) * 1000,
                }, sort_keys=True) + "\n")
                failures.flush()
            else:
                predictions.write(json.dumps(prediction, sort_keys=True) + "\n")
                predictions.flush()
                completed_ids.add(record_id)

    failure_rows = _read_jsonl(failure_path)
    result = {
        "sample_count": len(rows), "success_count": len(completed_ids),
        "failure_count": len(failure_rows), "complete": len(completed_ids) == len(rows),
    }
    (output / "run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return result
