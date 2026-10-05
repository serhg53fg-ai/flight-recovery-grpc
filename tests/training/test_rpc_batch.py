import json
from datetime import datetime, timedelta, timezone

from flight.v1 import prediction_pb2 as pb


class FakeClient:
    def predict(self, request, timeout=None):
        planned = request.flight.planned_off_block.ToDatetime(tzinfo=timezone.utc)
        response = pb.PredictResponse(
            trace_id=request.trace_id, source=pb.LLM,
            model_version="qwen-test", worker_id="worker-test",
        )
        for field, minutes in zip(
                ("off_block", "takeoff", "landing", "on_block"),
                (5, 15, 105, 112)):
            getattr(response.prediction, field).FromDatetime(
                planned + timedelta(minutes=minutes))
        return response


def sample(record_id="a"):
    return {
        "record_id": record_id,
        "features": {
            "flight_number": "CZ1234", "tail_number": "B-1234",
            "aircraft_type": "A320", "flight_nature": "J",
            "departure_airport": "ZGGG", "arrival_airport": "ZBAA",
            "planned_off_block": "2026-01-01T00:00:00Z",
            "planned_on_block": "2026-01-01T02:00:00Z",
            "planned_distance_miles": 1000,
        },
        "labels": {},
    }


def test_rpc_batch_exports_duration_components(tmp_path):
    from training.rpc_batch import export_rpc_predictions

    source = tmp_path / "validation.jsonl"
    source.write_text(json.dumps(sample()) + "\n")
    output = tmp_path / "run"

    result = export_rpc_predictions(source, output, FakeClient(), timeout=20)

    assert result["success_count"] == 1
    prediction = json.loads((output / "predictions.jsonl").read_text())
    assert prediction == {
        "record_id": "a", "off_block_delay_min": 5.0,
        "taxi_out_min": 10.0, "airborne_min": 90.0,
        "taxi_in_min": 7.0,
    }
    assert (output / "failures.jsonl").read_text() == ""


def test_rpc_batch_resume_skips_success_and_retries_failure(tmp_path):
    from training.rpc_batch import export_rpc_predictions

    source = tmp_path / "validation.jsonl"
    source.write_text("".join(json.dumps(sample(key)) + "\n" for key in ("a", "b")))
    output = tmp_path / "run"; output.mkdir()
    (output / "predictions.jsonl").write_text(json.dumps({
        "record_id": "a", "off_block_delay_min": 1, "taxi_out_min": 2,
        "airborne_min": 3, "taxi_in_min": 4,
    }) + "\n")
    (output / "failures.jsonl").write_text(json.dumps({"record_id": "b", "error_type": "old"}) + "\n")

    result = export_rpc_predictions(source, output, FakeClient(), timeout=20, resume=True)

    assert result["success_count"] == 2
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 2
    assert (output / "failures.jsonl").read_text() == ""


def test_rpc_batch_rejects_existing_output_without_resume(tmp_path):
    from training.rpc_batch import export_rpc_predictions

    source = tmp_path / "validation.jsonl"; source.write_text(json.dumps(sample()) + "\n")
    output = tmp_path / "run"; output.mkdir()
    try:
        export_rpc_predictions(source, output, FakeClient(), timeout=20)
    except ValueError as error:
        assert "new directory" in str(error)
    else:
        raise AssertionError("existing output was accepted")
