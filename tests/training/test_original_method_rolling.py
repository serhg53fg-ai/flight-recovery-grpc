import pytest

from training.original_method_rolling import fold_gate, evaluate_fold_predictions, main


def _row(record_id, delay, severe=False):
    return {"record_id": record_id, "features": {
        "airport_direction": "DEPARTURE", "departure_airport": "ZGGG", "arrival_airport": "ZBAA",
        "planned_off_block": "2025-05-05T00:00:00Z", "planned_on_block": "2025-05-05T02:00:00Z",
        "prediction_cutoff_time": "2025-05-04T23:00:00Z", "planned_total_flow": 10,
        "departure_weather": {"metar": {"missing": not severe, "features": {"thunderstorm": severe}}},
    }, "labels": {"off_block_delay_min": delay, "taxi_out_min": 10,
                   "airborne_min": 100, "taxi_in_min": 10}}


def _prediction(row, delay):
    return {"record_id": row["record_id"], "off_block_delay_min": delay,
            "taxi_out_min": 10, "airborne_min": 100, "taxi_in_min": 10}


def test_gate_rejects_severe_regression_despite_aggregate_improvement():
    rows = [_row("normal", 0), _row("storm", 60, severe=True)]
    baseline = [_prediction(rows[0], 20), _prediction(rows[1], 50)]
    candidate = [_prediction(rows[0], 0), _prediction(rows[1], 40)]
    paired = evaluate_fold_predictions(rows, baseline, candidate, "manifest")
    assert paired["overall"]["on_block"]["paired_delta_mae"] < 0
    assert paired["groups"]["severe_weather=true"]["on_block"]["paired_delta_mae"] > 0
    assert fold_gate(paired) is False


def test_fold_rejects_mismatched_prediction_ids():
    rows = [_row("a", 0)]
    with pytest.raises(ValueError, match="IDs"):
        evaluate_fold_predictions(rows, [_prediction(rows[0], 0)],
                                  [_prediction(rows[0], 0) | {"record_id": "b"}], "manifest")


def test_cli_rejects_existing_output_before_reading_dataset(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="new file"):
        main(["--dataset", str(tmp_path / "missing"), "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "preserve"
