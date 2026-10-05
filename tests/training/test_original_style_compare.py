import pytest

from training.original_style_compare import arm_metrics, operational_metrics, temporal_gate, main


def _row(identity, delay=0, direction="DEPARTURE"):
    return {"record_id": identity, "features": {
        "airport_direction": direction, "departure_airport": "ZGGG" if direction == "DEPARTURE" else "ZBAA",
        "arrival_airport": "ZBAA" if direction == "DEPARTURE" else "ZGGG",
        "planned_off_block": "2025-05-26T00:00:00Z", "planned_on_block": "2025-05-26T01:00:00Z",
        "departure_weather": {"metar": {"missing": True}},
    }, "labels": {"off_block_delay_min": delay, "taxi_out_min": 10,
                   "airborne_min": 40, "taxi_in_min": 10}}


def _prediction(identity, delay):
    return {"record_id": identity, "off_block_delay_min": delay,
            "taxi_out_min": 10, "airborne_min": 40, "taxi_in_min": 10}


def test_comparison_requires_identical_ids_and_keeps_arm_scope():
    rows = [_row("a", 10), _row("b", 20, "ARRIVAL")]
    with pytest.raises(ValueError, match="IDs"):
        arm_metrics(rows, [_prediction("a", 10), _prediction("b", 20)],
                    [_prediction("a", 10), _prediction("wrong", 20)], "hash")
    result = arm_metrics(rows, [_prediction("a", 0), _prediction("b", 0)],
                         [_prediction("a", 10), _prediction("b", 20)], "hash")
    assert result["samples"] == 2
    assert operational_metrics(result)["departure_off_block"]["candidate_mae"] == 0
    result["evaluation_scope"] = "temporal_test"
    assert temporal_gate(result) is True


def test_random_arm_cannot_authorize_promotion():
    assert temporal_gate({"overall": {"off_block": {"paired_delta_mae": -10},
                                       "on_block": {"paired_delta_mae": -10}}, "groups": {},
                          "evaluation_scope": "random_split_retrospective"}) is False


def test_cli_refuses_existing_report(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="new file"):
        main(["--dataset", str(tmp_path / "missing"), "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "keep"
