import json

import pytest


def flight_rows():
    base = {
        "planned_off_block": "2025-05-01T00:00:00Z",
        "planned_on_block": "2025-05-01T02:20:00Z",
        "departure_weather": {"metar": {"missing": False, "features": {"precipitation": False, "visibility_m": 9999}}},
        "arrival_weather": {"metar": {"missing": True}},
    }
    return [
        {"record_id": "d", "features": {**base, "departure_airport": "ZGGG",
          "arrival_airport": "ZBAA", "airport_direction": "DEPARTURE"},
         "labels": {"off_block_delay_min": 10, "taxi_out_min": 20,
                    "airborne_min": 100, "taxi_in_min": 10}},
        {"record_id": "a", "features": {**base, "departure_airport": "ZBAA",
          "arrival_airport": "ZGGG", "airport_direction": "ARRIVAL",
          "arrival_weather": {"metar": {"missing": False, "features": {
              "precipitation": True, "visibility_m": 2000}}}},
         "labels": {"off_block_delay_min": 5, "taxi_out_min": 15,
                    "airborne_min": 110, "taxi_in_min": 10}},
    ]


def flight_predictions(delta=0):
    return [
        {"record_id": row["record_id"], **{
            name: value + delta for name, value in row["labels"].items()
        }} for row in flight_rows()
    ]


def flow_rows():
    labels = {}
    features = {"airport": "ZGGG", "prediction_cutoff_time": "2025-05-01T00:00:00Z"}
    for horizon in (15, 30, 60):
        labels.update({f"takeoff_{horizon}m": horizon // 15,
                       f"landing_{horizon}m": horizon // 15 + 1,
                       f"total_{horizon}m": horizon // 15 * 2 + 1})
    return [{"record_id": "f", "features": features, "labels": labels}]


def flow_predictions(delta=0):
    truth = flow_rows()[0]
    return [{"record_id": "f", **{key: value + delta
            for key, value in truth["labels"].items()}}]


def test_reports_zggg_flight_flow_and_operational_groups():
    from training.zggg_evaluate import evaluate_zggg_predictions

    report = evaluate_zggg_predictions(
        flight_rows(), flight_predictions(1), flow_rows(), flow_predictions(1),
        "same", "same",
    )
    assert report["flight"]["departure_off_block"]["mae"] == 1
    assert report["flight"]["arrival_on_block"]["mae"] == 4
    assert set(report["flight"]["components"]) == {
        "off_block_delay_min", "taxi_out_min", "airborne_min", "taxi_in_min"}
    assert len(report["flow"]) == 9
    assert report["groups"]["direction=DEPARTURE"]["samples"] == 1
    assert report["groups"]["direction=ARRIVAL"]["samples"] == 1
    assert report["groups"]["peak_period=true"]["samples"] == 2
    assert report["groups"]["severe_weather=true"]["samples"] == 1
    assert not any("ZBAA" in key for key in report["groups"])
    json.dumps(report, allow_nan=False)


def test_rejects_manifest_id_scope_and_non_finite_mismatches():
    from training.zggg_evaluate import evaluate_zggg_predictions

    args = (flight_rows(), flight_predictions(), flow_rows(), flow_predictions())
    with pytest.raises(ValueError, match="manifest"):
        evaluate_zggg_predictions(*args, "a", "b")
    with pytest.raises(ValueError, match="IDs"):
        evaluate_zggg_predictions(args[0], args[1][:-1], *args[2:], "a", "a")
    outside = flight_rows()
    outside[0]["features"]["departure_airport"] = "ZSPD"
    with pytest.raises(ValueError, match="ZGGG"):
        evaluate_zggg_predictions(outside, args[1], *args[2:], "a", "a")


def test_structure_and_time_order_are_explicit():
    from training.zggg_evaluate import evaluate_zggg_predictions

    predictions = flight_predictions()
    predictions[0]["taxi_out_min"] = -1
    report = evaluate_zggg_predictions(
        flight_rows(), predictions, flow_rows(), flow_predictions(), "same", "same")
    assert report["flight"]["structured_success_rate"] == 1
    assert report["flight"]["time_order_rate"] == .5


def test_gate_uses_frozen_thresholds_and_ignores_unrelated_airports():
    from training.zggg_evaluate import gate_zggg_release

    candidate = evaluate(delta=0)
    reference = evaluate(delta=3)
    result = gate_zggg_release(candidate, reference)
    assert result["passed"] is True
    assert "thresholds" in result
    degraded = evaluate(delta=5)
    result = gate_zggg_release(degraded, reference)
    assert result["passed"] is False
    assert "flight_regression" in result["failures"]


def evaluate(delta):
    from training.zggg_evaluate import evaluate_zggg_predictions
    return evaluate_zggg_predictions(
        flight_rows(), flight_predictions(delta), flow_rows(), flow_predictions(delta),
        "same", "same",
    )
