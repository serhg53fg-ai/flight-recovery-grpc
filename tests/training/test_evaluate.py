import math

import pytest

from training.evaluate import check_release_gate, evaluate_predictions, render_markdown


def labels(count=2):
    base = [
        {"record_id": "a", "labels": {"off_block_delay_min": 0.0, "taxi_out_min": 10.0, "airborne_min": 100.0, "taxi_in_min": 5.0}},
        {"record_id": "b", "labels": {"off_block_delay_min": 20.0, "taxi_out_min": 20.0, "airborne_min": 120.0, "taxi_in_min": 15.0}},
    ]
    return base[:count]


def predictions():
    return [
        {"record_id": "a", "off_block_delay_min": 2.0, "taxi_out_min": 12.0, "airborne_min": 96.0, "taxi_in_min": 7.0},
        {"record_id": "b", "off_block_delay_min": 16.0, "taxi_out_min": 18.0, "airborne_min": 126.0, "taxi_in_min": 11.0},
    ]


def test_computes_component_and_cumulative_time_metrics():
    report = evaluate_predictions(labels(), predictions(), label_manifest_hash="labels-v1", prediction_label_hash="labels-v1")
    assert report.sample_count == 2
    assert report.components["off_block_delay_min"]["mae"] == 3.0
    assert report.components["off_block_delay_min"]["rmse"] == pytest.approx(math.sqrt(10))
    assert report.absolute_times["off_block"]["mae"] == 3.0
    assert report.absolute_times["takeoff"]["mae"] == 5.0
    assert report.absolute_times["landing"]["mae"] == 0.0
    assert report.absolute_times["on_block"]["mae"] == 3.0
    assert report.absolute_times["on_block"]["within_15_min"] == 1.0
    assert report.time_order_rate == 1.0


@pytest.mark.parametrize("bad", [
    [{**predictions()[0]}, {**predictions()[0]}],
    predictions()[:1],
    predictions() + [{**predictions()[0], "record_id": "extra"}],
    [{**predictions()[0], "taxi_out_min": float("nan")}, predictions()[1]],
])
def test_rejects_invalid_prediction_coverage_or_values(bad):
    with pytest.raises(ValueError):
        evaluate_predictions(labels(), bad, label_manifest_hash="same", prediction_label_hash="same")


def test_rejects_label_manifest_mismatch():
    with pytest.raises(ValueError, match="hash"):
        evaluate_predictions(labels(), predictions(), label_manifest_hash="a", prediction_label_hash="b")


def test_release_gate_requires_sample_quality_improvement_and_validity():
    schedule = {"sample_count": 500, "departure_mae": 20.0, "arrival_mae": 30.0}
    candidate = {"sample_count": 500, "departure_mae": 18.0, "arrival_mae": 27.0,
                 "structured_success_rate": 0.995, "time_order_rate": 1.0,
                 "maximum_subgroup_regression": 0.05}
    assert check_release_gate(candidate, schedule).passed
    for field, value in (("sample_count", 499), ("structured_success_rate", .98),
                         ("time_order_rate", .999), ("maximum_subgroup_regression", .11),
                         ("arrival_mae", 30.0)):
        broken = dict(candidate, **{field: value})
        assert not check_release_gate(broken, schedule).passed


def test_reports_only_groups_meeting_minimum_sample_size():
    grouped_labels = labels()
    grouped_labels[0]["features"] = {"departure_airport": "ZGGG"}
    grouped_labels[1]["features"] = {"departure_airport": "ZGGG"}
    report = evaluate_predictions(grouped_labels, predictions(), label_manifest_hash="x",
                                  prediction_label_hash="x", group_fields=("departure_airport",),
                                  minimum_group_size=2)
    assert report.groups["departure_airport=ZGGG"]["samples"] == 2
    assert "departure_airport=ZGGG" in render_markdown(report)


def test_failed_records_reduce_structured_success_rate():
    report = evaluate_predictions(labels(), predictions()[:1], failed_record_ids=("b",),
                                  label_manifest_hash="x", prediction_label_hash="x")
    assert report.sample_count == 2
    assert report.structured_success_rate == 0.5


@pytest.mark.parametrize(("field", "value"), [
    ("departure_mae", float("nan")), ("arrival_mae", float("inf")),
    ("departure_mae", -1.0), ("structured_success_rate", 1.01),
    ("maximum_subgroup_regression", -0.01),
])
def test_release_gate_rejects_invalid_metric_domains(field, value):
    schedule = {"sample_count": 500, "departure_mae": 20.0, "arrival_mae": 30.0}
    candidate = {"sample_count": 500, "departure_mae": 18.0, "arrival_mae": 27.0,
                 "structured_success_rate": 0.995, "time_order_rate": 1.0,
                 "maximum_subgroup_regression": 0.05}
    assert not check_release_gate({**candidate, field: value}, schedule).passed
