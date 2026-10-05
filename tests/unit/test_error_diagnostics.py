import pytest

from training.error_diagnostics import compare_flight_predictions


def _row(record_id, direction, delay, *, severe=False, peak=False):
    hour = 0 if peak else 3  # 08:00 or 11:00 Asia/Shanghai
    weather = {"metar": {"missing": False, "features": {"thunderstorm": severe}}}
    return {
        "record_id": record_id,
        "features": {
            "departure_airport": "ZGGG" if direction == "DEPARTURE" else "ZHHH",
            "arrival_airport": "ZHHH" if direction == "DEPARTURE" else "ZGGG",
            "airport_direction": direction,
            "planned_off_block": f"2025-05-12T{hour:02d}:00:00Z",
            "departure_weather": weather,
            "arrival_weather": weather,
        },
        "labels": {"off_block_delay_min": delay, "taxi_out_min": 10,
                   "airborne_min": 60, "taxi_in_min": 5},
    }


def _prediction(record_id, delay):
    return {"record_id": record_id, "off_block_delay_min": delay,
            "taxi_out_min": 10, "airborne_min": 60, "taxi_in_min": 5}


def test_paired_diagnostic_exposes_high_delay_and_weather_regression():
    labels = [_row("a", "DEPARTURE", 0), _row("b", "ARRIVAL", 60, severe=True, peak=True)]
    baseline = [_prediction("a", 0), _prediction("b", 40)]
    candidate = [_prediction("a", 2), _prediction("b", 20)]

    report = compare_flight_predictions(labels, baseline, candidate, "manifest")

    assert report["samples"] == 2
    assert report["overall"]["on_block"]["baseline_mae"] == 10
    assert report["overall"]["on_block"]["candidate_mae"] == 21
    assert report["groups"]["severe_weather=true"]["on_block"]["paired_delta_mae"] == 20
    assert report["groups"]["actual_off_block_delay_ge_30m"]["samples"] == 1
    assert report["groups"]["peak_period=true"]["samples"] == 1
    assert report["overall"]["off_block"]["candidate_better_rows"] == 0


@pytest.mark.parametrize("candidate", [
    [_prediction("a", 0)],
    [_prediction("a", 0), _prediction("a", 1)],
    [_prediction("a", 0), _prediction("extra", 1)],
])
def test_rejects_missing_duplicate_or_extra_candidate_ids(candidate):
    labels = [_row("a", "DEPARTURE", 0), _row("b", "ARRIVAL", 0)]
    baseline = [_prediction("a", 0), _prediction("b", 0)]
    with pytest.raises(ValueError, match="IDs"):
        compare_flight_predictions(labels, baseline, candidate, "manifest")


def test_rejects_nonfinite_prediction():
    labels = [_row("a", "DEPARTURE", 0), _row("b", "ARRIVAL", 0)]
    baseline = [_prediction("a", 0), _prediction("b", 0)]
    candidate = [_prediction("a", float("nan")), _prediction("b", 0)]
    with pytest.raises(ValueError, match="finite"):
        compare_flight_predictions(labels, baseline, candidate, "manifest")
