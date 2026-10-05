from datetime import datetime, timezone

import pytest

from training.delay_risk import DelayRiskModel, risk_features, brier_score


def flight(record_id, severe, delay, *, issue=None):
    report = {"missing": False, "issue_time": issue or "2025-05-01T00:00:00Z",
              "features": {"thunderstorm": severe, "precipitation": severe}}
    return {"record_id": record_id,
            "features": {"airport_direction": "DEPARTURE",
                         "planned_off_block": "2025-05-01T01:00:00Z",
                         "prediction_cutoff_time": "2025-05-01T01:00:00Z",
                         "planned_total_flow": 40,
                         "departure_weather": {"metar": report},
                         "arrival_weather": {"metar": {"missing": True}}},
            "labels": {"off_block_delay_min": delay}}


def test_risk_model_learns_weather_signal_and_predicts_probability():
    train = [flight(f"normal-{i}", False, 0) for i in range(40)] + [
        flight(f"storm-{i}", True, 60) for i in range(40)]
    model = DelayRiskModel().fit(train)
    normal, storm = model.predict_proba([flight("n", False, 0), flight("s", True, 60)])
    assert 0 < normal < storm < 1
    assert brier_score([0, 1], [normal, storm]) < brier_score([0, 1], [.5, .5])


def test_rejects_weather_issued_after_prediction_cutoff():
    with pytest.raises(ValueError, match="future weather"):
        risk_features(flight("late", True, 60, issue="2025-05-01T02:00:00Z"))


def test_rejects_nonfinite_flow_and_invalid_labels():
    invalid = flight("bad", False, 0)
    invalid["features"]["planned_total_flow"] = float("nan")
    with pytest.raises(ValueError, match="planned_total_flow"):
        risk_features(invalid)
    with pytest.raises(ValueError, match="binary"):
        brier_score([2], [0.5])
