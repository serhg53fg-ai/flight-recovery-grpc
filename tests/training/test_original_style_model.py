import numpy as np
import pytest

from training.original_style_model import OriginalStyleEnsemble, standardize


def test_standardization_uses_train_statistics_only():
    train = np.array([[0.0], [2.0]])
    test = np.array([[100.0]])
    train_scaled, center, scale = standardize(train)
    assert np.allclose(train_scaled[:, 0], [-1, 1])
    assert np.allclose((test - center) / scale, [[99]])


def test_ensemble_outputs_finite_components_for_same_ids():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    rows = []
    for index in range(12):
        rows.append({"record_id": str(index), "features": {
            "planned_off_block": "2025-05-01T00:00:00Z", "prediction_cutoff_time": "2025-05-01T00:00:00Z",
            "departure_airport": "ZGGG", "arrival_airport": "ZBAA", "flight_number": str(index),
            "departure_weather": {"metar": {"missing": True}},
            "planned_total_flow": index,
        }, "labels": {"off_block_delay_min": index, "taxi_out_min": 10,
                      "airborne_min": 60, "taxi_in_min": 10}})
    model = OriginalStyleEnsemble(epochs=1, trees=2, batch_size=4).fit(rows[:10])
    predictions = model.predict(rows[10:])
    assert [item["record_id"] for item in predictions] == ["10", "11"]
    assert all(np.isfinite(item[name]) for item in predictions for name in
               ("off_block_delay_min", "taxi_out_min", "airborne_min", "taxi_in_min"))
    assert "10" not in model.features.categories["flight_number"]
    repeated = OriginalStyleEnsemble(epochs=1, trees=2, batch_size=4).fit(rows[:10]).predict(rows[10:])
    assert all(np.isclose(predictions[i][name], repeated[i][name], atol=1e-3)
               for i in range(2) for name in ("off_block_delay_min", "taxi_out_min", "airborne_min", "taxi_in_min"))
