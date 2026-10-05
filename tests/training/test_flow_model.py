import json
import pickle

import pytest


def flow_rows(count=12):
    rows = []
    for index in range(count):
        features = {"airport": "ZGGG", "prediction_cutoff_time": f"2025-05-{index + 1:02d}T00:00:00Z",
                    "flow_schema_version": "zggg-airport-flow-v1"}
        labels = {}
        for horizon in (15, 30, 60):
            takeoff, landing = index % 4 + horizon // 15, index % 3 + horizon // 15
            features.update({
                f"planned_takeoff_{horizon}m": takeoff + 1,
                f"planned_landing_{horizon}m": landing + 1,
                f"planned_total_{horizon}m": takeoff + landing + 2,
                f"completed_takeoff_{horizon}m": index % 4,
                f"completed_landing_{horizon}m": index % 3,
                f"completed_total_{horizon}m": index % 4 + index % 3,
            })
            labels.update({f"takeoff_{horizon}m": takeoff,
                           f"landing_{horizon}m": landing,
                           f"total_{horizon}m": takeoff + landing})
        rows.append({"record_id": f"flow-{index}", "features": features, "labels": labels})
    return rows


@pytest.mark.parametrize("model_name", ["schedule", "seasonal_median", "gradient_boosting"])
def test_models_predict_all_nine_outputs_deterministically(model_name):
    from training.flow_model import make_flow_model, FLOW_LABELS

    rows = flow_rows()
    model = make_flow_model(model_name).fit(rows[:9])
    first = model.predict(rows[9:])
    second = model.predict(rows[9:])
    assert first == second
    assert all(set(item) == {"record_id", *FLOW_LABELS} for item in first)
    for item in first:
        for horizon in (15, 30, 60):
            assert item[f"takeoff_{horizon}m"] >= 0
            assert item[f"landing_{horizon}m"] >= 0
            assert item[f"total_{horizon}m"] == (
                item[f"takeoff_{horizon}m"] + item[f"landing_{horizon}m"])


def test_serialized_artifact_round_trip_and_manifest_guard(tmp_path):
    from training.flow_model import FlowModelArtifact, load_flow_artifact

    rows = flow_rows()
    artifact = FlowModelArtifact.fit("gradient_boosting", rows, "manifest-a")
    path = tmp_path / "flow.pkl"
    path.write_bytes(pickle.dumps(artifact))
    restored = load_flow_artifact(path, "manifest-a")
    assert restored.predict(rows[-2:]) == artifact.predict(rows[-2:])
    with pytest.raises(ValueError, match="manifest"):
        load_flow_artifact(path, "manifest-b")


def test_training_selects_on_validation_and_refits_frozen_candidate(tmp_path):
    from training.flow_model import train_flow_candidate, predict_flow_file

    dataset, output = tmp_path / "dataset", tmp_path / "model"
    dataset.mkdir()
    rows = flow_rows()
    for name, selected in (("train", rows[:7]), ("validation", rows[7:10]),
                           ("test", rows[10:])):
        (dataset / f"flow_{name}.jsonl").write_text("".join(
            json.dumps(item) + "\n" for item in selected))
    (dataset / "manifest.json").write_text(json.dumps({"manifest_hash": "manifest-a"}))

    result = train_flow_candidate(dataset, output, verify_manifest=False)
    assert result["selection_split"] == "validation"
    assert set(result["validation_mae"]) == {"schedule", "seasonal_median", "gradient_boosting"}
    predictions = predict_flow_file(output / "flow_model.pkl", dataset / "flow_test.jsonl",
                                    output / "predictions.jsonl", "manifest-a")
    assert predictions == 2
