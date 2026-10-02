"""Deterministic baselines and deployable candidate for ZGGG flow prediction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import pickle

import numpy as np

from .baselines import _HistogramStumpBoost
from .flow import FLOW_HORIZONS, FLOW_SCHEMA_VERSION


FLOW_COMPONENTS = tuple(
    f"{kind}_{horizon}m" for horizon in FLOW_HORIZONS for kind in ("takeoff", "landing")
)
FLOW_LABELS = tuple(
    f"{kind}_{horizon}m" for horizon in FLOW_HORIZONS
    for kind in ("takeoff", "landing", "total")
)
FLOW_FEATURES = tuple(
    f"{prefix}_{kind}_{horizon}m" for horizon in FLOW_HORIZONS
    for prefix in ("planned", "completed") for kind in ("takeoff", "landing", "total")
) + ("cutoff_hour", "cutoff_weekday")


def _time(features):
    return datetime.fromisoformat(features["prediction_cutoff_time"].replace("Z", "+00:00"))


def _matrix(rows):
    values = []
    for row in rows:
        features, timestamp = row["features"], _time(row["features"])
        values.append([
            float(timestamp.hour if name == "cutoff_hour" else
                  timestamp.weekday() if name == "cutoff_weekday" else
                  features.get(name, 0.0))
            for name in FLOW_FEATURES
        ])
    return np.asarray(values, dtype=float)


def _project(record_id, raw):
    result = {"record_id": record_id}
    for horizon in FLOW_HORIZONS:
        takeoff = max(0.0, float(raw[f"takeoff_{horizon}m"]))
        landing = max(0.0, float(raw[f"landing_{horizon}m"]))
        result[f"takeoff_{horizon}m"] = takeoff
        result[f"landing_{horizon}m"] = landing
        result[f"total_{horizon}m"] = takeoff + landing
    return result


class ScheduleFlowBaseline:
    def fit(self, rows):
        return self

    def predict(self, rows):
        return [_project(row["record_id"], {
            f"{kind}_{horizon}m": row["features"].get(f"planned_{kind}_{horizon}m", 0)
            for horizon in FLOW_HORIZONS for kind in ("takeoff", "landing")
        }) for row in rows]


class SeasonalMedianFlowBaseline:
    def fit(self, rows):
        if not rows:
            raise ValueError("flow training rows are empty")
        grouped = defaultdict(list)
        for row in rows:
            timestamp = _time(row["features"])
            grouped[(timestamp.weekday(), timestamp.hour)].append(row)
            grouped[(None, timestamp.hour)].append(row)
        self.groups = {key: self._median(items) for key, items in grouped.items()}
        self.global_values = self._median(rows)
        return self

    @staticmethod
    def _median(rows):
        return {name: float(np.median([row["labels"][name] for row in rows]))
                for name in FLOW_COMPONENTS}

    def predict(self, rows):
        results = []
        for row in rows:
            timestamp = _time(row["features"])
            values = self.groups.get((timestamp.weekday(), timestamp.hour),
                                     self.groups.get((None, timestamp.hour), self.global_values))
            results.append(_project(row["record_id"], values))
        return results


class GradientBoostingFlowModel:
    def fit(self, rows):
        if not rows:
            raise ValueError("flow training rows are empty")
        matrix = _matrix(rows)
        self.models = {
            name: _HistogramStumpBoost().fit(
                matrix, np.asarray([row["labels"][name] for row in rows], dtype=float),
            ) for name in FLOW_COMPONENTS
        }
        return self

    def predict(self, rows):
        if not rows:
            return []
        matrix = _matrix(rows)
        values = {name: model.predict(matrix) for name, model in self.models.items()}
        return [_project(row["record_id"], {
            name: values[name][index] for name in FLOW_COMPONENTS
        }) for index, row in enumerate(rows)]


def make_flow_model(name):
    models = {"schedule": ScheduleFlowBaseline,
              "seasonal_median": SeasonalMedianFlowBaseline,
              "gradient_boosting": GradientBoostingFlowModel}
    try:
        return models[name]()
    except KeyError as error:
        raise ValueError("unsupported flow model") from error


@dataclass
class FlowModelArtifact:
    model_name: str
    dataset_manifest_hash: str
    feature_contract: tuple[str, ...]
    flow_schema_version: str
    model: object

    @classmethod
    def fit(cls, model_name, rows, dataset_manifest_hash):
        return cls(model_name, dataset_manifest_hash, FLOW_FEATURES,
                   FLOW_SCHEMA_VERSION, make_flow_model(model_name).fit(rows))

    def predict(self, rows):
        return self.model.predict(rows)


def load_flow_artifact(path: Path, expected_manifest_hash: str) -> FlowModelArtifact:
    try:
        artifact = pickle.loads(Path(path).read_bytes())
    except (OSError, pickle.PickleError, EOFError) as error:
        raise ValueError("invalid flow model artifact") from error
    if not isinstance(artifact, FlowModelArtifact):
        raise ValueError("invalid flow model artifact")
    if artifact.dataset_manifest_hash != expected_manifest_hash:
        raise ValueError("flow model dataset manifest mismatch")
    if artifact.feature_contract != FLOW_FEATURES or artifact.flow_schema_version != FLOW_SCHEMA_VERSION:
        raise ValueError("flow model feature contract mismatch")
    return artifact


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _mae(rows, predictions):
    by_id = {item["record_id"]: item for item in predictions}
    errors = []
    for row in rows:
        prediction = by_id[row["record_id"]]
        errors.extend(abs(float(prediction[name]) - float(row["labels"][name]))
                      for name in FLOW_LABELS)
    return float(np.mean(errors)) if errors else float("inf")


def train_flow_candidate(dataset: Path, output: Path, verify_manifest: bool = True):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if verify_manifest:
        from .zggg_dataset import verify_zggg_dataset
        manifest = verify_zggg_dataset(dataset)
    else:
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    manifest_hash = manifest.get("manifest_hash")
    if not manifest_hash:
        raise ValueError("dataset manifest hash is missing")
    train = _read_jsonl(dataset / "flow_train.jsonl")
    validation = _read_jsonl(dataset / "flow_validation.jsonl")
    if not train or not validation:
        raise ValueError("flow train and validation splits must be non-empty")
    scores = {}
    for name in ("schedule", "seasonal_median", "gradient_boosting"):
        model = make_flow_model(name).fit(train)
        scores[name] = _mae(validation, model.predict(validation))
    selected = min(scores, key=lambda name: (scores[name], name))
    artifact = FlowModelArtifact.fit(selected, train + validation, manifest_hash)
    output.mkdir(parents=True, exist_ok=True)
    (output / "flow_model.pkl").write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    result = {"model_name": selected, "dataset_manifest_hash": manifest_hash,
              "selection_split": "validation", "validation_mae": scores,
              "feature_contract": list(FLOW_FEATURES),
              "flow_schema_version": FLOW_SCHEMA_VERSION}
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def predict_flow_file(artifact_path: Path, input_path: Path, output_path: Path,
                      expected_manifest_hash: str):
    artifact = load_flow_artifact(artifact_path, expected_manifest_hash)
    rows = _read_jsonl(input_path)
    predictions = artifact.predict(rows)
    text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return len(predictions)
