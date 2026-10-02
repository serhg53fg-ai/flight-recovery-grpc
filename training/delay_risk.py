"""Cutoff-safe, train-only rolling evaluation of major departure-delay risk."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .rolling_weather import expanding_folds
from .zggg_dataset import verify_zggg_dataset
from .zggg_evaluate import _severe


FEATURE_NAMES = ("intercept", "is_departure", "severe_weather", "precipitation",
                 "thunderstorm", "weather_missing", "planned_total_flow_centered",
                 "local_hour_sin", "local_hour_cos")
THRESHOLD_MINUTES = 30
STEPS = 500
LEARNING_RATE = 0.2
L2_PENALTY = 0.02


def _aware(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("risk feature time must include timezone")
    return stamp


def risk_features(row):
    features = row["features"]
    planned = _aware(features["planned_off_block"])
    cutoff = _aware(features["prediction_cutoff_time"])
    if cutoff > planned:
        raise ValueError("future prediction cutoff")
    direction = features["airport_direction"]
    if direction not in {"DEPARTURE", "ARRIVAL"}:
        raise ValueError("invalid airport direction")
    for side in ("departure_weather", "arrival_weather"):
        for kind in ("metar", "taf"):
            item = features.get(side, {}).get(kind, {})
            if not item or item.get("missing", True):
                continue
            if _aware(item["issue_time"]) > cutoff:
                raise ValueError("future weather issue")
            if item.get("received_time") and _aware(item["received_time"]) > cutoff:
                raise ValueError("future weather receipt")
    flow = features.get("planned_total_flow")
    if isinstance(flow, bool) or not isinstance(flow, (int, float)) or not math.isfinite(flow) or flow < 0:
        raise ValueError("planned_total_flow must be a finite nonnegative number")
    metar = features.get("departure_weather" if direction == "DEPARTURE" else "arrival_weather", {}).get("metar", {})
    weather = metar.get("features", {}) if not metar.get("missing", True) else {}
    hour = planned.astimezone(ZoneInfo("Asia/Shanghai")).hour
    return np.asarray((1.0, float(direction == "DEPARTURE"), float(_severe(features)),
                       float(bool(weather.get("precipitation"))),
                       float(bool(weather.get("thunderstorm"))),
                       float(metar.get("missing", True)), (float(flow) - 40) / 20,
                       math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)),
                      dtype=float)


def _target(row):
    value = row["labels"]["off_block_delay_min"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("delay label must be finite")
    return float(value >= THRESHOLD_MINUTES)


def brier_score(labels, probabilities):
    labels, probabilities = list(labels), list(probabilities)
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Brier inputs must have equal nonempty lengths")
    if any(value not in (0, 1) for value in labels):
        raise ValueError("Brier labels must be binary")
    if any(isinstance(value, bool) or not isinstance(value, (int, float, np.floating))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Brier probabilities must be finite values in [0,1]")
    return float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2))


class DelayRiskModel:
    def fit(self, rows):
        if not rows:
            raise ValueError("risk training rows are empty")
        matrix = np.stack([risk_features(row) for row in rows])
        labels = np.asarray([_target(row) for row in rows])
        prevalence = float(labels.mean())
        weights = np.zeros(matrix.shape[1])
        weights[0] = math.log((prevalence + .01) / (1 - prevalence + .01))
        for _ in range(STEPS):
            logits = np.clip(matrix @ weights, -20, 20)
            probabilities = 1 / (1 + np.exp(-logits))
            gradient = matrix.T @ (probabilities - labels) / len(labels)
            gradient[1:] += L2_PENALTY * weights[1:]
            weights -= LEARNING_RATE * gradient
        self.weights = weights
        self.prevalence = prevalence
        return self

    def predict_proba(self, rows):
        if not hasattr(self, "weights"):
            raise ValueError("risk model is not fitted")
        if not rows:
            return []
        matrix = np.stack([risk_features(row) for row in rows])
        logits = np.clip(matrix @ self.weights, -20, 20)
        return [float(value) for value in 1 / (1 + np.exp(-logits))]


def evaluate_rolling_risk(dataset):
    dataset = Path(dataset)
    manifest = verify_zggg_dataset(dataset)
    rows = [json.loads(line) for line in (dataset / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    reports = []
    for index, fold in enumerate(expanding_folds(rows), 1):
        model = DelayRiskModel().fit(fold.train_rows)
        labels = [_target(row) for row in fold.validation_rows]
        predictions = model.predict_proba(fold.validation_rows)
        constant = [model.prevalence] * len(labels)
        baseline = brier_score(labels, constant)
        candidate = brier_score(labels, predictions)
        reports.append({"fold": index, "train_rows": len(fold.train_rows),
                        "validation_rows": len(fold.validation_rows),
                        "train_excluded_rows": fold.train_audit["excluded_rows"],
                        "validation_dates": list(fold.validation_dates),
                        "train_prevalence": model.prevalence,
                        "validation_prevalence": float(np.mean(labels)),
                        "constant_brier": baseline, "risk_brier": candidate,
                        "brier_improvement": baseline - candidate})
    return {"dataset_manifest_hash": manifest["manifest_hash"],
            "evaluation_scope": "train_split_expanding_dates_only",
            "target": "off_block_delay_ge_30_minutes",
            "features": FEATURE_NAMES,
            "training": {"steps": STEPS, "learning_rate": LEARNING_RATE,
                         "l2_penalty": L2_PENALTY},
            "folds": reports,
            "improves_all_folds": all(fold["brier_improvement"] > 0 for fold in reports)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("risk report output must be a new file")
    report = evaluate_rolling_risk(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "improves_all_folds": report["improves_all_folds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
