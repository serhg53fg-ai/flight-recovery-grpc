"""CPU flight-duration baselines with a common prediction contract."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np

from .features import FeaturePipeline, ZgggWeatherFeaturePipeline
from .schema import LABEL_FIELDS


def _median_labels(rows):
    return {name: float(np.median([row["labels"][name] for row in rows])) for name in LABEL_FIELDS}


def _prediction(record_id, values):
    return {"record_id": record_id, **{name: float(values[name]) for name in LABEL_FIELDS}}


class ScheduleZeroBaseline:
    def fit(self, rows):
        values = _median_labels(rows)
        self.taxi_out = values["taxi_out_min"]
        self.taxi_in = values["taxi_in_min"]
        return self
    def predict(self, rows):
        result = []
        for row in rows:
            features = row["features"]
            off = datetime.fromisoformat(features["planned_off_block"].replace("Z", "+00:00"))
            on = datetime.fromisoformat(features["planned_on_block"].replace("Z", "+00:00"))
            block = (on - off).total_seconds() / 60
            result.append(_prediction(row["record_id"], {
                "off_block_delay_min": 0.0, "taxi_out_min": self.taxi_out,
                "airborne_min": max(0.0, block - self.taxi_out - self.taxi_in),
                "taxi_in_min": self.taxi_in,
            }))
        return result


def _keys(row):
    f = row["features"]
    hour = datetime.fromisoformat(f["planned_off_block"].replace("Z", "+00:00")).hour
    route = (f.get("departure_airport"), f.get("arrival_airport"))
    return (("route-hour", *route, hour), ("route", *route),
            ("airport-hour", f.get("departure_airport"), hour),
            ("airport", f.get("departure_airport")))


class HistoricalMedianBaseline:
    def fit(self, rows):
        grouped = defaultdict(list)
        for row in rows:
            for key in _keys(row): grouped[key].append(row)
        self.groups = {key: _median_labels(values) for key, values in grouped.items()}
        self.global_values = _median_labels(rows)
        return self
    def predict(self, rows):
        result = []
        for row in rows:
            values = next((self.groups[key] for key in _keys(row) if key in self.groups), self.global_values)
            result.append(_prediction(row["record_id"], values))
        return result


class GradientBoostingBaseline:
    def __init__(self, random_state=42, feature_pipeline_cls=FeaturePipeline):
        self.random_state = random_state
        self.feature_pipeline_cls = feature_pipeline_cls
    def fit(self, rows):
        self.features = self.feature_pipeline_cls().fit(rows)
        matrix = self.features.transform(rows)
        self.models = {}
        for name in LABEL_FIELDS:
            self.models[name] = _HistogramStumpBoost().fit(
                matrix, np.asarray([row["labels"][name] for row in rows], dtype=float)
            )
        return self
    def predict(self, rows):
        matrix = self.features.transform(rows)
        outputs = {name: model.predict(matrix) for name, model in self.models.items()}
        return [_prediction(row["record_id"], {name: outputs[name][index] for name in LABEL_FIELDS}) for index, row in enumerate(rows)]


class ZgggWeatherGradientBoostingBaseline(GradientBoostingBaseline):
    def __init__(self, random_state=42):
        super().__init__(random_state, ZgggWeatherFeaturePipeline)


class _HistogramStumpBoost:
    def __init__(self, rounds=40, learning_rate=0.1, bins=16):
        self.rounds, self.learning_rate, self.bins = rounds, learning_rate, bins

    def fit(self, matrix, target):
        self.initial = float(np.mean(target))
        prediction = np.full(len(target), self.initial)
        self.stumps = []
        for _ in range(self.rounds):
            residual = target - prediction
            best = None
            for feature in range(matrix.shape[1]):
                values = matrix[:, feature]
                for threshold in np.unique(np.quantile(values, np.linspace(0, 1, self.bins + 1)[1:-1])):
                    left = values <= threshold
                    if not left.any() or left.all(): continue
                    left_value, right_value = float(residual[left].mean()), float(residual[~left].mean())
                    error = float(((residual[left] - left_value) ** 2).sum() + ((residual[~left] - right_value) ** 2).sum())
                    candidate = (error, feature, float(threshold), left_value, right_value)
                    if best is None or candidate < best: best = candidate
            if best is None: break
            _, feature, threshold, left_value, right_value = best
            self.stumps.append((feature, threshold, left_value, right_value))
            prediction += self.learning_rate * np.where(matrix[:, feature] <= threshold, left_value, right_value)
        return self

    def predict(self, matrix):
        prediction = np.full(matrix.shape[0], self.initial)
        for feature, threshold, left_value, right_value in self.stumps:
            prediction += self.learning_rate * np.where(matrix[:, feature] <= threshold, left_value, right_value)
        return prediction
