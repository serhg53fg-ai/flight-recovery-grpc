"""Deterministic train-only feature processing."""

from __future__ import annotations

from datetime import datetime
import math
import re

import numpy as np


CATEGORICAL = ("flight_number", "tail_number", "aircraft_type", "flight_nature", "departure_airport", "arrival_airport")
NUMERIC = ("planned_distance_miles", "planned_flight_minutes", "planned_takeoff_count", "planned_landing_count", "planned_total_flow")


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _metar(text):
    text = text if isinstance(text, str) else ""
    wind = re.search(r"(?:\d{3}|VRB)(\d{2,3})KT", text)
    visibility = re.search(r"\s(\d{4})\s", text)
    temperature = re.search(r"\s(M?\d{2})/(?:M?\d{2})\s", text)
    pressure = re.search(r"[QA](\d{4})", text)
    def signed(value): return -float(value[1:]) if value.startswith("M") else float(value)
    return [float(wind.group(1)) if wind else 0.0,
            float(visibility.group(1)) if visibility else 10000.0,
            signed(temperature.group(1)) if temperature else 0.0,
            float(pressure.group(1)) if pressure else 0.0]


class FeaturePipeline:
    def __init__(self):
        self.categories = {}
        self.medians = {}
        self.fitted = False

    def fit(self, rows):
        if not rows: raise ValueError("training rows are empty")
        self.categories = {name: {value: index + 1 for index, value in enumerate(sorted({str(row["features"].get(name, "")) for row in rows}))} for name in CATEGORICAL}
        for name in NUMERIC:
            values = sorted(value for row in rows if (value := _number(row["features"].get(name))) is not None)
            self.medians[name] = float(np.median(values)) if values else 0.0
        self.fitted = True
        return self

    def transform(self, rows):
        if not self.fitted: raise ValueError("feature pipeline is not fitted")
        matrix = []
        for row in rows:
            feature = row["features"]
            timestamp = datetime.fromisoformat(feature["planned_off_block"].replace("Z", "+00:00"))
            values = [float(self.categories[name].get(str(feature.get(name, "")), 0)) for name in CATEGORICAL]
            values += [self.medians[name] if _number(feature.get(name)) is None else _number(feature.get(name)) for name in NUMERIC]
            values += [float(timestamp.month), float(timestamp.weekday()), float(timestamp.hour)]
            matrix.append(values)
        return np.asarray(matrix, dtype=float)

    def state(self):
        if not self.fitted: raise ValueError("feature pipeline is not fitted")
        return {"categories": self.categories, "medians": self.medians, "version": "features-v1"}


class ZgggWeatherFeaturePipeline(FeaturePipeline):
    """Append only the ZGGG METAR available at the prediction cutoff."""

    WEATHER_NUMERIC = ("report_age_minutes", "wind_speed_kt", "visibility_m",
                       "ceiling_ft", "temperature_c", "pressure_hpa")
    WEATHER_BOOLEAN = ("precipitation", "thunderstorm", "cumulonimbus", "cavok")

    def transform(self, rows):
        original = super().transform(rows)
        added = []
        for row in rows:
            features = row["features"]
            departure = features.get("departure_airport") == "ZGGG"
            arrival = features.get("arrival_airport") == "ZGGG"
            if departure == arrival:
                raise ValueError("weather model requires one ZGGG endpoint")
            side = "departure_weather" if departure else "arrival_weather"
            report = features.get(side, {}).get("metar", {})
            cutoff = datetime.fromisoformat(features["prediction_cutoff_time"].replace("Z", "+00:00"))
            if not report.get("missing", True):
                issue = datetime.fromisoformat(str(report["issue_time"]).replace("Z", "+00:00"))
                if issue > cutoff:
                    raise ValueError("future weather report is not a valid feature")
            values = report.get("features", {}) if not report.get("missing", True) else {}
            numeric = []
            for name in self.WEATHER_NUMERIC:
                value = _number(report.get(name) if name == "report_age_minutes" else values.get(name))
                numeric.append(value if value is not None else -1.0)
            added.append([
                float(bool(report.get("missing", True))), float(bool(report.get("stale", False))),
                *numeric,
                *[float(bool(values.get(name, False))) for name in self.WEATHER_BOOLEAN],
            ])
        return np.hstack((original, np.asarray(added, dtype=float)))

    def state(self):
        return {**super().state(), "version": "zggg-cutoff-weather-features-v1"}
