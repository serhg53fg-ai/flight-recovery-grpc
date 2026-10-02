"""Conservative, cutoff-safe weather correction to a historical flight baseline."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from .baselines import HistoricalMedianBaseline
from .schema import LABEL_FIELDS


def _date(row, timezone):
    value = datetime.fromisoformat(row['features']['planned_off_block'].replace('Z', '+00:00'))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('planned_off_block must include timezone')
    return value.astimezone(timezone).date().isoformat()


def _weather_key(row):
    features = row['features']
    departure = features.get('departure_airport') == 'ZGGG'
    arrival = features.get('arrival_airport') == 'ZGGG'
    if departure == arrival:
        raise ValueError('weather correction requires one ZGGG endpoint')
    side = 'departure_weather' if departure else 'arrival_weather'
    report = features.get(side, {}).get('metar', {})
    cutoff = datetime.fromisoformat(features['prediction_cutoff_time'].replace('Z', '+00:00'))
    if not report.get('missing', True):
        issue = datetime.fromisoformat(report['issue_time'].replace('Z', '+00:00'))
        if issue > cutoff:
            raise ValueError('future weather report is not a valid feature')
    if report.get('missing', True) or report.get('stale', False):
        return None
    values = report.get('features', {})
    adverse = bool(values.get('precipitation') or values.get('thunderstorm')
                   or values.get('cumulonimbus'))
    return ('departure' if departure else 'arrival', 'adverse' if adverse else 'normal')


class WeatherResidualBaseline:
    """Fit corrections on past-date out-of-fold errors, never in-sample errors."""

    def __init__(self, *, min_bucket_samples=100, shrink_samples=200,
                 max_adjustment_min=10, initial_dates=3,
                 input_timezone='Asia/Shanghai', adverse_only=False):
        if min_bucket_samples < 1 or shrink_samples < 0 or max_adjustment_min <= 0 or initial_dates < 1:
            raise ValueError('invalid residual correction parameters')
        self.min_bucket_samples = min_bucket_samples
        self.shrink_samples = shrink_samples
        self.max_adjustment_min = max_adjustment_min
        self.initial_dates = initial_dates
        self.timezone = ZoneInfo(input_timezone)
        self.adverse_only = adverse_only

    def fit(self, rows):
        grouped = defaultdict(list)
        for row in rows:
            grouped[_date(row, self.timezone)].append(row)
        dates = sorted(grouped)
        if len(dates) <= self.initial_dates:
            raise ValueError('insufficient dates for out-of-fold residuals')
        errors = defaultdict(lambda: defaultdict(list))
        past = [row for date in dates[:self.initial_dates] for row in grouped[date]]
        for date in dates[self.initial_dates:]:
            current = grouped[date]
            predictions = HistoricalMedianBaseline().fit(past).predict(current)
            for row, prediction in zip(current, predictions):
                key = _weather_key(row)
                if key is not None:
                    for field in LABEL_FIELDS:
                        errors[key][field].append(float(row['labels'][field]) - prediction[field])
            past.extend(current)
        self.adjustments = {}
        for key, fields in errors.items():
            if self.adverse_only and key[1] != 'adverse':
                continue
            if len(fields[LABEL_FIELDS[0]]) < self.min_bucket_samples:
                continue
            weight = len(fields[LABEL_FIELDS[0]]) / (len(fields[LABEL_FIELDS[0]]) + self.shrink_samples)
            self.adjustments[key] = {field: float(np.clip(np.median(values) * weight,
                                                         -self.max_adjustment_min,
                                                         self.max_adjustment_min))
                                     for field, values in fields.items()}
        self.base = HistoricalMedianBaseline().fit(rows)
        return self

    def predict(self, rows):
        predictions = self.base.predict(rows)
        for row, prediction in zip(rows, predictions):
            corrections = self.adjustments.get(_weather_key(row), {})
            for field in LABEL_FIELDS:
                prediction[field] += corrections.get(field, 0.0)
            for field in ('taxi_out_min', 'airborne_min', 'taxi_in_min'):
                prediction[field] = max(0.0, prediction[field])
        return predictions
