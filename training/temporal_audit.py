"""Audit fold-time feature and label availability in an existing flight dataset."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import math
from zoneinfo import ZoneInfo


def _time(value):
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    else:
        raise ValueError('timestamp is missing')
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('timestamp must include timezone')
    return parsed.astimezone(timezone.utc)


def _classification(row, boundary, seen):
    reasons, limits = set(), set()
    if not isinstance(row, dict) or not isinstance(row.get('features'), dict):
        return {'invalid_row'}, limits, None, False
    features = row['features']
    record_id = row.get('record_id')
    if not record_id or record_id in seen:
        reasons.add('duplicate_or_missing_record_id')
    else:
        seen.add(record_id)
    try:
        planned = _time(features['planned_off_block'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return reasons | {'invalid_planned_time'}, limits, None, False
    if planned >= boundary:
        reasons.add('flight_not_before_boundary')
    cutoff = planned
    if features.get('prediction_cutoff_time') is not None:
        try:
            cutoff = _time(features['prediction_cutoff_time'])
            if cutoff > planned:
                reasons.add('future_prediction_cutoff')
        except (ValueError, TypeError, OverflowError):
            reasons.add('invalid_prediction_cutoff')
    for side in ('departure_weather', 'arrival_weather'):
        bundle = features.get(side) or {}
        if not isinstance(bundle, dict):
            reasons.add('invalid_weather_bundle')
            continue
        for kind in ('metar', 'taf'):
            report = bundle.get(kind) or {}
            if not isinstance(report, dict):
                reasons.add('invalid_weather_bundle')
                continue
            if report.get('missing', True):
                continue
            try:
                if _time(report['issue_time']) > cutoff:
                    reasons.add('future_weather_issue')
            except (KeyError, TypeError, ValueError, OverflowError):
                reasons.add('invalid_weather_issue')
            if report.get('received_time') is None:
                limits.add('weather_received_at_unknown')
            else:
                try:
                    if _time(report['received_time']) > cutoff:
                        reasons.add('future_weather_receipt')
                except (TypeError, ValueError, OverflowError):
                    reasons.add('invalid_weather_receipt')
    available = features.get('feature_available_at')
    if available is None:
        limits.add('feature_available_at_unknown')
    else:
        try:
            if _time(available) > cutoff:
                reasons.add('future_feature_availability')
        except (TypeError, ValueError, OverflowError):
            reasons.add('invalid_feature_availability')
    labels = row.get('labels') or {}
    names = ('off_block_delay_min', 'taxi_out_min', 'airborne_min', 'taxi_in_min')
    try:
        durations = [float(labels[name]) for name in names]
        if not all(math.isfinite(value) for value in durations):
            raise ValueError('nonfinite label')
        if any(value < 0 for value in durations[1:]):
            reasons.add('invalid_label_duration')
        finished = planned + timedelta(minutes=sum(durations))
        if finished > boundary:
            reasons.add('label_not_available_at_boundary')
        extreme = abs(durations[0]) > 360
    except (KeyError, TypeError, ValueError, OverflowError):
        reasons.add('invalid_label')
        extreme = False
    received = row.get('label_received_at')
    if received is None:
        limits.add('label_received_at_unknown')
    else:
        try:
            if _time(received) > boundary:
                reasons.add('label_not_received_at_boundary')
        except (TypeError, ValueError, OverflowError):
            reasons.add('invalid_label_receipt')
    return reasons, limits, planned, extreme


def audit_temporal_rows(rows: list[dict], cutoff: str, *, input_timezone='Asia/Shanghai') -> dict:
    """Classify every row against the first instant of a validation fold.

    Unknown receipt times remain explicit assumptions; an event-time label can
    be eligible only after reconstructed actual on-block time.
    """
    boundary = _time(cutoff)
    zone = ZoneInfo(input_timezone)
    seen, reasons, limitations, dates = set(), Counter(), Counter(), Counter()
    eligible = extreme = 0
    for row in rows:
        failures, unknowns, planned, outlier = _classification(row, boundary, seen)
        for reason in failures:
            reasons[reason] += 1
        for limitation in unknowns:
            limitations[limitation] += 1
        if planned is not None:
            dates[planned.astimezone(zone).date().isoformat()] += 1
        extreme += int(outlier)
        eligible += int(not failures)
    return {'cutoff': boundary.isoformat(), 'rows': len(rows),
            'eligible_rows': eligible, 'excluded_rows': len(rows) - eligible,
            'reasons': dict(sorted(reasons.items())),
            'limitations': dict(sorted(limitations.items())),
            'extreme_delay_rows': extreme, 'rows_by_local_date': dict(sorted(dates.items()))}


def eligible_training_rows(rows: list[dict], cutoff: str) -> tuple[dict, ...]:
    boundary, seen = _time(cutoff), set()
    return tuple(row for row in rows if not _classification(row, boundary, seen)[0])
