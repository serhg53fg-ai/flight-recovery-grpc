"""ZGGG flow features available at a prediction cutoff; no future labels."""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from .schema import ZGGG_AIRPORT

FLOW_HORIZONS = (15, 30, 60)
FLOW_SCHEMA_VERSION = 'zggg-airport-flow-v1'


def _utc(value, timezone_name):
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(timezone_name)
    return parsed.tz_convert('UTC').to_pydatetime()


def _events(frame, airport_column, time_column, timezone_name):
    values = []
    if airport_column not in frame or time_column not in frame:
        return ()
    for airport, value in zip(frame[airport_column], frame[time_column]):
        if str(airport).upper() != ZGGG_AIRPORT:
            continue
        try:
            parsed = _utc(value, timezone_name)
        except (ValueError, TypeError, OverflowError):
            continue
        if parsed is not None:
            values.append(parsed)
    return tuple(sorted(values))


def _count(events, start, end):
    return bisect_left(events, end) - bisect_left(events, start)


def build_flow_features(frame: pd.DataFrame, cutoffs: Iterable[datetime],
                        input_timezone: str = 'Asia/Shanghai',
                        horizons: tuple[int, ...] = FLOW_HORIZONS) -> list[dict]:
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError('flow horizons must be positive')
    planned_takeoffs = _events(frame, '计划起飞站四字码', '计划离港时间', input_timezone)
    planned_landings = _events(frame, '计划到达站四字码', '计划到港时间', input_timezone)
    completed_takeoffs = _events(frame, '计划起飞站四字码', '实际起飞时间', input_timezone)
    completed_landings = _events(frame, '计划到达站四字码', '实际落地时间', input_timezone)
    result = []
    for value in cutoffs:
        cutoff = _utc(value, input_timezone)
        if cutoff is None:
            raise ValueError('flow cutoff is missing')
        features = {'airport': ZGGG_AIRPORT,
                    'prediction_cutoff_time': cutoff.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z'),
                    'flow_schema_version': FLOW_SCHEMA_VERSION}
        for minutes in horizons:
            future_end = cutoff + timedelta(minutes=minutes)
            past_start = cutoff - timedelta(minutes=minutes)
            planned_out = _count(planned_takeoffs, cutoff, future_end)
            planned_in = _count(planned_landings, cutoff, future_end)
            completed_out = _count(completed_takeoffs, past_start, cutoff)
            completed_in = _count(completed_landings, past_start, cutoff)
            features.update({
                f'planned_takeoff_{minutes}m': planned_out,
                f'planned_landing_{minutes}m': planned_in,
                f'planned_total_{minutes}m': planned_out + planned_in,
                f'completed_takeoff_{minutes}m': completed_out,
                f'completed_landing_{minutes}m': completed_in,
                f'completed_total_{minutes}m': completed_out + completed_in,
            })
        result.append(features)
    return result
