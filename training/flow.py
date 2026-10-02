"""Leakage-safe airport flow windows for ZGGG."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from typing import Iterable

import pandas as pd

from .flow_features import FLOW_HORIZONS, FLOW_SCHEMA_VERSION, _utc, _events, _count, build_flow_features


def build_flow_snapshots(frame: pd.DataFrame, cutoffs: Iterable[datetime],
                         input_timezone: str = "Asia/Shanghai",
                         horizons: tuple[int, ...] = FLOW_HORIZONS):
    """Create flow records whose features are knowable at each cutoff."""
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("flow horizons must be positive")
    completed_takeoffs = _events(frame, "计划起飞站四字码", "实际起飞时间", input_timezone)
    completed_landings = _events(frame, "计划到达站四字码", "实际落地时间", input_timezone)
    feature_rows = build_flow_features(frame, cutoffs, input_timezone, horizons)
    records = []
    for features in feature_rows:
        cutoff = _utc(features['prediction_cutoff_time'], input_timezone)
        labels = {}
        for minutes in horizons:
            future_end = cutoff + timedelta(minutes=minutes)
            actual_out = _count(completed_takeoffs, cutoff, future_end)
            actual_in = _count(completed_landings, cutoff, future_end)
            labels.update({
                f"takeoff_{minutes}m": actual_out,
                f"landing_{minutes}m": actual_in,
                f"total_{minutes}m": actual_out + actual_in,
            })
        identity = json.dumps({"features": features, "labels": labels},
                              sort_keys=True, separators=(",", ":"))
        records.append({"record_id": hashlib.sha256(identity.encode()).hexdigest()[:32],
                        "features": features, "labels": labels})
    return records
