import json
from datetime import datetime, timedelta

import pandas as pd

from tests.training.test_dataset import row, source
from training.dataset import DatasetConfig


def test_half_open_windows_count_boundaries_and_keep_total_consistent():
    from training.flow import build_flow_snapshots

    cutoff = pd.Timestamp("2025-05-01T00:00:00Z")
    rows = []
    events = ((0, "takeoff"), (14, "landing"), (15, "takeoff"),
              (29, "landing"), (30, "takeoff"), (59, "landing"),
              (60, "takeoff"))
    for suffix, (minute, direction) in enumerate(events):
        item = row(1, suffix)
        event = cutoff + timedelta(minutes=minute)
        if direction == "takeoff":
            item["计划起飞站四字码"] = "ZGGG"
            item["计划到达站四字码"] = "ZBAA"
            item["实际起飞时间"] = event
        else:
            item["计划起飞站四字码"] = "ZBAA"
            item["计划到达站四字码"] = "ZGGG"
            item["实际落地时间"] = event
        rows.append(item)

    snapshot = build_flow_snapshots(pd.DataFrame(rows), [cutoff.to_pydatetime()])[0]
    assert snapshot["labels"] == {
        "takeoff_15m": 1, "landing_15m": 1, "total_15m": 2,
        "takeoff_30m": 2, "landing_30m": 2, "total_30m": 4,
        "takeoff_60m": 3, "landing_60m": 3, "total_60m": 6,
    }


def test_features_use_plans_and_only_completed_actual_events():
    from training.flow import build_flow_snapshots

    cutoff = pd.Timestamp("2025-05-01T00:00:00Z")
    past = row(1)
    past["实际起飞时间"] = cutoff - timedelta(minutes=1)
    past["计划离港时间"] = cutoff - timedelta(minutes=1)
    future = row(1, 1)
    future["实际起飞时间"] = cutoff + timedelta(minutes=1)
    future["计划离港时间"] = cutoff + timedelta(minutes=1)

    record = build_flow_snapshots(pd.DataFrame([past, future]),
                                  [cutoff.to_pydatetime()])[0]
    features = record["features"]
    assert features["completed_takeoff_15m"] == 1
    assert features["planned_takeoff_15m"] == 1
    assert record["labels"]["takeoff_15m"] == 1
    encoded_features = json.dumps(features, ensure_ascii=False)
    assert "actual" not in encoded_features and "实际" not in encoded_features


def test_timezone_conversion_and_zero_traffic_snapshot():
    from training.flow import build_flow_snapshots

    records = build_flow_snapshots(pd.DataFrame(columns=[
        "计划起飞站四字码", "计划到达站四字码", "计划离港时间", "计划到港时间",
        "实际起飞时间", "实际落地时间",
    ]), [datetime(2025, 5, 1, 8, 0)], input_timezone="Asia/Shanghai")
    assert records[0]["features"]["prediction_cutoff_time"] == "2025-05-01T00:00:00Z"
    assert set(records[0]["labels"].values()) == {0}
    assert all(value == 0 for key, value in records[0]["features"].items()
               if key.startswith(("planned_", "completed_")))


def test_zggg_builder_writes_hashed_flow_splits(tmp_path):
    from training.zggg_dataset import build_zggg_dataset, verify_zggg_dataset

    dataset = tmp_path / "zggg"
    build_zggg_dataset(source(tmp_path, [row(day) for day in range(1, 13)]), dataset,
                       DatasetConfig(min_test_rows=1))
    manifest = verify_zggg_dataset(dataset)
    for split in ("train", "validation", "test"):
        path = dataset / f"flow_{split}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert manifest["flow_splits"][split]["rows"] == len(records)
        assert records
        assert all(item["features"]["airport"] == "ZGGG" for item in records)
