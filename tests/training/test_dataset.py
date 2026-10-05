import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from training.dataset import DatasetConfig, build_dataset, verify_dataset
from training.schema import LABEL_FIELDS, PROHIBITED_FEATURE_COLUMNS


def row(day: int, suffix: int = 0):
    planned = datetime(2025, 5, day, 8, suffix)
    off_block = planned - timedelta(minutes=5)
    takeoff = off_block + timedelta(minutes=20)
    landing = takeoff + timedelta(minutes=100)
    on_block = landing + timedelta(minutes=10)
    return {
        "机尾号": f"B{day:03d}{suffix}", "航班号": f"CZ{day:04d}", "机型": "A320", "性质": "J",
        "计划起飞站四字码": "ZGGG", "实际起飞站四字码": "ZGGG",
        "计划到达站四字码": "ZBAA", "实际到达站四字码": "ZBAA",
        "计划离港时间": planned, "计划到港时间": planned + timedelta(minutes=140),
        "实际离港时间": off_block, "实际起飞时间": takeoff,
        "实际落地时间": landing, "实际到港时间": on_block,
        "计划地面航程_Mile": 1000, "实际航程_Mile": 9999,
        "计划航段时间\n（24年同航季平均值）": 120,
        "实际航段时间\n（24年同航季平均值）": 999,
        "计划起飞数": 10, "计划降落数": 11, "计划总流量": 21,
        "实际起飞数": 999, "实际降落数": 999, "实际总流量": 1998,
        "起飞站METAR": "METAR ZGGG 010800Z 18005KT 9999 CLR 20/10 Q1013=",
        "到达站METAR": "METAR ZBAA 011000Z 20004KT 8000 FEW020 18/08 Q1015=",
    }


def source(tmp_path: Path, rows) -> Path:
    path = tmp_path / "history.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_builds_duration_labels_and_excludes_every_prohibited_feature(tmp_path):
    manifest = build_dataset(
        source(tmp_path, [row(day) for day in range(1, 13)]),
        tmp_path / "dataset", DatasetConfig(min_test_rows=1, minimum_dates=10),
    )
    records = sum((read_jsonl(tmp_path / "dataset" / f"{split}.jsonl") for split in ("train", "validation", "test")), [])
    assert manifest.accepted_rows == 12
    assert records[0]["labels"] == {
        "off_block_delay_min": -5.0, "taxi_out_min": 20.0,
        "airborne_min": 100.0, "taxi_in_min": 10.0,
    }
    assert set(records[0]["labels"]) == set(LABEL_FIELDS)
    assert not set(records[0]["features"]).intersection(PROHIBITED_FEATURE_COLUMNS)
    assert "actual_distance" not in json.dumps(records)
    assert "arrival_metar" not in records[0]["features"]
    assert "departure_metar" not in records[0]["features"]


def test_rejects_invalid_order_and_duplicate_rows_with_audit_counts(tmp_path):
    valid = row(1)
    duplicate = dict(valid)
    invalid = row(2)
    invalid["实际落地时间"] = invalid["实际起飞时间"] - timedelta(minutes=1)
    manifest = build_dataset(
        source(tmp_path, [valid, duplicate, invalid]), tmp_path / "dataset",
        DatasetConfig(min_test_rows=1, minimum_dates=1),
    )
    audit = json.loads((tmp_path / "dataset/audit.json").read_text())
    assert manifest.accepted_rows == 1
    assert audit["rejections"]["duplicate"] == 1
    assert audit["rejections"]["invalid_time_order"] == 1


def test_date_splits_do_not_overlap_and_are_deterministic(tmp_path):
    input_path = source(tmp_path, [row(day, suffix) for day in range(1, 13) for suffix in range(2)])
    first = build_dataset(input_path, tmp_path / "one", DatasetConfig(min_test_rows=1))
    second = build_dataset(input_path, tmp_path / "two", DatasetConfig(min_test_rows=1))
    assert first.manifest_hash == second.manifest_hash
    split_dates = [set(first.splits[name].dates) for name in ("train", "validation", "test")]
    assert not split_dates[0] & split_dates[1]
    assert not split_dates[0] & split_dates[2]
    assert not split_dates[1] & split_dates[2]
    assert sum(first.splits[name].rows for name in first.splits) == 24


def test_split_dates_follow_configured_local_timezone(tmp_path):
    rows = [row(day) for day in range(1, 13)]
    extra = row(7, 1)
    extra["计划离港时间"] = "2025-05-07T16:30:00Z"
    extra["计划到港时间"] = "2025-05-07T18:50:00Z"
    extra["实际离港时间"] = "2025-05-07T16:25:00Z"
    extra["实际起飞时间"] = "2025-05-07T16:45:00Z"
    extra["实际落地时间"] = "2025-05-07T18:25:00Z"
    extra["实际到港时间"] = "2025-05-07T18:35:00Z"
    manifest = build_dataset(source(tmp_path, rows + [extra]), tmp_path / "dataset",
                             DatasetConfig(min_test_rows=1))
    assert sum("2025-05-08" in split.dates for split in manifest.splits.values()) == 1
    assert verify_dataset(tmp_path / "dataset")["manifest_hash"] == manifest.manifest_hash


def test_dataset_verification_rejects_tampered_split(tmp_path):
    build_dataset(source(tmp_path, [row(day) for day in range(1, 13)]), tmp_path / "dataset",
                  DatasetConfig(min_test_rows=1))
    (tmp_path / "dataset/test.jsonl").write_text(
        (tmp_path / "dataset/train.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        verify_dataset(tmp_path / "dataset")


def test_dataset_verification_rejects_reversed_split_chronology(tmp_path):
    dataset = tmp_path / "dataset"
    build_dataset(source(tmp_path, [row(day) for day in range(1, 13)]), dataset,
                  DatasetConfig(min_test_rows=1))
    manifest = json.loads((dataset / "manifest.json").read_text())
    train_text = (dataset / "train.jsonl").read_text()
    test_text = (dataset / "test.jsonl").read_text()
    (dataset / "train.jsonl").write_text(test_text)
    (dataset / "test.jsonl").write_text(train_text)
    manifest["splits"]["train"], manifest["splits"]["test"] = (
        manifest["splits"]["test"], manifest["splits"]["train"])
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    import hashlib
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="chronological"):
        verify_dataset(dataset)


def test_small_dataset_builds_but_blocks_publication(tmp_path):
    manifest = build_dataset(
        source(tmp_path, [row(day) for day in range(1, 6)]), tmp_path / "dataset",
        DatasetConfig(min_test_rows=500, minimum_dates=10),
    )
    assert manifest.publishable is False
    assert set(manifest.publication_blocks) == {"insufficient_dates", "insufficient_test_rows"}
