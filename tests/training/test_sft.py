import json
from datetime import datetime, timedelta
from pathlib import Path
import pytest

from training.sft import export_sft
from training.dataset import DatasetConfig, build_dataset
from tests.training.test_dataset import row, source


def test_exports_splits_without_leakage_or_cross_split_records(tmp_path):
    dataset = tmp_path / "dataset"
    manifest = build_dataset(source(tmp_path, [row(day) for day in range(1, 13)]), dataset,
                             DatasetConfig(min_test_rows=1))
    result = export_sft(dataset, tmp_path / "sft")
    assert result["dataset_manifest_hash"] == manifest.manifest_hash
    for split in ("train", "validation", "test"):
        text = (tmp_path / "sft" / f"{split}.jsonl").read_text()
        assert "actual_distance" not in text
        assert "arrival_metar" not in text


def test_sft_export_rejects_tampered_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    build_dataset(source(tmp_path, [row(day) for day in range(1, 13)]), dataset,
                  DatasetConfig(min_test_rows=1))
    (dataset / "test.jsonl").write_text((dataset / "train.jsonl").read_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        export_sft(dataset, tmp_path / "sft")


def test_zggg_sft_records_weather_scope_and_split_hashes(tmp_path):
    from training.zggg_dataset import build_zggg_dataset
    from training.prompt import PROMPT_VERSION
    dataset = tmp_path / "zggg"
    rows = [row(day) for day in range(1, 13)]
    for day, item in enumerate(rows, 1):
        issue_day = 30 if day == 1 else day - 1
        item["起飞站METAR"] = f"METAR ZGGG {issue_day:02d}2350Z 18005KT 4000 -RA BKN010 20/19 Q1013="
    manifest = build_zggg_dataset(source(tmp_path, rows), dataset,
                                  DatasetConfig(min_test_rows=1))
    result = export_sft(dataset, tmp_path / "sft")
    assert result["dataset_manifest_hash"] == manifest.manifest_hash
    assert result["prompt_version"] == PROMPT_VERSION
    assert result["airport_scope"] == "ZGGG"
    assert result["weather_feature_version"] == "aviation-weather-v1"
    text = (tmp_path / "sft/train.jsonl").read_text()
    assert "visibility_m" in text and "raw_text" not in text


def test_zggg_sft_excludes_labels_unfinished_at_next_split(tmp_path):
    from training.zggg_dataset import build_zggg_dataset
    dataset = tmp_path / "zggg"
    rows = [row(day) for day in range(1, 13)]
    late = rows[7]
    start = datetime(2025, 5, 8, 23, 0)
    late["计划离港时间"] = start
    late["计划到港时间"] = start + timedelta(minutes=140)
    late["实际离港时间"] = start + timedelta(minutes=5)
    late["实际起飞时间"] = start + timedelta(minutes=25)
    late["实际落地时间"] = start + timedelta(minutes=125)
    late["实际到港时间"] = start + timedelta(minutes=135)
    build_zggg_dataset(source(tmp_path, rows), dataset, DatasetConfig(min_test_rows=1))
    raw_train = [json.loads(x) for x in (dataset / "train.jsonl").read_text().splitlines()]
    assert len(raw_train) == 8

    manifest = export_sft(dataset, tmp_path / "sft")
    safe_train = [json.loads(x) for x in (tmp_path / "sft/train.jsonl").read_text().splitlines()]

    assert len(safe_train) == 7
    assert all(late["机尾号"] not in item["prompt"] for item in safe_train)
    assert manifest["temporal_fit_filter"]["train"]["excluded_rows"] == 1
    assert manifest["temporal_fit_filter"]["train"]["reasons"] == {"label_not_available_at_boundary": 1}


def test_zggg_v2_sft_manifest_keeps_weather_version(tmp_path):
    from training.zggg_dataset import build_zggg_dataset
    dataset = tmp_path / "zggg"
    build_zggg_dataset(source(tmp_path, [row(day) for day in range(1, 13)]),
                       dataset, DatasetConfig(min_test_rows=1),
                       weather_feature_version="aviation-weather-v2")
    result = export_sft(dataset, tmp_path / "sft")
    assert result["weather_feature_version"] == "aviation-weather-v2"
