"""Build audited, deterministic date-split training data from history."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import FEATURE_COLUMNS, LABEL_FIELDS, SCHEMA_VERSION, TARGET_COLUMNS


@dataclass(frozen=True)
class DatasetConfig:
    input_timezone: str = "Asia/Shanghai"
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    minimum_dates: int = 10
    min_test_rows: int = 500
    minimum_off_block_delay_min: float = -360
    maximum_off_block_delay_min: float = 1440
    maximum_taxi_min: float = 300
    maximum_airborne_min: float = 1440


@dataclass(frozen=True)
class DatasetSplit:
    rows: int
    dates: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class AuditReport:
    source_rows: int
    accepted_rows: int
    rejections: dict[str, int]
    date_min: str | None
    date_max: str | None
    missing_rates: dict[str, float]


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    source_sha256: str
    accepted_rows: int
    splits: dict[str, DatasetSplit]
    publishable: bool
    publication_blocks: tuple[str, ...]
    manifest_hash: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _utc(value: Any, timezone: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("missing timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(timezone)
    return parsed.tz_convert("UTC")


def _local_date(value: Any, timezone: str) -> str:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("missing timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(timezone)
    else:
        parsed = parsed.tz_convert(timezone)
    return parsed.date().isoformat()


def _minutes(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    return round((later - earlier).total_seconds() / 60.0, 6)


def _record(raw: pd.Series, config: DatasetConfig) -> tuple[dict[str, Any], str]:
    planned_off = _utc(raw["计划离港时间"], config.input_timezone)
    planned_on = _utc(raw["计划到港时间"], config.input_timezone)
    actual = {name: _utc(raw[source], config.input_timezone) for source, name in TARGET_COLUMNS.items()}
    if planned_on <= planned_off:
        raise ValueError("invalid_planned_order")
    if not (actual["actual_off_block"] <= actual["actual_takeoff"] <= actual["actual_landing"] <= actual["actual_on_block"]):
        raise ValueError("invalid_time_order")
    labels = {
        "off_block_delay_min": _minutes(actual["actual_off_block"], planned_off),
        "taxi_out_min": _minutes(actual["actual_takeoff"], actual["actual_off_block"]),
        "airborne_min": _minutes(actual["actual_landing"], actual["actual_takeoff"]),
        "taxi_in_min": _minutes(actual["actual_on_block"], actual["actual_landing"]),
    }
    if not config.minimum_off_block_delay_min <= labels["off_block_delay_min"] <= config.maximum_off_block_delay_min:
        raise ValueError("out_of_range")
    if labels["taxi_out_min"] > config.maximum_taxi_min or labels["taxi_in_min"] > config.maximum_taxi_min or labels["airborne_min"] > config.maximum_airborne_min:
        raise ValueError("out_of_range")
    features = {target: _json(raw.get(source)) for source, target in FEATURE_COLUMNS.items()}
    features["planned_off_block"] = planned_off.isoformat().replace("+00:00", "Z")
    features["planned_on_block"] = planned_on.isoformat().replace("+00:00", "Z")
    identity = json.dumps({"features": features, "labels": labels}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    record_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return {"record_id": record_id, "features": features, "labels": labels}, _local_date(raw["计划离港时间"], config.input_timezone)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for record in records)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_dataset(source: Path, output: Path, config: DatasetConfig = DatasetConfig()) -> DatasetManifest:
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file() or source.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError("source must be an existing Excel file")
    if not 0 < config.train_fraction < 1 or not 0 < config.validation_fraction < 1 or config.train_fraction + config.validation_fraction >= 1:
        raise ValueError("invalid split fractions")
    frame = pd.read_excel(source)
    required = set(FEATURE_COLUMNS) | set(TARGET_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("source is missing required columns")
    missing_rates = {column: round(float(frame[column].isna().mean()), 6) for column in sorted(required)}
    records_by_date: list[tuple[dict[str, Any], str]] = []
    rejected: dict[str, int] = {}
    seen: set[str] = set()
    for _, raw in frame.iterrows():
        try:
            record, date = _record(raw, config)
            if record["record_id"] in seen:
                raise ValueError("duplicate")
            seen.add(record["record_id"])
            records_by_date.append((record, date))
        except (ValueError, TypeError, OverflowError, KeyError) as error:
            reason = str(error) if str(error) in {"duplicate", "invalid_time_order", "invalid_planned_order", "out_of_range"} else "missing_or_invalid"
            rejected[reason] = rejected.get(reason, 0) + 1
    records_by_date.sort(key=lambda item: (item[1], item[0]["record_id"]))
    dates = sorted({date for _, date in records_by_date})
    if dates:
        train_count = max(1, int(len(dates) * config.train_fraction))
        validation_count = max(1, int(len(dates) * config.validation_fraction)) if len(dates) >= 3 else 0
        if train_count + validation_count >= len(dates):
            validation_count = max(0, len(dates) - train_count - 1)
    else:
        train_count = validation_count = 0
    split_dates = {
        "train": dates[:train_count],
        "validation": dates[train_count:train_count + validation_count],
        "test": dates[train_count + validation_count:],
    }
    output.mkdir(parents=True, exist_ok=True)
    splits: dict[str, DatasetSplit] = {}
    for name, assigned_dates in split_dates.items():
        assigned = [record for record, date in records_by_date if date in set(assigned_dates)]
        digest = _atomic_jsonl(output / f"{name}.jsonl", assigned)
        splits[name] = DatasetSplit(len(assigned), tuple(assigned_dates), digest)
    audit = AuditReport(
        source_rows=len(frame), accepted_rows=len(records_by_date), rejections=dict(sorted(rejected.items())),
        date_min=dates[0] if dates else None, date_max=dates[-1] if dates else None,
        missing_rates=missing_rates,
    )
    _atomic_json(output / "audit.json", asdict(audit))
    blocks = []
    if len(dates) < config.minimum_dates:
        blocks.append("insufficient_dates")
    if splits["test"].rows < config.min_test_rows:
        blocks.append("insufficient_test_rows")
    core = {
        "schema_version": SCHEMA_VERSION, "source_sha256": _sha256(source),
        "accepted_rows": len(records_by_date),
        "splits": {name: asdict(split) for name, split in splits.items()},
        "publishable": not blocks, "publication_blocks": blocks,
        "config": asdict(config),
    }
    manifest_hash = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION, source_sha256=core["source_sha256"],
        accepted_rows=len(records_by_date), splits=splits, publishable=not blocks,
        publication_blocks=tuple(blocks), manifest_hash=manifest_hash,
    )
    _atomic_json(output / "manifest.json", {**core, "manifest_hash": manifest_hash})
    return manifest


def verify_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir).resolve()
    try:
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid dataset manifest") from error
    expected_hash = manifest.get("manifest_hash")
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    actual_hash = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError("dataset manifest hash mismatch")
    timezone = manifest.get("config", {}).get("input_timezone")
    if not isinstance(timezone, str) or not timezone:
        raise ValueError("dataset timezone is missing")
    seen_ids: set[str] = set()
    seen_dates: set[str] = set()
    dates_by_split: list[tuple[str, ...]] = []
    total_rows = 0
    for split_name in ("train", "validation", "test"):
        path = dataset_dir / f"{split_name}.jsonl"
        split = manifest.get("splits", {}).get(split_name, {})
        if not path.is_file() or split.get("sha256") != _sha256(path):
            raise ValueError(f"{split_name} split hash mismatch")
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except json.JSONDecodeError as error:
            raise ValueError(f"{split_name} split is invalid") from error
        if split.get("rows") != len(rows):
            raise ValueError(f"{split_name} row count mismatch")
        ids = {row.get("record_id") for row in rows}
        if None in ids or len(ids) != len(rows) or ids & seen_ids:
            raise ValueError("record IDs are invalid or overlap")
        local_dates = {_local_date(row["features"]["planned_off_block"], timezone) for row in rows}
        if tuple(sorted(local_dates)) != tuple(split.get("dates", ())):
            raise ValueError(f"{split_name} local dates mismatch")
        if local_dates & seen_dates:
            raise ValueError("local dates overlap across splits")
        dates_by_split.append(tuple(sorted(local_dates)))
        seen_ids.update(ids)
        seen_dates.update(local_dates)
        total_rows += len(rows)
    if total_rows != manifest.get("accepted_rows"):
        raise ValueError("accepted row count mismatch")
    populated = [dates for dates in dates_by_split if dates]
    if any(left[-1] >= right[0] for left, right in zip(populated, populated[1:])):
        raise ValueError("dataset splits are not chronological")
    return manifest
