"""Build and verify cutoff-safe ZGGG flight and weather datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import (
    DatasetConfig, DatasetSplit, _atomic_json, _atomic_jsonl, _local_date,
    _record, _sha256,
)
from .schema import (
    FEATURE_COLUMNS, PROHIBITED_FEATURE_COLUMNS, TARGET_COLUMNS,
    WEATHER_FEATURE_VERSION, ZGGG_AIRPORT,
)
from .weather import WeatherReport, WeatherSnapshot, parse_weather_report, select_weather_snapshot
from .flow import FLOW_HORIZONS, build_flow_snapshots


SCHEMA_VERSION = "zggg-flight-weather-v1"


@dataclass(frozen=True)
class ZGGGDatasetManifest:
    schema_version: str
    airport: str
    source_sha256: str
    accepted_rows: int
    splits: dict[str, DatasetSplit]
    flow_splits: dict[str, DatasetSplit]
    publishable: bool
    publication_blocks: tuple[str, ...]
    manifest_hash: str


def _utc_datetime(value: Any, timezone_name: str) -> datetime:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("missing timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(timezone_name)
    return parsed.tz_convert("UTC").to_pydatetime()


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _snapshot_json(snapshot: WeatherSnapshot) -> dict[str, Any]:
    result: dict[str, Any] = {
        "missing": snapshot.missing,
        "stale": snapshot.stale,
        "report_age_minutes": snapshot.report_age_minutes,
    }
    if snapshot.report is None:
        return result
    report = snapshot.report
    result.update({
        "kind": report.kind,
        "airport": report.airport,
        "issue_time": _iso(report.issue_time),
        "valid_from": _iso(report.valid_from),
        "valid_to": _iso(report.valid_to),
        "source": report.source,
        "raw_text": report.raw_text,
        "parse_status": report.parse_status,
        "features": report.feature_dict(),
    })
    return result


def _load_archive(path: Path | None, timezone_name: str,
                  weather_feature_version: str) -> dict[str, list[WeatherReport]]:
    reports: dict[str, list[WeatherReport]] = {}
    if path is None:
        return reports
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("weather archive must be an existing CSV file")
    frame = pd.read_csv(path)
    required = {"kind", "airport", "issue_time", "report_text", "source"}
    if required - set(frame.columns):
        raise ValueError("weather archive is missing required columns")
    for _, raw in frame.iterrows():
        issue = _utc_datetime(raw["issue_time"], timezone_name)
        report = parse_weather_report(raw["report_text"], issue, raw["source"],
                                      feature_version=weather_feature_version)
        kind, airport = str(raw["kind"]).upper(), str(raw["airport"]).upper()
        if report.parse_status != "OK" or report.kind != kind or report.airport != airport:
            raise ValueError("weather archive report metadata mismatch")
        report = replace(report, issue_time=issue)
        reports.setdefault(airport, []).append(report)
    return reports


def _embedded_reports(raw: pd.Series, cutoff: datetime,
                      weather_feature_version: str) -> dict[str, list[WeatherReport]]:
    result: dict[str, list[WeatherReport]] = {}
    for column, airport_column in (
        ("起飞站METAR", "计划起飞站四字码"),
        ("到达站METAR", "计划到达站四字码"),
    ):
        value = raw.get(column)
        if pd.isna(value) or not str(value).strip():
            continue
        report = parse_weather_report(value, cutoff, source="embedded",
                                      feature_version=weather_feature_version)
        expected = str(raw.get(airport_column, "")).upper()
        if report.parse_status == "OK" and report.airport == expected:
            result.setdefault(expected, []).append(report)
    return result


def _weather_bundle(reports: list[WeatherReport], airport: str, cutoff: datetime,
                    target: datetime) -> dict[str, dict[str, Any]]:
    matching = [item for item in reports if item.airport == airport]
    return {
        kind.lower(): _snapshot_json(select_weather_snapshot(
            matching, cutoff, target, kind,
        ))
        for kind in ("METAR", "TAF")
    }


def _split_dates(dates: list[str], config: DatasetConfig) -> dict[str, list[str]]:
    if dates:
        train_count = max(1, int(len(dates) * config.train_fraction))
        validation_count = max(1, int(len(dates) * config.validation_fraction)) if len(dates) >= 3 else 0
        if train_count + validation_count >= len(dates):
            validation_count = max(0, len(dates) - train_count - 1)
    else:
        train_count = validation_count = 0
    return {
        "train": dates[:train_count],
        "validation": dates[train_count:train_count + validation_count],
        "test": dates[train_count + validation_count:],
    }


def build_zggg_dataset(source: Path, output: Path, config: DatasetConfig = DatasetConfig(),
                       weather_archive: Path | None = None,
                       weather_feature_version: str = WEATHER_FEATURE_VERSION) -> ZGGGDatasetManifest:
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file() or source.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError("source must be an existing Excel file")
    if not 0 < config.train_fraction < 1 or not 0 < config.validation_fraction < 1 or config.train_fraction + config.validation_fraction >= 1:
        raise ValueError("invalid split fractions")
    if weather_feature_version not in ('aviation-weather-v1', 'aviation-weather-v2'):
        raise ValueError('unsupported weather feature version')
    frame = pd.read_excel(source)
    required = set(FEATURE_COLUMNS) | set(TARGET_COLUMNS)
    if required - set(frame.columns):
        raise ValueError("source is missing required columns")
    archive = _load_archive(weather_archive, config.input_timezone, weather_feature_version)
    weather_archive_sha256 = _sha256(Path(weather_archive).resolve()) if weather_archive else None
    records_by_date: list[tuple[dict[str, Any], str]] = []
    rejected: dict[str, int] = {}
    scope_rejections = 0
    seen: set[str] = set()
    weather_counts = {"departure_metar_available": 0, "arrival_metar_available": 0,
                      "taf_available": 0}
    for _, raw in frame.iterrows():
        departure = str(raw.get("计划起飞站四字码", "")).upper()
        arrival = str(raw.get("计划到达站四字码", "")).upper()
        if (departure == ZGGG_AIRPORT) == (arrival == ZGGG_AIRPORT):
            scope_rejections += 1
            continue
        try:
            record, local_date = _record(raw, config)
            if record["record_id"] in seen:
                raise ValueError("duplicate")
            cutoff = _utc_datetime(raw["计划离港时间"], config.input_timezone)
            departure_target = cutoff
            arrival_target = _utc_datetime(raw["计划到港时间"], config.input_timezone)
            available = {key: list(value) for key, value in archive.items()}
            for airport, values in _embedded_reports(raw, cutoff, weather_feature_version).items():
                available.setdefault(airport, []).extend(values)
            departure_weather = _weather_bundle(available.get(departure, []), departure,
                                                cutoff, departure_target)
            arrival_weather = _weather_bundle(available.get(arrival, []), arrival,
                                              cutoff, arrival_target)
            features = record["features"]
            features.update({
                "airport_direction": "DEPARTURE" if departure == ZGGG_AIRPORT else "ARRIVAL",
                "prediction_cutoff_time": _iso(cutoff),
                "departure_weather": departure_weather,
                "arrival_weather": arrival_weather,
            })
            for side, bundle in (("departure", departure_weather), ("arrival", arrival_weather)):
                if not bundle["metar"]["missing"]:
                    weather_counts[f"{side}_metar_available"] += 1
                if not bundle["taf"]["missing"]:
                    weather_counts["taf_available"] += 1
            seen.add(record["record_id"])
            records_by_date.append((record, local_date))
        except (ValueError, TypeError, OverflowError, KeyError) as error:
            reason = str(error) if str(error) in {"duplicate", "invalid_time_order", "invalid_planned_order", "out_of_range"} else "missing_or_invalid"
            rejected[reason] = rejected.get(reason, 0) + 1
    records_by_date.sort(key=lambda item: (item[1], item[0]["record_id"]))
    dates = sorted({date for _, date in records_by_date})
    assigned_dates = _split_dates(dates, config)
    output.mkdir(parents=True, exist_ok=True)
    splits: dict[str, DatasetSplit] = {}
    for name, selected_dates in assigned_dates.items():
        selected = set(selected_dates)
        rows = [record for record, date in records_by_date if date in selected]
        splits[name] = DatasetSplit(len(rows), tuple(selected_dates),
                                    _atomic_jsonl(output / f"{name}.jsonl", rows))
    flow_splits: dict[str, DatasetSplit] = {}
    maximum_horizon = max(FLOW_HORIZONS)
    for name, selected_dates in assigned_dates.items():
        cutoffs = []
        for date in selected_dates:
            start = pd.Timestamp(date).tz_localize(config.input_timezone)
            end = start + pd.Timedelta(days=1, minutes=-maximum_horizon)
            cutoffs.extend(item.to_pydatetime() for item in pd.date_range(
                start=start, end=end, freq="15min",
            ))
        rows = build_flow_snapshots(frame, cutoffs, config.input_timezone)
        flow_splits[name] = DatasetSplit(
            len(rows), tuple(selected_dates),
            _atomic_jsonl(output / f"flow_{name}.jsonl", rows),
        )
    audit = {
        "source_rows": len(frame), "accepted_rows": len(records_by_date),
        "scope_rejections": scope_rejections, "rejections": dict(sorted(rejected.items())),
        "date_min": dates[0] if dates else None, "date_max": dates[-1] if dates else None,
        "weather": weather_counts, "weather_archive_sha256": weather_archive_sha256,
    }
    _atomic_json(output / "audit.json", audit)
    blocks = []
    if len(dates) < config.minimum_dates:
        blocks.append("insufficient_dates")
    if splits["test"].rows < config.min_test_rows:
        blocks.append("insufficient_test_rows")
    core = {
        "schema_version": SCHEMA_VERSION, "airport": ZGGG_AIRPORT,
        "weather_feature_version": weather_feature_version,
        "source_sha256": _sha256(source),
        "weather_archive_sha256": weather_archive_sha256,
        "accepted_rows": len(records_by_date),
        "splits": {name: asdict(split) for name, split in splits.items()},
        "flow_splits": {name: asdict(split) for name, split in flow_splits.items()},
        "publishable": not blocks, "publication_blocks": blocks,
        "config": asdict(config),
    }
    manifest_hash = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _atomic_json(output / "manifest.json", {**core, "manifest_hash": manifest_hash})
    return ZGGGDatasetManifest(
        SCHEMA_VERSION, ZGGG_AIRPORT, core["source_sha256"], len(records_by_date),
        splits, flow_splits, not blocks, tuple(blocks), manifest_hash,
    )


def verify_zggg_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir).resolve()
    try:
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid dataset manifest") from error
    expected_hash = manifest.get("manifest_hash")
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    actual_hash = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError("dataset manifest hash mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("airport") != ZGGG_AIRPORT:
        raise ValueError("invalid ZGGG dataset identity")
    timezone_name = manifest.get("config", {}).get("input_timezone")
    seen_ids: set[str] = set()
    seen_dates: set[str] = set()
    dates_by_split = []
    total = 0
    for name in ("train", "validation", "test"):
        path = dataset_dir / f"{name}.jsonl"
        split = manifest.get("splits", {}).get(name, {})
        if not path.is_file() or split.get("sha256") != _sha256(path):
            raise ValueError(f"{name} split hash mismatch")
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} split is invalid") from error
        if split.get("rows") != len(rows):
            raise ValueError(f"{name} row count mismatch")
        local_dates = set()
        for row in rows:
            record_id, features = row.get("record_id"), row.get("features", {})
            if not record_id or record_id in seen_ids:
                raise ValueError("record IDs are invalid or overlap")
            seen_ids.add(record_id)
            departure, arrival = features.get("departure_airport"), features.get("arrival_airport")
            if (departure == ZGGG_AIRPORT) == (arrival == ZGGG_AIRPORT):
                raise ValueError("record is outside ZGGG scope")
            direction = "DEPARTURE" if departure == ZGGG_AIRPORT else "ARRIVAL"
            if features.get("airport_direction") != direction:
                raise ValueError("record direction is invalid")
            cutoff_text = features.get("prediction_cutoff_time")
            if cutoff_text != features.get("planned_off_block"):
                raise ValueError("prediction cutoff does not match planned off-block")
            cutoff = pd.Timestamp(cutoff_text)
            for side in ("departure_weather", "arrival_weather"):
                for kind in ("metar", "taf"):
                    weather = features.get(side, {}).get(kind, {})
                    issue = weather.get("issue_time")
                    if issue is not None and pd.Timestamp(issue) > cutoff:
                        raise ValueError("weather report is after prediction cutoff")
            if set(features) & PROHIBITED_FEATURE_COLUMNS:
                raise ValueError("prohibited feature is present")
            local_dates.add(_local_date(cutoff_text, timezone_name))
        if tuple(sorted(local_dates)) != tuple(split.get("dates", ())):
            raise ValueError(f"{name} local dates mismatch")
        if local_dates & seen_dates:
            raise ValueError("local dates overlap across splits")
        seen_dates.update(local_dates)
        dates_by_split.append(tuple(sorted(local_dates)))
        total += len(rows)
        flow_path = dataset_dir / f"flow_{name}.jsonl"
        flow_split = manifest.get("flow_splits", {}).get(name, {})
        if not flow_path.is_file() or flow_split.get("sha256") != _sha256(flow_path):
            raise ValueError(f"flow {name} split hash mismatch")
        try:
            flow_rows = [json.loads(line) for line in flow_path.read_text(encoding="utf-8").splitlines()]
        except json.JSONDecodeError as error:
            raise ValueError(f"flow {name} split is invalid") from error
        if flow_split.get("rows") != len(flow_rows):
            raise ValueError(f"flow {name} row count mismatch")
        for flow_row in flow_rows:
            features, labels = flow_row.get("features", {}), flow_row.get("labels", {})
            if features.get("airport") != ZGGG_AIRPORT:
                raise ValueError("flow record is outside ZGGG scope")
            cutoff = pd.Timestamp(features.get("prediction_cutoff_time"))
            if _local_date(cutoff, timezone_name) not in set(flow_split.get("dates", ())):
                raise ValueError(f"flow {name} local dates mismatch")
            for minutes in FLOW_HORIZONS:
                if labels.get(f"total_{minutes}m") != (
                    labels.get(f"takeoff_{minutes}m", 0) + labels.get(f"landing_{minutes}m", 0)
                ):
                    raise ValueError("flow total label is inconsistent")
    if total != manifest.get("accepted_rows"):
        raise ValueError("accepted row count mismatch")
    populated = [dates for dates in dates_by_split if dates]
    if any(left[-1] >= right[0] for left, right in zip(populated, populated[1:])):
        raise ValueError("dataset splits are not chronological")
    return manifest
