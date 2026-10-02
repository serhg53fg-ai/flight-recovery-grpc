"""Derive a timestamped METAR archive from the immutable flight workbook."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from .weather import parse_weather_report
from .zggg_dataset import _utc_datetime


SOURCE = "workbook-derived-metar-v1"
_COLUMNS = (("起飞站METAR", "计划起飞站四字码"),
            ("到达站METAR", "计划到达站四字码"))


def export_workbook_metar_archive(source: Path, output: Path,
                                  input_timezone: str = "Asia/Shanghai") -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file() or source.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError("source must be an existing Excel workbook")
    if output.exists():
        raise ValueError("archive output must be a new directory")
    frame = pd.read_excel(source)
    required = {"计划离港时间", *(airport for _, airport in _COLUMNS)}
    if required - set(frame.columns):
        raise ValueError("source is missing scheduled time or airport columns")
    reports: dict[tuple[str, str, str], dict[str, str]] = {}
    audit = {"source_rows": len(frame), "candidate_reports": 0,
             "invalid_reports": 0, "airport_mismatch": 0,
             "future_relative_to_source_row": 0,
             "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "source": SOURCE}
    for _, row in frame.iterrows():
        try:
            reference = _utc_datetime(row["计划离港时间"], input_timezone)
        except (ValueError, TypeError, OverflowError):
            audit["invalid_reports"] += sum(
                not pd.isna(row.get(column)) for column, _ in _COLUMNS)
            continue
        for column, airport_column in _COLUMNS:
            raw = row.get(column)
            if pd.isna(raw) or not str(raw).strip():
                continue
            audit["candidate_reports"] += 1
            report = parse_weather_report(raw, reference, SOURCE)
            if report.parse_status != "OK" or report.kind != "METAR":
                audit["invalid_reports"] += 1
                continue
            if report.airport != str(row[airport_column]).strip().upper():
                audit["airport_mismatch"] += 1
                continue
            # A DDHHMMZ group has no month/year. Keep only reports close to
            # the scheduled date used to resolve that ambiguity.
            if abs((report.issue_time - reference).total_seconds()) > 7 * 86400:
                audit["invalid_reports"] += 1
                continue
            if report.issue_time > reference:
                audit["future_relative_to_source_row"] += 1
            issue = report.issue_time.isoformat().replace("+00:00", "Z")
            key = (report.airport, issue, report.raw_text)
            reports[key] = {"kind": "METAR", "airport": report.airport,
                            "issue_time": issue, "report_text": report.raw_text,
                            "source": SOURCE}
    output.mkdir(parents=True)
    archive = output / "reports.csv"
    with archive.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("kind", "airport", "issue_time",
                                                  "report_text", "source"))
        writer.writeheader()
        writer.writerows(reports[key] for key in sorted(reports))
    audit["unique_reports"] = len(reports)
    audit["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def export_iem_metar_archive(source: Path, output: Path,
                             airport: str = "ZGGG") -> dict:
    """Convert IEM's independent station,valid,metar CSV to our audit format."""
    source, output = Path(source).resolve(), Path(output).resolve()
    airport = airport.strip().upper()
    if not source.is_file() or source.suffix.lower() != ".csv":
        raise ValueError("source must be an existing IEM CSV file")
    if output.exists():
        raise ValueError("archive output must be a new directory")
    if len(airport) != 4 or not airport.isalnum():
        raise ValueError("airport must be an ICAO identifier")
    source_name = f"iem-asos-{airport.lower()}-v1"
    audit = {"source": source_name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "source_rows": 0, "outside_airport": 0, "invalid_reports": 0,
             "timestamp_mismatch": 0}
    reports: dict[tuple[str, str], dict[str, str]] = {}
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or {"station", "valid", "metar"} - set(reader.fieldnames):
            raise ValueError("IEM CSV is missing required columns")
        for row in reader:
            audit["source_rows"] += 1
            if str(row["station"]).strip().upper() != airport:
                audit["outside_airport"] += 1
                continue
            raw = str(row["metar"] or "").strip()
            if raw.startswith(airport + " "):
                raw = "METAR " + raw
            try:
                valid = pd.Timestamp(row["valid"])
                if pd.isna(valid):
                    raise ValueError("missing IEM timestamp")
                if valid.tzinfo is None:
                    valid = valid.tz_localize("UTC")
                valid = valid.tz_convert("UTC").to_pydatetime()
                report = parse_weather_report(raw, valid, source_name)
            except (ValueError, TypeError, OverflowError):
                audit["invalid_reports"] += 1
                continue
            if report.parse_status != "OK" or report.kind != "METAR" or report.airport != airport:
                audit["invalid_reports"] += 1
                continue
            if report.issue_time != valid:
                audit["timestamp_mismatch"] += 1
                continue
            issue = valid.isoformat().replace("+00:00", "Z")
            reports[(issue, report.raw_text)] = {
                "kind": "METAR", "airport": airport, "issue_time": issue,
                "report_text": report.raw_text, "source": source_name,
            }
    output.mkdir(parents=True)
    archive = output / "reports.csv"
    with archive.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("kind", "airport", "issue_time",
                                                  "report_text", "source"))
        writer.writeheader()
        writer.writerows(reports[key] for key in sorted(reports))
    audit["unique_reports"] = len(reports)
    audit["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Derive a timestamped METAR archive")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("workbook", "iem"), default="workbook")
    parser.add_argument("--airport", default="ZGGG")
    args = parser.parse_args(argv)
    action = export_iem_metar_archive if args.format == "iem" else export_workbook_metar_archive
    kwargs = {"airport": args.airport} if args.format == "iem" else {}
    print(json.dumps(action(args.source, args.output, **kwargs),
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
