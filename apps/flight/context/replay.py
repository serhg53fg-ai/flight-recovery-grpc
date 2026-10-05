"""Load a server-registered, hash-pinned ZGGG replay dataset."""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from apps.flight.domain.prediction import parse_time
from training.flow_features import FLOW_SCHEMA_VERSION
from training.weather import parse_weather_report


def _file(directory: Path, name: str, expected_hash: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
        raise ValueError('replay file name is invalid')
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError('replay file is missing or linked')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if not isinstance(expected_hash, str) or digest.hexdigest() != expected_hash:
        raise ValueError('replay file hash mismatch')
    return path


def _weather(path: Path | None) -> dict:
    if path is None:
        return {}
    result = {}
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or {'kind', 'airport', 'issue_time', 'report_text', 'source'} - set(reader.fieldnames):
            raise ValueError('weather archive columns are incomplete')
        for row in reader:
            issue = parse_time(row['issue_time'])
            report = parse_weather_report(row['report_text'], issue, row['source'])
            airport, kind = row['airport'].strip().upper(), row['kind'].strip().upper()
            if report.parse_status != 'OK' or report.airport != airport or report.kind != kind:
                raise ValueError('weather archive report is invalid')
            if report.issue_time != issue:
                raise ValueError('weather archive issue time does not match report')
            report = replace(report, issue_time=issue)
            received = row.get('received_time') or None
            received = parse_time(received) if received else None
            if received is not None and received < issue:
                raise ValueError('weather received time precedes issue time')
            result.setdefault(airport, []).append((report, received))
    return {airport: tuple(values) for airport, values in result.items()}


def load_replay_dataset(dataset_id: str, root: Path) -> dict:
    if not isinstance(dataset_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', dataset_id):
        raise ValueError('invalid replay dataset ID')
    directory = Path(root) / dataset_id
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('replay dataset is not registered')
    manifest_path = directory / 'replay.json'
    if manifest_path.is_symlink():
        raise ValueError('replay manifest must not be linked')
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError('invalid replay manifest') from error
    if (manifest.get('schema_version') != 'zggg-replay-v1' or
            manifest.get('airport') != 'ZGGG' or manifest.get('dataset_id') != dataset_id or
            manifest.get('feature_contract_version') != FLOW_SCHEMA_VERSION or
            manifest.get('schedule_coverage') != 'complete_zggg_schedule_declared' or
            not manifest.get('dataset_manifest_hash')):
        raise ValueError('replay manifest contract is invalid')
    source = _file(directory, manifest.get('source_file'), manifest.get('source_sha256'))
    if source.suffix.lower() == '.csv':
        frame = pd.read_csv(source)
    elif source.suffix.lower() in ('.xlsx', '.xls'):
        frame = pd.read_excel(source)
    else:
        raise ValueError('unsupported replay source format')
    columns = {'计划起飞站四字码', '计划到达站四字码', '计划离港时间', '计划到港时间',
               '实际起飞时间', '实际落地时间'}
    if columns - set(frame.columns):
        raise ValueError('replay source lacks airport schedule or observed events')
    weather_path = None
    if manifest.get('weather_file'):
        weather_path = _file(directory, manifest['weather_file'], manifest.get('weather_sha256'))
    start = parse_time(manifest.get('coverage_start'))
    end = parse_time(manifest.get('coverage_end'))
    if end <= start:
        raise ValueError('invalid replay coverage interval')
    return {'dataset_id': dataset_id, 'dataset_manifest_hash': manifest['dataset_manifest_hash'],
            'feature_contract_version': manifest['feature_contract_version'],
            'source_sha256': manifest['source_sha256'],
            'weather_sha256': manifest.get('weather_sha256'),
            'coverage_start': start, 'coverage_end': end,
            'schedule_as_of_known': manifest.get('schedule_as_of_known') is True,
            'frame': frame, 'weather': _weather(weather_path)}
