"""Read-only admission audit for new, labeled ZGGG blind-test workbooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .schema import FEATURE_COLUMNS, TARGET_COLUMNS, ZGGG_AIRPORT


def audit_blind_source(source: Path, *, after_date: str, min_rows: int = 500,
                       input_timezone: str = 'Asia/Shanghai') -> dict:
    source = Path(source).resolve()
    if not source.is_file() or source.suffix.lower() not in ('.xlsx', '.xls'):
        raise ValueError('source must be an existing Excel workbook')
    if min_rows < 1:
        raise ValueError('min_rows must be positive')
    previous_end = pd.Timestamp(after_date).date()
    frame = pd.read_excel(source)
    required = set(FEATURE_COLUMNS) | set(TARGET_COLUMNS)
    missing_columns = sorted(required - set(frame.columns))
    blocks = []
    if missing_columns:
        blocks.append('missing_required_columns')
    if {'计划起飞站四字码', '计划到达站四字码'} <= set(frame.columns):
        departure = frame['计划起飞站四字码'].astype(str).str.strip().str.upper()
        arrival = frame['计划到达站四字码'].astype(str).str.strip().str.upper()
        zggg = frame.loc[(departure == ZGGG_AIRPORT) ^ (arrival == ZGGG_AIRPORT)]
    else:
        zggg = frame.iloc[0:0]
    if len(zggg) < min_rows:
        blocks.append('insufficient_zggg_rows')
    local_dates = []
    if '计划离港时间' in zggg:
        for value in zggg['计划离港时间']:
            try:
                timestamp = pd.Timestamp(value)
                if pd.isna(timestamp):
                    raise ValueError('missing timestamp')
                timestamp = timestamp.tz_localize(input_timezone) if timestamp.tzinfo is None else timestamp.tz_convert(input_timezone)
                local_dates.append(timestamp.date())
            except (ValueError, TypeError, OverflowError):
                blocks.append('invalid_planned_times')
                break
    else:
        blocks.append('invalid_planned_times')
    if not local_dates or min(local_dates) <= previous_end:
        blocks.append('not_after_previous_dataset')
    actual_columns = list(TARGET_COLUMNS)
    if not set(actual_columns) <= set(zggg.columns):
        blocks.append('missing_actual_times')
    elif zggg[actual_columns].isna().any().any() or any(
        pd.to_datetime(zggg[name], errors='coerce').isna().any() for name in actual_columns
    ):
        blocks.append('missing_actual_times')
    return {
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_rows': len(frame), 'zggg_rows': len(zggg),
        'date_min': min(local_dates).isoformat() if local_dates else None,
        'date_max': max(local_dates).isoformat() if local_dates else None,
        'after_date': previous_end.isoformat(),
        'missing_columns': missing_columns,
        'blocks': sorted(set(blocks)), 'eligible': not blocks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--after-date', required=True)
    parser.add_argument('--min-rows', type=int, default=500)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit_blind_source(args.source, after_date=args.after_date, min_rows=args.min_rows)
    content = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')
    else:
        print(content, end='')
    return 0 if report['eligible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
