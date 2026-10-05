"""Evaluate Resident readiness and Prometheus counter deltas."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.error import HTTPError
from urllib.request import urlopen


LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([-+0-9.eE]+)$')
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"(?:,|$)')
CIRCUITS = {1: 'CLOSED', 2: 'OPEN', 3: 'HALF_OPEN'}


def parse_metrics(text: str) -> list[tuple[str, dict[str, str], float]]:
    rows = []
    for raw in text.splitlines():
        match = LINE.match(raw.strip())
        if not match:
            continue
        labels = {}
        if match.group(2):
            for label in LABEL.finditer(match.group(2)):
                labels[label.group(1)] = json.loads('"' + label.group(2) + '"')
        rows.append((match.group(1), labels, float(match.group(3))))
    return rows


def evaluate_monitoring(ready_status: int, ready: dict, metrics_text: str,
                        previous: dict | None = None) -> tuple[dict, dict]:
    rows = parse_metrics(metrics_text)
    workers: dict[str, dict] = {}
    failures = {}
    gateway_serving = None
    queue_count = queue_le_10 = 0
    fields = {
        'flight_gateway_worker_healthy': 'healthy',
        'flight_gateway_worker_circuit_state': 'circuit',
        'flight_gateway_worker_inflight': 'inflight',
        'flight_gateway_worker_capacity': 'capacity',
    }
    for name, labels, value in rows:
        if name == 'flight_gateway_serving':
            gateway_serving = int(value)
        elif name in fields and labels.get('worker'):
            field = fields[name]
            worker = workers.setdefault(labels['worker'], {})
            if field == 'healthy':
                worker[field] = bool(value)
            elif field == 'circuit':
                worker[field] = CIRCUITS.get(int(value), 'UNKNOWN')
            else:
                worker[field] = int(value)
        elif name == 'flight_gateway_worker_failure_total' and labels.get('worker') and labels.get('code'):
            failures[f"{labels['worker']}:{labels['code']}"] = int(value)
        elif labels.get('stage') == 'queue' and labels.get('outcome') == 'success':
            if name == 'flight_stage_duration_seconds_count':
                queue_count = int(value)
            elif name == 'flight_stage_duration_seconds_bucket' and labels.get('le') == '10':
                queue_le_10 = int(value)

    reasons = []
    if ready_status != 200 or ready.get('ready') is not True:
        reasons.append('deployment_not_ready')
    if gateway_serving != 1:
        reasons.append('gateway_not_serving')
    if not workers:
        reasons.append('worker_metrics_missing')
    for worker_id, worker in sorted(workers.items()):
        if worker.get('healthy') is not True:
            reasons.append(f'worker_unhealthy:{worker_id}')
        if worker.get('circuit') != 'CLOSED':
            reasons.append(f'worker_circuit_not_closed:{worker_id}')
        if worker.get('inflight', 0) > worker.get('capacity', 0):
            reasons.append(f'worker_over_capacity:{worker_id}')

    state = {'worker_failures': failures, 'queue_success_count': queue_count,
             'queue_success_le_10': queue_le_10}
    previous = previous if isinstance(previous, dict) else None
    new_failures = {}
    queue_over = 0
    if previous is not None:
        old_failures = previous.get('worker_failures', {})
        for key, value in failures.items():
            delta = value - int(old_failures.get(key, 0))
            if delta > 0:
                new_failures[key] = delta
        count_delta = max(0, queue_count - int(previous.get('queue_success_count', 0)))
        le_delta = max(0, queue_le_10 - int(previous.get('queue_success_le_10', 0)))
        queue_over = max(0, count_delta - le_delta)

    status = 'FAIL' if reasons else 'WARN' if new_failures or queue_over else 'PASS'
    report = {
        'status': status, 'reasons': reasons, 'ready': ready_status == 200 and ready.get('ready') is True,
        'gateway_serving': gateway_serving == 1, 'workers': workers,
        'delta_observed': previous is not None, 'new_worker_failures': new_failures,
        'queue_over_10_seconds': queue_over,
    }
    return report, state


def write_report(output: Path, report: dict) -> None:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('monitor report already exists')
    if not output.parent.is_dir():
        raise ValueError('monitor report parent is missing')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def _origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.path not in ('', '/') or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError('base-url must be an explicit HTTP(S) origin without credentials')
    return value.rstrip('/')


def _get_json(url: str) -> tuple[int, dict]:
    try:
        with urlopen(url, timeout=10) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        try:
            return exc.code, json.load(exc)
        finally:
            exc.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Resident readiness and metrics delta gate')
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--metrics-url', required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        base = _origin(args.base_url)
        metrics_origin = _origin(args.metrics_url)
        ready_status, ready = _get_json(base + '/ready')
        with urlopen(metrics_origin + '/metrics', timeout=10) as response:
            metrics = response.read().decode('utf-8')
        previous = json.loads(args.state.read_text()) if args.state.is_file() else None
        report, state = evaluate_monitoring(ready_status, ready, metrics, previous)
        report['created_at'] = datetime.now(timezone.utc).isoformat()
        write_report(args.output, report)
        temporary = args.state.with_suffix(args.state.suffix + '.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(args.state)
        print(json.dumps(report, ensure_ascii=False))
        return {'PASS': 0, 'WARN': 1, 'FAIL': 2}[report['status']]
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f'monitoring gate failed: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
