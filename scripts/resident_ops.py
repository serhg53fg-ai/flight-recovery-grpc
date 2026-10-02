"""One-command Resident lifecycle and auditable ZGGG demonstration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from scripts.benchmark_workflow import HttpClient, TERMINAL


DEMO_FLIGHT = {
    '航班号': 'CZ9933', '机尾号': 'B20E8', '机型': 'A320', '性质': 'J',
    '计划起飞站四字码': 'ZGGG', '计划到达站四字码': 'ZSPD',
    '计划离港时间': '2025-05-02 09:30:00', '计划到港时间': '2025-05-02 11:45:00',
    '起飞站METAR': 'ZGGG 020100Z 08005MPS CAVOK 25/18 Q1012',
    '计划总流量': 0, '计划地面航程_Mile': 650,
    '计划航段时间\n（24年同航季平均值）': 135,
}
DEMO_SCENARIO = {
    'airport': 'ZGGG', 'horizon_start': '2025-05-02T08:00:00+08:00',
    'horizon_end': '2025-05-02T16:00:00+08:00', 'slot_minutes': 15,
    'departure_capacity': 1, 'arrival_capacity': 1, 'mtt_minutes': 30,
    'closures': [],
}


def run_stack(action: str, config_path: Path, *, runner=subprocess.run) -> int:
    if action not in {'up', 'status', 'health', 'stop'}:
        raise ValueError('invalid Resident operation')
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text(encoding='utf-8'))
    runtime = Path(raw['runtime']).resolve()
    base = [sys.executable, '-m', 'deploy.distributed.resident']
    if action == 'up':
        if (runtime / 'supervisor.sock').exists():
            actions = ['health']
        elif (runtime / 'initialized.json').is_file():
            actions = ['start']
        else:
            actions = ['preflight', 'prepare', 'start']
    else:
        actions = [action]
    for resident_action in actions:
        result = runner(base + [resident_action, '--config', str(config_path)], check=False)
        if result.returncode:
            return result.returncode
    return 0


def run_demo(client, output: Path, *, poll_seconds=.25, timeout_seconds=180) -> dict:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('demo report already exists')
    if not output.parent.is_dir():
        raise ValueError('demo report parent is missing')
    if not .01 <= poll_seconds <= 5 or not 1 <= timeout_seconds <= 600:
        raise ValueError('invalid demo timing')
    status, ready = client.get('/ready')
    if status != 200 or ready.get('ready') is not True:
        raise ValueError('selected Resident stack is not ready')
    status, accepted = client.post('/api/v1/prediction-jobs', {'flights': [DEMO_FLIGHT]},
                                   'resident-demo-' + uuid4().hex)
    if status != 202:
        raise RuntimeError('prediction submission failed')
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status, prediction = client.get(accepted['status_url'])
        if status != 200:
            raise RuntimeError('prediction polling failed')
        if prediction['status'] in TERMINAL:
            break
        time.sleep(poll_seconds)
    else:
        raise TimeoutError('prediction demonstration timed out')
    if prediction['status'] != 'SUCCEEDED':
        raise RuntimeError('prediction demonstration failed')
    result = prediction['results'][0]
    status, recovery = client.post('/api/v1/recovery-jobs', {
        'source_job_id': accepted['job_id'], 'scenario': DEMO_SCENARIO,
        'scenario_version': 'resident-demo-v1'}, 'resident-recovery-' + uuid4().hex)
    if status != 201 or recovery.get('status') != 'SUCCEEDED' or recovery.get('validation_errors') != []:
        raise RuntimeError('recovery demonstration failed')
    report = {
        'status': 'PASS', 'job_id': accepted['job_id'],
        'recovery_id': recovery['recovery_id'],
        'prediction': {name: result.get(name) for name in (
            'source', 'model_version', 'worker_id', 'flight_degraded', 'flow_degraded')},
        'situation': {
            'risk_level': recovery['situation_snapshot']['risk_level'],
            'snapshot_digest': recovery['situation_snapshot']['snapshot_digest'],
        },
        'recovery': {
            'status': recovery['status'],
            'scheduled_count': recovery['summary']['scheduled_count'],
            'validation_errors': recovery['validation_errors'],
        },
    }
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Resident lifecycle and ZGGG demonstration')
    parser.add_argument('action', choices=('up', 'status', 'health', 'demo', 'stop'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--base-url')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action != 'demo':
            return run_stack(args.action, args.config)
        raw = json.loads(args.config.read_text(encoding='utf-8'))
        base_url = args.base_url or f"http://127.0.0.1:{raw.get('http_port', 8080)}"
        if args.output is None:
            parser.error('demo requires --output')
        report = run_demo(HttpClient(base_url), args.output)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    except (ValueError, RuntimeError, TimeoutError, OSError, KeyError, TypeError,
            json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f'resident operation failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
