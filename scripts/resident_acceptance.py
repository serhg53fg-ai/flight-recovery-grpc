"""One-command monitored Resident prediction, situation and recovery acceptance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import urlopen

from scripts.benchmark_workflow import HttpClient
from scripts.resident_monitor import evaluate_monitoring
from scripts.resident_ops import run_demo


def _write(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def run_acceptance(client, metrics_loader, output: Path, *, poll_seconds=.25,
                   timeout_seconds=180) -> dict:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('acceptance output already exists')
    if not output.parent.is_dir():
        raise ValueError('acceptance output parent is missing')
    output.mkdir()

    ready_status, ready = client.get('/ready')
    before, state = evaluate_monitoring(ready_status, ready, metrics_loader())
    _write(output / 'monitor-before.json', before)
    if before['status'] != 'PASS':
        report = {'status': 'FAIL', 'created_at': datetime.now(timezone.utc).isoformat(),
                  'before': before, 'demo': None, 'after': None}
        _write(output / 'summary.json', report)
        return report

    demo = run_demo(client, output / 'demo.json', poll_seconds=poll_seconds,
                    timeout_seconds=timeout_seconds)
    ready_status, ready = client.get('/ready')
    after, _ = evaluate_monitoring(ready_status, ready, metrics_loader(), state)
    _write(output / 'monitor-after.json', after)
    report = {'status': after['status'], 'created_at': datetime.now(timezone.utc).isoformat(),
              'before': before, 'demo': demo, 'after': after}
    _write(output / 'summary.json', report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Monitored Resident business acceptance')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url')
    parser.add_argument('--metrics-url')
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text())
        base_url = args.base_url or f"http://127.0.0.1:{config.get('http_port', 8080)}"
        metrics_url = (args.metrics_url or
                       f"http://127.0.0.1:{config.get('metrics_port', 9097)}").rstrip('/')

        def load_metrics():
            with urlopen(metrics_url + '/metrics', timeout=10) as response:
                return response.read().decode('utf-8')

        report = run_acceptance(HttpClient(base_url), load_metrics, args.output)
        print(json.dumps(report, ensure_ascii=False))
        return {'PASS': 0, 'WARN': 1, 'FAIL': 2}[report['status']]
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f'acceptance failed: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
