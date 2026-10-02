"""Bounded HTTP benchmark for an explicitly selected historical replay deployment."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4


DEFAULT_FLIGHT = {
    '航班号': 'CZ9933', '机尾号': 'B20E8', '机型': 'A320',
    '计划起飞站四字码': 'ZGGG', '计划到达站四字码': 'ZSPD',
    '计划离港时间': '2025-05-15T09:30:00+08:00',
    '计划到港时间': '2025-05-15T11:45:00+08:00',
}
TERMINAL = frozenset(('SUCCEEDED', 'PARTIAL', 'FAILED', 'EXPIRED', 'CANCELLED'))


class HttpClient:
    def __init__(self, base_url, timeout=60):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.path not in ('', '/') or parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError('base-url must be an explicit HTTP(S) origin without credentials')
        self.base_url = base_url.rstrip('/')
        if type(timeout) not in (int, float) or not 5 <= timeout <= 120:
            raise ValueError('HTTP timeout must be between 5 and 120 seconds')
        self.timeout = timeout

    def _request(self, method, path, body=None, key=None):
        if not path.startswith('/') or path.startswith('//'):
            raise ValueError('benchmark path must be local to the selected origin')
        headers = {'Accept': 'application/json'}
        if key:
            headers['Idempotency-Key'] = key
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        if payload is not None:
            headers['Content-Type'] = 'application/json'
        request = Request(self.base_url + path, data=payload, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            try:
                return exc.code, json.load(exc)
            finally:
                exc.close()

    def get(self, path):
        return self._request('GET', path)

    def post(self, path, body, key):
        return self._request('POST', path, body, key)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def run_benchmark(client, dataset_id, concurrency, requests, output, flight=None, poll_seconds=.25):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('benchmark report already exists')
    if not output.parent.is_dir():
        raise ValueError('benchmark output parent is missing')
    if not isinstance(dataset_id, str) or not dataset_id or type(concurrency) is not int or type(requests) is not int or not 1 <= concurrency <= min(requests, 8) or not 1 <= requests <= 100:
        raise ValueError('invalid dataset or bounded load')
    if type(poll_seconds) not in (int, float) or not .01 <= poll_seconds <= 5:
        raise ValueError('invalid poll interval')
    flight = DEFAULT_FLIGHT if flight is None else flight
    if not isinstance(flight, dict):
        raise ValueError('flight input must be one object')
    status, ready = client.get('/ready')
    if status != 200 or not ready.get('ready'):
        raise ValueError('selected deployment is not ready')
    status, capabilities = client.get('/api/v1/prediction-capabilities')
    if status != 200 or capabilities.get('replay_dataset_id') != dataset_id:
        raise ValueError('selected replay dataset does not match deployment')

    started = time.monotonic()

    def one(index):
        request_started = time.monotonic()
        status, accepted = client.post('/api/v1/prediction-jobs',
                                       {'flights': [flight], 'replay_dataset_id': dataset_id},
                                       'benchmark-' + uuid4().hex)
        if status != 202:
            return {'index': index, 'status': 'REJECTED', 'error_code': accepted.get('error_code', str(status)),
                    'latency_seconds': time.monotonic() - request_started,
                    'queue_wait_upper_seconds': None}
        job_id, path = accepted['job_id'], accepted['status_url']
        first_nonqueued = None
        accepted_at = time.monotonic()
        deadline = accepted_at + 90
        while time.monotonic() < deadline:
            status, document = client.get(path)
            if status != 200:
                return {'index': index, 'job_id': job_id, 'status': 'POLL_FAILED',
                        'error_code': document.get('error_code', str(status)),
                        'latency_seconds': time.monotonic() - request_started,
                        'queue_wait_upper_seconds': None}
            if document['status'] != 'QUEUED' and first_nonqueued is None:
                first_nonqueued = time.monotonic()
            if document['status'] in TERMINAL:
                result = (document.get('results') or [{}])[0]
                return {'index': index, 'job_id': job_id, 'status': document['status'],
                        'error_code': result.get('error_code'), 'source': result.get('source'),
                        'model_version': result.get('model_version'),
                        'latency_seconds': time.monotonic() - request_started,
                        'queue_wait_upper_seconds': first_nonqueued - accepted_at}
            time.sleep(poll_seconds)
        return {'index': index, 'job_id': job_id, 'status': 'TIMEOUT',
                'error_code': 'BENCHMARK_TIMEOUT', 'latency_seconds': time.monotonic() - request_started,
                'queue_wait_upper_seconds': None}

    rows = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(one, index): index for index in range(requests)}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({'index': futures[future], 'status': 'CLIENT_ERROR',
                             'error_code': type(exc).__name__,
                             'latency_seconds': None, 'queue_wait_upper_seconds': None})
    rows.sort(key=lambda row: row.get('index', requests))
    wall = time.monotonic() - started
    success = sum(row['status'] == 'SUCCEEDED' for row in rows)
    latencies = [row['latency_seconds'] for row in rows if row['status'] == 'SUCCEEDED']
    queue_bounds = [row['queue_wait_upper_seconds'] for row in rows
                    if row['queue_wait_upper_seconds'] is not None]
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'base_url': getattr(client, 'base_url', None), 'dataset_id': dataset_id,
              'source': capabilities['source'], 'model_version': capabilities['model_version'],
              'machine': platform.node(), 'cpu_count': os.cpu_count(), 'batch_size': 1,
              'concurrency': concurrency, 'requested': requests, 'succeeded': success,
              'errors': requests - success, 'wall_seconds': wall,
              'throughput_jobs_per_second': success / wall if wall > 0 else 0,
              'latency_seconds': {'p50': percentile(latencies, .5), 'p95': percentile(latencies, .95)},
              'queue_wait_upper_seconds': {'p50': percentile(queue_bounds, .5),
                                           'p95': percentile(queue_bounds, .95),
                                           'observed': len(queue_bounds)},
              'jobs': rows}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Bounded historical replay HTTP benchmark')
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--requests', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--flight-json', type=Path)
    parser.add_argument('--http-timeout', type=float, default=60)
    args = parser.parse_args(argv)
    flight = json.loads(args.flight_json.read_text()) if args.flight_json else None
    result = run_benchmark(HttpClient(args.base_url, args.http_timeout), args.dataset_id, args.concurrency,
                           args.requests, args.output, flight)
    print(json.dumps({key: result[key] for key in ('requested', 'succeeded', 'errors',
                                                   'throughput_jobs_per_second', 'latency_seconds')},
                     ensure_ascii=False))
    return 0 if result['errors'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
