"""Expose GatewayAdmin snapshots as loopback-only Prometheus text metrics."""

from __future__ import annotations

import argparse
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import grpc
from google.protobuf.empty_pb2 import Empty

from flight.v1 import prediction_pb2_grpc


def _label(value):
    return str(value).replace('\\', '\\\\').replace('\n', '\\n').replace('"', '\\"')


def render_metrics(snapshot):
    """Render one GatewayAdmin snapshot without caching or request data."""
    lines = [
        '# TYPE flight_gateway_serving gauge',
        f'flight_gateway_serving {int(snapshot.serving)}',
    ]
    gauges = {
        'healthy': 'healthy', 'circuit_state': 'circuit_state',
        'inflight': 'inflight', 'capacity': 'capacity',
    }
    counters = {
        'selected_count': 'selected_total', 'success_count': 'success_total',
        'failover_count': 'failover_total',
        'total_latency_ms': 'latency_milliseconds_total',
    }
    for suffix in (*gauges.values(), *counters.values(), 'failure_total'):
        kind = 'counter' if suffix in (*counters.values(), 'failure_total') else 'gauge'
        lines.append(f'# TYPE flight_gateway_worker_{suffix} {kind}')
    for node in snapshot.nodes:
        worker = f'worker="{_label(node.id)}"'
        for field, suffix in gauges.items():
            lines.append(f'flight_gateway_worker_{suffix}{{{worker}}} {int(getattr(node, field))}')
        for field, suffix in counters.items():
            lines.append(f'flight_gateway_worker_{suffix}{{{worker}}} {int(getattr(node, field))}')
        for code, count in sorted(node.failure_counts.items()):
            lines.append(f'flight_gateway_worker_failure_total{{{worker},code="{_label(code)}"}} {int(count)}')
    return '\n'.join(lines) + '\n'


BUCKETS = (.05, .1, .25, .5, 1, 2, 5, 10, 30, 60)


def render_stage_histograms(observations):
    """Aggregate durable observations with fixed label values and cumulative buckets."""
    from apps.flight.observability import OUTCOMES, STAGES
    groups = {}
    for row in observations:
        stage, outcome, seconds = row.get('stage'), row.get('outcome'), row.get('seconds')
        if stage not in STAGES or outcome not in OUTCOMES or not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
            continue
        groups.setdefault((stage, outcome), []).append(float(seconds))
    lines = ['# TYPE flight_stage_duration_seconds histogram']
    for (stage, outcome), values in sorted(groups.items()):
        labels = f'stage="{stage}",outcome="{outcome}"'
        for boundary in BUCKETS:
            lines.append(f'flight_stage_duration_seconds_bucket{{{labels},le="{boundary:g}"}} {sum(value <= boundary for value in values)}')
        lines.append(f'flight_stage_duration_seconds_bucket{{{labels},le="+Inf"}} {len(values)}')
        lines.append(f'flight_stage_duration_seconds_sum{{{labels}}} {sum(values):g}')
        lines.append(f'flight_stage_duration_seconds_count{{{labels}}} {len(values)}')
    return '\n'.join(lines) + '\n'


def scrape_metrics(stub, timeout=0.5, stage_loader=None):
    gateway = render_metrics(stub.GetClusterStatus(Empty(), timeout=timeout))
    return gateway + (render_stage_histograms(stage_loader()) if stage_loader else '')


def handler_for(stub, timeout, stage_loader=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/metrics':
                self.send_error(404)
                return
            try:
                body = scrape_metrics(stub, timeout, stage_loader).encode('utf-8')
            except Exception:
                self.send_error(503, 'metrics dependencies unavailable')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description='Loopback GatewayAdmin metrics exporter')
    parser.add_argument('--listen-port', type=int, default=9097)
    parser.add_argument('--gateway-address', default='127.0.0.1:50051')
    parser.add_argument('--timeout-seconds', type=float, default=0.5)
    parser.add_argument('--mysql-socket')
    args = parser.parse_args(argv)
    if not 1024 <= args.listen_port <= 65535 or not 0 < args.timeout_seconds <= 5:
        parser.error('invalid listen port or scrape timeout')
    channel = grpc.insecure_channel(args.gateway_address)
    try:
        stub = prediction_pb2_grpc.GatewayAdminStub(channel)
        stage_loader = None
        if args.mysql_socket:
            from apps.flight.storage.mysql_jobs import MySQLConfig
            from apps.flight.tasks.repository import DurableRepository
            repository = DurableRepository(MySQLConfig(database='flight_resident', user='root',
                                                        unix_socket=args.mysql_socket))
            stage_loader = repository.stage_observations
        with ThreadingHTTPServer(('127.0.0.1', args.listen_port),
                                 handler_for(stub, args.timeout_seconds, stage_loader)) as server:
            server.serve_forever()
    finally:
        channel.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
