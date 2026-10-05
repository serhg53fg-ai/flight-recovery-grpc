import socket

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from services.inference.adapters.test_backend import TestBackend
from services.inference.server import serve


def _free_tcp_address():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return f'127.0.0.1:{sock.getsockname()[1]}'


def test_worker_reports_standard_serving_health():
    address = _free_tcp_address()
    server, executor = serve(address, TestBackend(), 'health-worker')
    channel = grpc.insecure_channel(address)
    stub = health_pb2_grpc.HealthStub(channel)
    try:
        for service_name in ('', 'flight.v1.InferenceWorker'):
            response = stub.Check(
                health_pb2.HealthCheckRequest(service=service_name), timeout=2
            )
            assert response.status == health_pb2.HealthCheckResponse.SERVING
    finally:
        channel.close()
        server.stop(0).wait()
        executor.shutdown(wait=True, cancel_futures=True)
