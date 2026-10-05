"""Bounded standard gRPC Health polling."""

from __future__ import annotations

import threading
import time

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


def wait_for_serving(
    address: str,
    service: str,
    timeout_seconds: float,
    stop_event: threading.Event,
) -> bool:
    if timeout_seconds <= 0:
        return False
    deadline = time.monotonic() + timeout_seconds
    channel = grpc.insecure_channel(address)
    stub = health_pb2_grpc.HealthStub(channel)
    try:
        while not stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                response = stub.Check(
                    health_pb2.HealthCheckRequest(service=service),
                    timeout=min(0.2, remaining),
                    wait_for_ready=False,
                )
                if response.status == health_pb2.HealthCheckResponse.SERVING:
                    return True
            except grpc.RpcError:
                pass
            stop_event.wait(min(0.05, max(0.0, deadline - time.monotonic())))
        return False
    finally:
        channel.close()
