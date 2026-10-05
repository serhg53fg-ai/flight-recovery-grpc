import threading
import time
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from deploy.autodl.health import wait_for_serving


def health_server(status):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    servicer.set("flight.v1.InferenceWorker", status)
    server.start()
    return server, f"127.0.0.1:{port}"


def test_serving_endpoint_succeeds():
    server, address = health_server(health_pb2.HealthCheckResponse.SERVING)
    try:
        assert wait_for_serving(address, "flight.v1.InferenceWorker", 1.0, threading.Event())
    finally:
        server.stop(0).wait()


def test_not_serving_endpoint_times_out():
    server, address = health_server(health_pb2.HealthCheckResponse.NOT_SERVING)
    try:
        started = time.monotonic()
        assert not wait_for_serving(address, "flight.v1.InferenceWorker", 0.25, threading.Event())
        assert time.monotonic() - started < 0.8
    finally:
        server.stop(0).wait()


def test_unreachable_endpoint_times_out():
    assert not wait_for_serving("127.0.0.1:1", "flight.v1.InferenceWorker", 0.2, threading.Event())


def test_stop_event_interrupts_promptly():
    stop = threading.Event()
    timer = threading.Timer(0.1, stop.set)
    timer.start()
    started = time.monotonic()
    try:
        assert not wait_for_serving("127.0.0.1:1", "flight.v1.InferenceWorker", 10.0, stop)
        assert time.monotonic() - started < 0.8
    finally:
        timer.cancel()
