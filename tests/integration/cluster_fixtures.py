from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
from concurrent import futures
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import grpc
from google.protobuf.empty_pb2 import Empty
from google.protobuf.timestamp_pb2 import Timestamp
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from flight.v1 import prediction_pb2 as pb, prediction_pb2_grpc as rpc


ROOT = Path(__file__).resolve().parents[2]
GATEWAY = ROOT / 'build' / 'phase2' / 'flight_gateway'


def free_address() -> str:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return f'127.0.0.1:{sock.getsockname()[1]}'


def valid_response(trace_id: str, worker_id: str, *, composite: bool = False,
                   flow_degraded: bool = False) -> pb.PredictResponse:
    def stamp(seconds: int) -> Timestamp:
        return Timestamp(seconds=seconds)
    response = pb.PredictResponse(
        trace_id=trace_id,
        prediction=pb.Prediction(
            off_block=stamp(1_789_124_700), takeoff=stamp(1_789_125_600),
            landing=stamp(1_789_130_900), on_block=stamp(1_789_131_900),
        ),
        source=pb.TEST, model_version='cluster-fixture-v1',
        worker_id=worker_id, inference_ms=1,
    )
    if composite:
        response.source = pb.LLM
        response.model_version = 'qwen-zggg-v2'
        response.airport_flow.model_version = 'zggg-flow-gbdt-v1'
        for horizon, takeoff, landing in ((15, 4, 3), (30, 8, 7), (60, 17, 14)):
            response.airport_flow.windows.add(
                horizon_minutes=horizon, takeoff=takeoff,
                landing=landing, total=takeoff + landing,
            )
        response.weather_stale = True
        response.flow_degraded = flow_degraded
        response.data_version = 'zggg-snapshot-v1'
        response.prompt_version = 'zggg-weather-duration-prompt-v2'
    return response


class ClusterWorker(rpc.InferenceWorkerServicer):
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.status: grpc.StatusCode | None = None
        self.delay_seconds = 0.0
        self.malformed = False
        self.composite = False
        self.flow_degraded = False
        self.calls: list[str] = []
        self.requests: list[pb.PredictRequest] = []
        self.deadlines: list[float] = []
        self._lock = threading.Lock()
        self._server = None

    def Predict(self, request, context):
        with self._lock:
            self.calls.append(request.trace_id)
            self.requests.append(pb.PredictRequest.FromString(request.SerializeToString()))
            self.deadlines.append(context.time_remaining())
            status, delay, malformed = self.status, self.delay_seconds, self.malformed
        if delay:
            time.sleep(delay)
        if status is not None:
            context.abort(status, f'{self.worker_id}:{status.name}')
        if malformed:
            return pb.PredictResponse(trace_id=request.trace_id)
        return valid_response(request.trace_id, self.worker_id, composite=self.composite,
                              flow_degraded=self.flow_degraded)

    def start(self) -> str:
        address = free_address()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        rpc.add_InferenceWorkerServicer_to_server(self, server)
        health_service = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_service, server)
        assert server.add_insecure_port(address)
        server.start()
        health_service.set('', health_pb2.HealthCheckResponse.SERVING)
        health_service.set('flight.v1.InferenceWorker', health_pb2.HealthCheckResponse.SERVING)
        self._server = server
        return address

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop(0).wait()
            self._server = None


@dataclass
class RunningCluster:
    prediction: rpc.PredictionGatewayStub
    admin: rpc.GatewayAdminStub
    health: health_pb2_grpc.HealthStub
    process: subprocess.Popen
    channel: grpc.Channel


@contextmanager
def running_gateway_cluster(tmp_path: Path, workers: list[ClusterWorker], **overrides):
    addresses = [worker.start() for worker in workers]
    listen = free_address()
    config = {
        'listen': listen, 'rpc_timeout_ms': 2000,
        'health_interval_ms': 250, 'health_timeout_ms': 100,
        'failure_threshold': 3, 'open_cooldown_ms': 1000,
        'minimum_retry_budget_ms': 100,
        'workers': [
            {'id': worker.worker_id, 'address': address, 'capacity': 1, 'enabled': True}
            for worker, address in zip(workers, addresses)
        ],
    }
    config.update(overrides)
    path = tmp_path / f'gateway-{time.monotonic_ns()}.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    process = subprocess.Popen([str(GATEWAY), '--config', str(path)], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    channel = grpc.insecure_channel(listen)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        health_stub = health_pb2_grpc.HealthStub(channel)
        deadline = time.monotonic() + 5
        while True:
            response = health_stub.Check(health_pb2.HealthCheckRequest(service=''), timeout=1)
            if response.status == health_pb2.HealthCheckResponse.SERVING:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('gateway health did not become serving')
            time.sleep(0.02)
        admin_stub = rpc.GatewayAdminStub(channel)
        deadline = time.monotonic() + 5
        while True:
            status = admin_stub.GetClusterStatus(Empty(), timeout=1)
            if len(status.nodes) == len(workers) and all(node.healthy for node in status.nodes):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('not every configured worker became healthy')
            time.sleep(0.02)
        yield RunningCluster(rpc.PredictionGatewayStub(channel), admin_stub,
                             health_stub, process, channel)
    finally:
        channel.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        for worker in workers:
            worker.stop()
