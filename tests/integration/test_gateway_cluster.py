import json
import socket
import subprocess
import time
from pathlib import Path

import grpc
import pytest
from google.protobuf.empty_pb2 import Empty
from grpc_health.v1 import health_pb2, health_pb2_grpc

from flight.v1 import prediction_pb2_grpc
from tests.integration.test_gateway import _request
from services.inference.adapters.test_backend import TestBackend
from services.inference.server import serve
from tests.integration.cluster_fixtures import ClusterWorker, running_gateway_cluster


ROOT = Path(__file__).resolve().parents[2]
GATEWAY = ROOT / 'build' / 'phase2' / 'flight_gateway'


@pytest.mark.parametrize('status', [
    grpc.StatusCode.INVALID_ARGUMENT,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.CANCELLED,
])
def test_non_unavailable_status_never_calls_alternate_worker(tmp_path, status):
    first, second = ClusterWorker('worker-a'), ClusterWorker('worker-b')
    first.status = status
    with running_gateway_cluster(tmp_path, [first, second]) as cluster:
        with pytest.raises(grpc.RpcError) as error:
            cluster.prediction.Predict(_request(f'no-retry-{status.name}'), timeout=2)
        assert error.value.code() == status
        assert first.calls == [f'no-retry-{status.name}']
        assert second.calls == []


def test_malformed_response_never_calls_alternate_worker(tmp_path):
    first, second = ClusterWorker('worker-a'), ClusterWorker('worker-b')
    first.malformed = True
    with running_gateway_cluster(tmp_path, [first, second]) as cluster:
        with pytest.raises(grpc.RpcError) as error:
            cluster.prediction.Predict(_request('no-retry-DATA_LOSS'), timeout=2)
        assert error.value.code() == grpc.StatusCode.DATA_LOSS
        assert first.calls == ['no-retry-DATA_LOSS']
        assert second.calls == []


def test_unavailable_calls_exactly_one_alternate_worker(tmp_path):
    first, second = ClusterWorker('worker-a'), ClusterWorker('worker-b')
    first.status = grpc.StatusCode.UNAVAILABLE
    with running_gateway_cluster(tmp_path, [first, second]) as cluster:
        response = cluster.prediction.Predict(_request('exact-failover'), timeout=2)
        assert response.worker_id == 'worker-b'
        assert first.calls == ['exact-failover']
        assert second.calls == ['exact-failover']
        status = cluster.admin.GetClusterStatus(Empty(), timeout=1)
        nodes = {node.id: node for node in status.nodes}
        assert nodes['worker-a'].failover_count == 0
        assert nodes['worker-b'].failover_count == 1


def test_expanded_composite_payload_survives_single_alternate_retry(tmp_path):
    first, second = ClusterWorker('worker-a'), ClusterWorker('worker-b')
    first.status = grpc.StatusCode.UNAVAILABLE
    second.composite = True
    request = _request('composite-failover')
    request.zggg_context.prediction_cutoff.CopyFrom(request.flight.planned_off_block)
    request.zggg_context.data_version = 'zggg-snapshot-v1'
    request.zggg_context.flow_features.add(horizon_minutes=15, planned_takeoff=4)
    with running_gateway_cluster(tmp_path, [first, second]) as cluster:
        response = cluster.prediction.Predict(request, timeout=2)

        assert first.calls == second.calls == ['composite-failover']
        assert second.requests[0].zggg_context == request.zggg_context
        assert response.worker_id == 'worker-b'
        assert response.model_version == 'qwen-zggg-v2'
        assert response.airport_flow.model_version == 'zggg-flow-gbdt-v1'
        assert [(item.horizon_minutes, item.takeoff, item.landing, item.total)
                for item in response.airport_flow.windows] == [
                    (15, 4, 3, 7), (30, 8, 7, 15), (60, 17, 14, 31),
                ]
        assert response.data_version == 'zggg-snapshot-v1'
        assert response.prompt_version == 'zggg-weather-duration-prompt-v2'
        assert response.weather_stale is True


def test_flow_degradation_is_success_and_does_not_trigger_failover(tmp_path):
    first, second = ClusterWorker('worker-a'), ClusterWorker('worker-b')
    first.composite = True
    first.flow_degraded = True
    with running_gateway_cluster(tmp_path, [first, second]) as cluster:
        response = cluster.prediction.Predict(_request('flow-degraded'), timeout=2)

        assert response.flow_degraded is True
        assert response.airport_flow.model_version == 'zggg-flow-gbdt-v1'
        assert first.calls == ['flow-degraded']
        assert second.calls == []
        status = cluster.admin.GetClusterStatus(Empty(), timeout=1)
        node = {item.id: item for item in status.nodes}['worker-a']
        assert node.success_count == 1
        assert node.failover_count == 0
        assert not node.failure_counts


def test_failover_uses_only_the_remaining_deadline(tmp_path):
    first, second = ClusterWorker('worker-a'), ClusterWorker('worker-b')
    first.status = grpc.StatusCode.UNAVAILABLE
    first.delay_seconds = 0.15
    with running_gateway_cluster(
        tmp_path, [first, second], rpc_timeout_ms=500, minimum_retry_budget_ms=50
    ) as cluster:
        response = cluster.prediction.Predict(_request('remaining-deadline'), timeout=2)
        assert response.worker_id == 'worker-b'
        assert len(first.deadlines) == len(second.deadlines) == 1
        assert 0 < second.deadlines[0] < first.deadlines[0] - 0.10
        assert second.deadlines[0] < 0.40


def test_insufficient_remaining_budget_does_not_retry(tmp_path):
    first, second = ClusterWorker('worker-a'), ClusterWorker('worker-b')
    first.status = grpc.StatusCode.UNAVAILABLE
    first.delay_seconds = 0.15
    with running_gateway_cluster(
        tmp_path, [first, second], rpc_timeout_ms=250, minimum_retry_budget_ms=150
    ) as cluster:
        with pytest.raises(grpc.RpcError) as error:
            cluster.prediction.Predict(_request('budget-exhausted'), timeout=2)
        assert error.value.code() == grpc.StatusCode.UNAVAILABLE
        assert first.calls == ['budget-exhausted']
        assert second.calls == []


def test_prediction_error_survives_successful_health_probe(tmp_path):
    first, second = ClusterWorker('worker-a'), ClusterWorker('worker-b')
    first.status = grpc.StatusCode.UNAVAILABLE
    with running_gateway_cluster(tmp_path, [first, second]) as cluster:
        cluster.prediction.Predict(_request('admin-error'), timeout=2)
        time.sleep(0.2)
        status = cluster.admin.GetClusterStatus(Empty(), timeout=1)
        node = {item.id: item for item in status.nodes}['worker-a']
        assert node.healthy
        assert 'UNAVAILABLE' in node.last_error


def _address():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return f'127.0.0.1:{sock.getsockname()[1]}'


def test_gateway_aggregates_worker_health_and_exposes_cluster_status(tmp_path):
    worker_addresses = [_address(), _address()]
    class SwitchableBackend(TestBackend):
        def __init__(self):
            self.unavailable = False
            self.calls = 0

        def predict(self, *args):
            self.calls += 1
            if self.unavailable:
                raise RuntimeError('injected unavailable')
            return super().predict(*args)

    backends = [SwitchableBackend(), SwitchableBackend()]
    workers = [serve(address, backends[index], f'worker-{index}')
               for index, address in enumerate(worker_addresses)]
    listen = _address()
    config = {
        'listen': listen, 'rpc_timeout_ms': 2000,
        'health_interval_ms': 100, 'health_timeout_ms': 50,
        'failure_threshold': 3, 'open_cooldown_ms': 1000,
        'minimum_retry_budget_ms': 100,
        'workers': [
            {'id': f'worker-{index}', 'address': address, 'capacity': 1, 'enabled': True}
            for index, address in enumerate(worker_addresses)
        ],
    }
    config_path = tmp_path / 'gateway.json'
    config_path.write_text(json.dumps(config))
    process = subprocess.Popen([str(GATEWAY), '--config', str(config_path)], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    channel = grpc.insecure_channel(listen)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        health = health_pb2_grpc.HealthStub(channel)
        deadline = time.monotonic() + 5
        while True:
            response = health.Check(health_pb2.HealthCheckRequest(service=''), timeout=1)
            if response.status == health_pb2.HealthCheckResponse.SERVING:
                break
            if time.monotonic() >= deadline:
                pytest.fail('gateway did not become serving')
            time.sleep(0.05)
        status = prediction_pb2_grpc.GatewayAdminStub(channel).GetClusterStatus(Empty(), timeout=1)
        assert status.serving

        assert {node.id for node in status.nodes} == {'worker-0', 'worker-1'}
        assert all(node.healthy for node in status.nodes)

        prediction = prediction_pb2_grpc.PredictionGatewayStub(channel)
        selected = {prediction.Predict(_request(f'balanced-{index}'), timeout=2).worker_id
                    for index in range(10)}
        assert selected == {'worker-0', 'worker-1'}
        backends[0].unavailable = True
        for index in range(10):
            prediction.Predict(_request(f'failover-{index}'), timeout=2)
        status = prediction_pb2_grpc.GatewayAdminStub(channel).GetClusterStatus(Empty(), timeout=1)
        by_id = {node.id: node for node in status.nodes}
        assert by_id['worker-1'].failover_count > 0
        assert backends[0].calls > 5
        assert backends[1].calls >= 15

        workers[0][0].stop(0).wait()
        deadline = time.monotonic() + 5
        while True:
            status = prediction_pb2_grpc.GatewayAdminStub(channel).GetClusterStatus(Empty(), timeout=1)
            states = {node.id: node.healthy for node in status.nodes}
            if states == {'worker-0': False, 'worker-1': True}:
                break
            if time.monotonic() >= deadline:
                pytest.fail(f'worker was not removed: {states}')
            time.sleep(0.05)
        assert status.serving

        workers[1][0].stop(0).wait()
        deadline = time.monotonic() + 5
        while True:
            response = health.Check(health_pb2.HealthCheckRequest(service=''), timeout=1)
            if response.status == health_pb2.HealthCheckResponse.NOT_SERVING:
                break
            if time.monotonic() >= deadline:
                pytest.fail('gateway stayed serving after all workers stopped')
            time.sleep(0.05)
    finally:
        channel.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        for server, executor in workers:
            server.stop(0).wait()
            executor.shutdown(wait=True, cancel_futures=True)
