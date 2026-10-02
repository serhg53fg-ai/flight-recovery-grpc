"""Real CPU composite Workers are admitted only after a prediction identity probe."""
import socket
import json
import subprocess
import time
from pathlib import Path

import grpc
import pytest
from google.protobuf.empty_pb2 import Empty
from grpc_health.v1 import health_pb2, health_pb2_grpc

from flight.v1 import prediction_pb2_grpc
from services.inference.adapters.composite import CompositeBackend, HistoricalFlightBackend
from services.inference.server import mark_not_serving, serve
from deploy.distributed.worker_pool import probe_request
from tests.training.test_baselines import labelled
from tests.training.test_flow_model import flow_rows
from training.baselines import HistoricalMedianBaseline
from training.flow_model import FlowModelArtifact

ROOT = Path(__file__).resolve().parents[2]
GATEWAY = ROOT / 'build/phase2/flight_gateway'


def _address():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return f'127.0.0.1:{sock.getsockname()[1]}'


def test_mixed_model_pool_rejected():
    from deploy.distributed.worker_pool import probe_worker_identity

    history = HistoricalMedianBaseline().fit(labelled())
    flow = FlowModelArtifact.fit('schedule', flow_rows(), 'dataset-hash')
    address = _address()
    server, pool = serve(address, CompositeBackend(HistoricalFlightBackend(history, 'other-history'), flow), 'worker-a')
    try:
        expected = {'source': 'BASELINE', 'model_version': 'historical-flight-v1+schedule',
                    'flow_model_version': 'schedule'}
        with pytest.raises(ValueError, match='model_version'):
            probe_worker_identity(address, 'worker-a', expected)
    finally:
        server.stop(0).wait()
        pool.shutdown(wait=True)


def test_real_composite_identity_probe_accepts_matching_worker():
    from deploy.distributed.worker_pool import probe_worker_identity

    history = HistoricalMedianBaseline().fit(labelled())
    flow = FlowModelArtifact.fit('schedule', flow_rows(), 'dataset-hash')
    address = _address()
    server, pool = serve(address, CompositeBackend(HistoricalFlightBackend(history), flow), 'worker-a')
    try:
        result = probe_worker_identity(address, 'worker-a', {
            'source': 'BASELINE', 'model_version': 'historical-flight-v1+schedule',
            'flow_model_version': 'schedule',
        })
        assert result['flow_degraded'] is False
    finally:
        server.stop(0).wait()
        pool.shutdown(wait=True)


def test_gateway_admission_rejects_healthy_wrong_composite():
    from deploy.distributed.resident_role import wait_worker_pool

    history = HistoricalMedianBaseline().fit(labelled())
    flow = FlowModelArtifact.fit('schedule', flow_rows(), 'dataset-hash')
    address = _address()
    server, pool = serve(address, CompositeBackend(HistoricalFlightBackend(history, 'other-history'), flow), 'worker-a')
    try:
        config = {'backend': 'composite', 'workers': [{
            'worker_id': 'worker-a', 'address': address, 'capacity': 1,
            'release_version': 'test-v1', 'managed': False,
        }], 'release_identity': {
            'source': 'BASELINE', 'model_version': 'historical-flight-v1+schedule',
            'flow_model_version': 'schedule',
        }}
        with pytest.raises(ValueError, match='model_version'):
            wait_worker_pool(config, timeout=1)
    finally:
        server.stop(0).wait()
        pool.shutdown(wait=True)


@pytest.fixture
def real_cluster(tmp_path):
    if not GATEWAY.is_file():
        pytest.fail('build/phase2/flight_gateway is required for E08')
    history = HistoricalMedianBaseline().fit(labelled())
    flow = FlowModelArtifact.fit('schedule', flow_rows(), 'dataset-hash')

    class CountingComposite(CompositeBackend):
        def __init__(self):
            super().__init__(HistoricalFlightBackend(history), flow)
            self.calls = []

        def predict_response(self, request, cancelled, time_budget):
            self.calls.append(request.trace_id)
            return super().predict_response(request, cancelled, time_budget)

    addresses = [_address(), _address()]
    backends = [CountingComposite(), CountingComposite()]
    workers = [serve(address, backend, f'worker-{index}', max_inflight=2)
               for index, (address, backend) in enumerate(zip(addresses, backends))]
    gateway_address = _address()
    config = {'listen': gateway_address, 'rpc_timeout_ms': 2000,
              'health_interval_ms': 100, 'health_timeout_ms': 50,
              'failure_threshold': 3, 'open_cooldown_ms': 1000,
              'minimum_retry_budget_ms': 50,
              'workers': [{'id': f'worker-{index}', 'address': address,
                           'capacity': 2, 'enabled': True}
                          for index, address in enumerate(addresses)]}
    path = tmp_path / 'gateway.json'
    path.write_text(json.dumps(config))
    process = subprocess.Popen([str(GATEWAY), '--config', str(path)], cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    channel = grpc.insecure_channel(gateway_address)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        health = health_pb2_grpc.HealthStub(channel)
        deadline = time.monotonic() + 5
        while health.Check(health_pb2.HealthCheckRequest(service=''), timeout=1).status != health_pb2.HealthCheckResponse.SERVING:
            if time.monotonic() >= deadline:
                pytest.fail('real composite Gateway did not become serving')
            time.sleep(.05)
        yield prediction_pb2_grpc.PredictionGatewayStub(channel), prediction_pb2_grpc.GatewayAdminStub(channel), workers, backends
    finally:
        channel.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
        for server, executor in workers:
            server.stop(0).wait()
            executor.shutdown(wait=True)


def test_two_real_workers_receive_requests(real_cluster):
    prediction, admin, _, backends = real_cluster
    responses = [prediction.Predict(probe_request(), timeout=2) for _ in range(16)]
    assert {response.worker_id for response in responses} == {'worker-0', 'worker-1'}
    assert all(response.model_version == 'historical-flight-v1+schedule' for response in responses)
    assert all(response.airport_flow.model_version == 'schedule' for response in responses)
    assert all(backend.calls for backend in backends)
    status = admin.GetClusterStatus(Empty(), timeout=1)
    assert all(node.success_count > 0 for node in status.nodes)


def test_drain_keeps_completed_results(real_cluster):
    prediction, admin, workers, _ = real_cluster
    completed = [prediction.Predict(probe_request(), timeout=2) for _ in range(8)]
    assert all(result.model_version == 'historical-flight-v1+schedule' for result in completed)
    mark_not_serving(workers[0][0])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = admin.GetClusterStatus(Empty(), timeout=1)
        if {node.id: node.healthy for node in status.nodes} == {'worker-0': False, 'worker-1': True}:
            break
        time.sleep(.05)
    else:
        pytest.fail('draining Worker was not removed from the serving pool')
    later = [prediction.Predict(probe_request(), timeout=2) for _ in range(4)]
    assert all(result.worker_id == 'worker-1' for result in later)
    assert all(result.HasField('prediction') for result in completed)
    workers[0][0].stop(1).wait()
