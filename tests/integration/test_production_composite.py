"""Isolated production stack with a real CPU CompositeBackend and replay snapshot."""
import hashlib
import io
import json
import pickle
import subprocess
import time
from uuid import uuid4

import grpc
import pytest
import pandas as pd
from grpc_health.v1 import health_pb2, health_pb2_grpc

from apps.flight.clients.inference import InferenceClient
from apps.flight.recovery.repository import initialize_schema
from services.inference.adapters.composite import CompositeBackend, HistoricalFlightBackend
from services.inference.server import serve
from tests.integration.production_fixtures import (
    ROOT, ProductionStack, free_port, http, nginx_binary, start_nginx, stop_process, wait_http,
    mysql_server, mysql_settings, mysql_config, redis_socket, queue, repo)
from tests.integration.test_durable_queue import publisher
from tests.integration.test_durable_execution import executor
from tests.training.test_baselines import labelled
from tests.training.test_flow_model import flow_rows
from tests.training.test_zggg_release import release_input
from tests.unit.test_context_snapshot import FLIGHT, replay_dataset
from training.baselines import HistoricalMedianBaseline
from training.flow_model import FlowModelArtifact
from scripts.zggg_release import stage_candidate


@pytest.fixture
def composite_stack(nginx_binary, mysql_config, repo, tmp_path):
    dataset, directory = replay_dataset(tmp_path)
    historical = HistoricalMedianBaseline().fit(labelled())
    flow = FlowModelArtifact.fit('gradient_boosting', flow_rows(), 'dataset-hash-v1')
    inputs = release_input(tmp_path)
    manifest = tmp_path / 'dataset' / 'manifest.json'
    value = json.loads(manifest.read_text())
    value['manifest_hash'] = 'dataset-hash-v1'
    manifest.write_text(json.dumps(value))
    for name, artifact in (('historical_model', historical), ('flow_model', flow)):
        path = tmp_path / f'{name}.pkl'
        path.write_bytes(pickle.dumps(artifact))
        inputs['artifacts'][name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ('model', 'adapter'):
        del inputs['artifacts'][name]
    inputs.update(flight_mode='historical', flight_model_version='historical-flight-v1',
                  flow_model_version='gradient_boosting', prompt_version='historical-duration-v1',
                  feature_contract_version='zggg-airport-flow-v1')
    release_dir = tmp_path / 'release'
    stage_candidate(inputs, release_dir)
    initialize_schema(mysql_config)
    worker_address = f'127.0.0.1:{free_port()}'
    gateway_address = f'127.0.0.1:{free_port()}'
    worker, pool = serve(worker_address, CompositeBackend(HistoricalFlightBackend(historical), flow),
                         'production-composite', 4)
    config = tmp_path / 'gateway.json'
    config.write_text(json.dumps({'listen': gateway_address, 'rpc_timeout_ms': 2000,
        'health_interval_ms': 100, 'health_timeout_ms': 50, 'failure_threshold': 3,
        'open_cooldown_ms': 1000, 'minimum_retry_budget_ms': 50,
        'workers': [{'id': 'production-composite', 'address': worker_address,
                     'capacity': 4, 'enabled': True}]}))
    gateway = subprocess.Popen([str(ROOT / 'build/phase2/flight_gateway'), '--config', str(config)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    stack = ProductionStack(nginx_binary, tmp_path / 'stack', mysql_config, gateway_address, {
        'FLIGHT_DURABLE_MODEL_VERSION': 'historical-flight-v1+gradient_boosting',
        'FLIGHT_DURABLE_SOURCE': 'BASELINE',
        'FLIGHT_DURABLE_PROMPT_VERSION': 'historical-duration-v1',
        'FLIGHT_REPLAY_ROOT': str(directory.parent),
        'FLIGHT_REPLAY_DATASET_ID': 'may-v1',
        'FLIGHT_RELEASE_MANIFEST_PATH': str(release_dir / 'candidate.json')})
    stack.directory.mkdir()
    channel = grpc.insecure_channel(gateway_address)
    try:
        grpc.channel_ready_future(channel).result(5)
        health = health_pb2_grpc.HealthStub(channel)
        deadline = time.monotonic() + 5
        while health.Check(health_pb2.HealthCheckRequest(service=''), timeout=1).status != health_pb2.HealthCheckResponse.SERVING:
            if time.monotonic() >= deadline:
                raise TimeoutError('composite gateway did not become serving')
            time.sleep(.05)
        stack.start_web(0)
        stack.start_web(1)
        stack.proxy = start_nginx(nginx_binary, tmp_path / 'nginx', stack.ports)
        wait_http(stack.url + '/health', stack.proxy)
        yield stack, worker, gateway_address
    finally:
        stack.close()
        channel.close()
        stop_process(gateway)
        worker.stop(0).wait()
        pool.shutdown(wait=True)


def _run(repo, queue, address):
    publisher(repo, queue).run_once()
    client = InferenceClient(address, 2)
    try:
        executor(repo, queue, client).run_once()
    finally:
        client.close()


def test_production_composite_source_snapshot_sse_and_download(composite_stack, repo, queue):
    stack, _, address = composite_stack
    with http(stack.url + '/api/v1/prediction-jobs', 'POST', {'flights': [FLIGHT]},
              {'Idempotency-Key': str(uuid4())}) as response:
        assert response.status == 202
        job = json.load(response)
    assert job['submission_context']['replay_dataset_id'] == 'may-v1'
    _run(repo, queue, address)
    with http(stack.url + job['status_url']) as response:
        final = json.load(response)
    assert final['status'] == 'SUCCEEDED'
    result = final['results'][0]
    assert result['source'] == 'BASELINE'
    assert result['worker_id'] == 'production-composite'
    assert result['airport_flow']['window_semantics'] == 'cumulative'
    assert final['submission_context']['snapshot_hash'] == job['submission_context']['snapshot_hash']
    with http(stack.url + job['events_url'], headers={'Last-Event-ID': '1'}) as response:
        assert 'event: complete' in response.read().decode()
    with http(stack.url + job['download_url']) as response:
        assert response.status == 200 and response.read(2) == b'PK'


def test_worker_failure_cannot_fabricate_success(composite_stack, repo, queue):
    stack, worker, address = composite_stack
    with http(stack.url + '/api/v1/prediction-jobs', 'POST', {'flights': [FLIGHT]},
              {'Idempotency-Key': str(uuid4())}) as response:
        job = json.load(response)
    worker.stop(0).wait()
    _run(repo, queue, address)
    with http(stack.url + job['status_url']) as response:
        final = json.load(response)
    assert final['status'] == 'FAILED'
    assert final['success_count'] == 0
    assert final['results'][0]['error_code'] in ('UNAVAILABLE', 'DEADLINE_EXCEEDED')


def test_production_composite_excel_upload_preserves_release(composite_stack, repo, queue):
    stack, _, address = composite_stack
    buffer = io.BytesIO()
    pd.DataFrame([FLIGHT]).to_excel(buffer, index=False)
    boundary = 'e05-' + uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="flight.xlsx"\r\n'
            'Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\r\n\r\n').encode()
    body += buffer.getvalue() + f'\r\n--{boundary}--\r\n'.encode()
    with http(stack.url + '/api/v1/prediction-jobs/upload', 'POST', body,
              {'Content-Type': 'multipart/form-data; boundary=' + boundary,
               'Idempotency-Key': str(uuid4())}) as response:
        assert response.status == 202
        job = json.load(response)
    _run(repo, queue, address)
    with http(stack.url + job['status_url']) as response:
        final = json.load(response)
    assert final['status'] == 'SUCCEEDED'
    assert final['results'][0]['source'] == 'BASELINE'
    assert final['submission_context']['replay_dataset_id'] == 'may-v1'
    with http(stack.url + job['download_url']) as response:
        exported = pd.read_excel(io.BytesIO(response.read()))
    assert exported.loc[0, '快照哈希'] == final['submission_context']['snapshot_hash']
    assert exported.loc[0, '发布版本'] == final['submission_context']['release_identity']['release_version']


def test_baseline_replay_prediction_to_recovery_roundtrip(composite_stack, repo, queue):
    stack, _, address = composite_stack
    with http(stack.url + '/api/v1/prediction-jobs', 'POST', {'flights': [FLIGHT]},
              {'Idempotency-Key': str(uuid4())}) as response:
        assert response.status == 202
        job = json.load(response)
    _run(repo, queue, address)
    scenario = {'airport': 'ZGGG', 'horizon_start': '2025-05-02T07:00:00+08:00',
                'horizon_end': '2025-05-02T16:00:00+08:00', 'slot_minutes': 15,
                'departure_capacity': 2, 'arrival_capacity': 2, 'mtt_minutes': 30,
                'closures': []}
    body = {'source_job_id': job['job_id'], 'scenario_version': 'may-e06-v1',
            'scenario': scenario}
    with http(stack.url + '/api/v1/recovery-jobs', 'POST', body,
              {'Idempotency-Key': str(uuid4())}) as response:
        assert response.status == 201
        plan = json.load(response)
    assert plan['status'] == 'SUCCEEDED'
    assert plan['source_context']['snapshot_hash'] == job['submission_context']['snapshot_hash']
    assert plan['summary']['scheduled_count'] == 1
    assert plan['summary']['validation_violation_count'] == 0
    with http(stack.url + plan['download_url']) as response:
        assert response.status == 200
        assert json.load(response)['source_context'] == plan['source_context']
