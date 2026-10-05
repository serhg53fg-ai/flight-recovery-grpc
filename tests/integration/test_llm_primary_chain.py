"""Real Gateway/MySQL/Redis chain with a controllable Qwen-contract backend."""
import json
import subprocess
import time
from uuid import uuid4

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc
from flight.v1 import prediction_pb2 as pb

from apps.flight.clients.inference import InferenceClient
from services.inference.adapters.composite import CompositeBackend
from services.inference.server import serve
from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.redis_fixtures import redis_socket, queue
from tests.integration.test_durable_storage import repo
from tests.integration.test_durable_queue import publisher
from tests.integration.test_durable_execution import executor
from tests.integration.test_prediction_flow import free_address, ROOT
from tests.unit.test_composite_worker import FlightBackend, FlowArtifact
from tests.unit.test_context_snapshot import FLIGHT, replay_dataset
from apps.flight.context.snapshot import prepare_replay_submission


@pytest.mark.parametrize('primary_fails', [False, True])
def test_real_gateway_durable_qwen_primary_and_historical_fallback(
        repo, queue, tmp_path, primary_fails):
    dataset, _ = replay_dataset(tmp_path)
    prepared = prepare_replay_submission([FLIGHT], 'Asia/Shanghai', dataset)
    flight = prepared['flights'][0]
    primary = FlightBackend(RuntimeError('qwen failure') if primary_fails else None)
    primary.model_version = 'qwen-v2'
    historical = FlightBackend(); historical.source = pb.BASELINE
    historical.model_version = 'historical-flight-v1'
    historical.prompt_version = 'historical-duration-v1'
    wa, ga = free_address(), free_address()
    worker, pool = serve(wa, CompositeBackend(primary, FlowArtifact(), historical), 'llm-primary-worker')
    config = tmp_path / 'gateway.json'
    config.write_text(json.dumps({'listen': ga, 'rpc_timeout_ms': 8000,
        'health_interval_ms': 100, 'health_timeout_ms': 50, 'failure_threshold': 3,
        'open_cooldown_ms': 1000, 'minimum_retry_budget_ms': 50,
        'workers': [{'id': 'llm-primary-worker', 'address': wa, 'capacity': 1, 'enabled': True}]}))
    gateway = subprocess.Popen([str(ROOT/'build/phase2/flight_gateway'), '--config', str(config)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    channel = grpc.insecure_channel(ga)
    try:
        grpc.channel_ready_future(channel).result(5)
        health = health_pb2_grpc.HealthStub(channel)
        deadline = time.monotonic() + 5
        while health.Check(health_pb2.HealthCheckRequest(service=''), timeout=1).status != health_pb2.HealthCheckResponse.SERVING:
            if time.monotonic() >= deadline: raise TimeoutError('gateway unhealthy')
            time.sleep(.05)
        identity = {'source': 'LLM', 'model_version': 'qwen-v2+gradient_boosting',
                    'flight_model_version': 'qwen-v2', 'flow_model_version': 'gradient_boosting',
                    'prompt_version': 'zggg-weather-duration-prompt-v2',
                    'fallback_identity': {'source':'BASELINE','flight_model_version':'historical-flight-v1',
                                          'prompt_version':'historical-duration-v1'},
                    'deployment_stage': 'experimental', 'flight_gate_passed': False}
        job = repo.submit([flight], 'Asia/Shanghai', str(uuid4()), identity['model_version'],
                          'LLM', identity['prompt_version'],
                          submission_context={'release_identity': identity})
        assert publisher(repo, queue).run_once() == 1
        client = InferenceClient(ga, timeout=8)
        try:
            assert executor(repo, queue, client).run_once()
        finally:
            client.close()
        final = repo.get(job['job_id'])
        assert final['status'] == 'SUCCEEDED', final['results']
        result = final['results'][0]
        expected_source = 'BASELINE' if primary_fails else 'LLM'
        assert result['source'] == expected_source
        assert result['flight_degraded'] == primary_fails
        assert result['flight_fallback_reason'] == ('PRIMARY_UNAVAILABLE' if primary_fails else '')
        assert final['llm_success_count'] == (0 if primary_fails else 1)
        assert final['sources'] == [expected_source]
    finally:
        channel.close()
        gateway.terminate()
        try: gateway.wait(3)
        except subprocess.TimeoutExpired: gateway.kill(); gateway.wait()
        worker.stop(0).wait(); pool.shutdown(wait=True)
