import threading
import time

import pytest

from apps.flight.clients.inference import InferenceClient
from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.redis_fixtures import redis_socket, queue
from tests.integration.test_durable_storage import repo, submit, sql, expire_lease, failure
from tests.integration.test_durable_queue import publisher
from tests.integration.test_prediction_flow import gateway_address, FLIGHT


def executor(repo, queue, client, owner='executor', lease_seconds=30):
    from apps.flight.tasks.executor import DurableExecutor
    return DurableExecutor(repo, queue, client, owner, lease_seconds)


def test_real_mysql_redis_grpc_execution_preserves_stable_ids_and_source(repo, queue, gateway_address):
    job = submit(repo)
    time.sleep(.15)  # Make queue delay observable rather than relying on scheduler timing.
    publisher(repo, queue).run_once()
    client = InferenceClient(gateway_address, timeout=1)
    try:
        assert executor(repo, queue, client).run_once()
        final = repo.get(job['job_id'])
        assert final['status'] == 'SUCCEEDED' and final['completed_count'] == 1
        result = final['results'][0]
        assert result['source'] == 'TEST' and result['model_version'] == 'test-v1'
        assert final['llm_success_count'] == 0
        assert result['flight_id'] == job['flights'][0]['flight_id']
        assert result['normalized_input'] == job['flights'][0]['normalized_input']
        assert result['prediction_data']['实际离港时间'] == '2025-05-02T09:40:00+08:00'
        assert queue.client.xlen(queue.stream) == 0
        stages = [row for row in repo.stage_observations() if row['job_id'] == job['job_id']]
        assert {row['stage'] for row in stages} == {'queue', 'rpc', 'persist', 'total'}
        assert all(row['outcome'] == 'success' and row['seconds'] >= 0 for row in stages)
        assert all(row['flight_id'] == result['flight_id'] and row['trace_id'] == result['trace_id']
                   for row in stages)
        duration = {row['stage']: row['seconds'] for row in stages}
        assert duration['total'] >= duration['queue']
    finally:
        client.close()


def test_duplicate_notification_does_not_repeat_grpc(repo, queue, gateway_address):
    submit(repo)
    payload = repo.pending_outbox()[0]
    queue.publish(payload)
    queue.publish(payload)
    client = InferenceClient(gateway_address, 1)
    class Counting:
        timeout = 1
        calls = 0
        def predict(self, *args, **kwargs):
            self.calls += 1
            return client.predict(*args, **kwargs)
    counting = Counting()
    try:
        worker = executor(repo, queue, counting)
        worker.run_once()
        worker.run_once()
        assert counting.calls == 1
        assert queue.client.xlen(queue.stream) == 0
    finally:
        client.close()


def test_crashed_claim_recovers_without_accepting_old_result(repo, queue, mysql_settings, gateway_address):
    job = submit(repo)
    publisher(repo, queue).run_once()
    _, payload = queue.receive('crashed')
    old = repo.claim(payload['job_id'], payload['flight_id'], 'crashed', 30)
    expire_lease(mysql_settings, old['flight_id'])
    repo.reconcile()
    publisher(repo, queue).run_once()
    client = InferenceClient(gateway_address, 1)
    try:
        executor(repo, queue, client, 'replacement').run_once()
        assert repo.get(job['job_id'])['status'] == 'SUCCEEDED'
        assert not repo.complete(old, failure(old))
    finally:
        client.close()


def test_model_version_mismatch_is_failed_not_success(repo, queue, gateway_address):
    job = repo.submit([FLIGHT], 'Asia/Shanghai', 'wrong', 'wrong-model', 'TEST', 'test-v1')
    publisher(repo, queue).run_once()
    client = InferenceClient(gateway_address, 1)
    try:
        executor(repo, queue, client).run_once()
        assert repo.get(job['job_id'])['results'][0]['error_code'] == 'DATA_LOSS'
        stages = [row for row in repo.stage_observations() if row['job_id'] == job['job_id']]
        assert {row['stage']: row['outcome'] for row in stages} == {
            'queue': 'success', 'rpc': 'failure', 'persist': 'success', 'total': 'failure',
        }
    finally:
        client.close()


def test_heartbeat_keeps_long_running_execution_lease_and_budget(repo, queue, gateway_address):
    job = submit(repo, ttl_seconds=3)
    publisher(repo, queue).run_once()
    real = InferenceClient(gateway_address, 10)
    class Slow:
        timeout = 10
        budget = None
        def predict(self, request, cancel, timeout=None):
            self.budget = timeout
            time.sleep(1.25)
            return real.predict(request, cancel, timeout=timeout)
    client = Slow()
    try:
        executor(repo, queue, client, lease_seconds=1).run_once()
        assert 0 < client.budget <= 3
        assert repo.get(job['job_id'])['status'] == 'SUCCEEDED'
    finally:
        real.close()


def test_lost_heartbeat_cancels_call_without_acknowledging(repo, queue, monkeypatch):
    job = submit(repo)
    publisher(repo, queue).run_once()
    monkeypatch.setattr(repo, 'heartbeat', lambda *args: False)
    class WaitForCancel:
        timeout = 2
        cancelled = False
        def predict(self, request, cancel, timeout=None):
            from apps.flight.clients.inference import PredictionError
            self.cancelled = cancel.wait(2)
            raise PredictionError('CANCELLED', 'cancelled')
    client = WaitForCancel()
    executor(repo, queue, client, lease_seconds=1).run_once()
    assert client.cancelled
    assert repo.get(job['job_id'])['completed_count'] == 0
    assert queue.client.xpending(queue.stream, queue.group)['pending'] == 1


def test_normal_grpc_failure_is_not_retried_by_executor(repo, queue):
    job = submit(repo)
    publisher(repo, queue).run_once()
    class Unavailable:
        timeout = 1
        calls = 0
        def predict(self, *args, **kwargs):
            from apps.flight.clients.inference import PredictionError
            self.calls += 1
            raise PredictionError('UNAVAILABLE', 'unavailable')
    client = Unavailable()
    worker = executor(repo, queue, client)
    worker.run_once()
    worker.run_once()
    assert client.calls == 1
    assert repo.get(job['job_id'])['results'][0]['error_code'] == 'UNAVAILABLE'


def test_crash_before_and_after_result_commit(repo, queue, mysql_settings, gateway_address):
    job = submit(repo)
    publisher(repo, queue).run_once()
    message_id, payload = queue.receive('crashed-before-commit')
    old = repo.claim(payload['job_id'], payload['flight_id'], 'crashed-before-commit', 30)
    expire_lease(mysql_settings, old['flight_id'])
    repo.reconcile()
    publisher(repo, queue).run_once()
    client = InferenceClient(gateway_address, 1)
    try:
        assert executor(repo, queue, client, 'replacement').run_once()
        committed = repo.get(job['job_id'])
        assert committed['status'] == 'SUCCEEDED'
        assert len(committed['results']) == 1
        assert not repo.complete(old, failure(old))
        # Crash after the MySQL commit but before acknowledging the Redis delivery.
        queue.client.xclaim(queue.stream, queue.group, 'late-reader', min_idle_time=0,
                            message_ids=[message_id], idle=2000)
        assert executor(repo, queue, client, 'late-reader', lease_seconds=1).run_once()
        assert repo.get(job['job_id']) == committed
        assert queue.client.xpending(queue.stream, queue.group)['pending'] == 0
    finally:
        client.close()


def test_stale_attempt_cannot_overwrite_result(repo, mysql_settings):
    job = submit(repo)
    flight_id = job['flights'][0]['flight_id']
    old = repo.claim(job['job_id'], flight_id, 'old-owner', 10)
    expire_lease(mysql_settings, flight_id)
    repo.reconcile()
    current = repo.claim(job['job_id'], flight_id, 'new-owner', 10)
    assert current['fence'] > old['fence']
    assert repo.complete(current, failure(current))
    before = repo.get(job['job_id'])
    assert not repo.complete(old, failure(old))
    assert repo.get(job['job_id']) == before
