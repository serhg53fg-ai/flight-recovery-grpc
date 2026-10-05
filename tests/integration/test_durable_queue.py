import time

import pytest

from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.test_durable_storage import repo, submit, sql
from tests.integration.redis_fixtures import redis_socket, queue
from tests.integration.test_prediction_flow import gateway_address


def test_repository_readiness_ping_uses_authoritative_database(repo):
    assert repo.ping() is True


def test_stage_metrics_are_visible_across_repository_processes(repo):
    from apps.flight.tasks.repository import DurableRepository

    job = submit(repo)
    flight_id = job['flights'][0]['flight_id']
    claim = repo.claim(job['job_id'], flight_id, 'stage-test-owner', 10)
    repo.record_stage(claim, 'rpc', .125, 'success')
    other = DurableRepository(repo.config)
    assert {'stage': 'rpc', 'seconds': .125, 'outcome': 'success'} in [
        {key: row[key] for key in ('stage', 'seconds', 'outcome')}
        for row in other.stage_observations()
    ]


def publisher(repo, queue):
    from apps.flight.tasks.publisher import OutboxPublisher
    return OutboxPublisher(repo, queue)


def test_publication_failure_retains_database_notification(repo, queue, mysql_settings):
    from apps.flight.tasks.queue import QueueError
    submit(repo)
    class Offline:
        def publish(self, payload):
            raise QueueError('offline')
    assert publisher(repo, Offline()).run_once() == 0
    assert sql(mysql_settings, 'SELECT published_at,failures FROM prediction_outbox') == ((None, 1),)
    sql(mysql_settings, 'UPDATE prediction_outbox SET next_publish_at=UTC_TIMESTAMP(6)')
    assert publisher(repo, queue).run_once() == 1


def test_crash_after_publish_causes_duplicate_but_same_flight_identity(repo, queue):
    job = submit(repo)
    payload = repo.pending_outbox()[0]
    queue.publish(payload)  # Simulate crash before marking published.
    assert publisher(repo, queue).run_once() == 1
    a_id, a = queue.receive('a')
    b_id, b = queue.receive('b')
    assert a_id != b_id and a == b == payload
    assert a['flight_id'] == job['flights'][0]['flight_id']
    queue.ack(a_id)
    queue.ack(b_id)
    assert queue.client.xlen(queue.stream) == 0


def test_pending_message_reclaimed_by_new_consumer(queue):
    from uuid import uuid4
    payload = {k: str(uuid4()) for k in ('job_id', 'flight_id', 'message_id')}
    queue.publish(payload)
    first = queue.receive('old')
    assert queue.receive('new', idle_ms=100000) is None
    time.sleep(.02)
    assert queue.receive('new', idle_ms=1) == first
    queue.ack(first[0])


def test_queue_capacity_is_bounded_without_dropping_messages(queue):
    from uuid import uuid4
    from apps.flight.tasks.queue import QueueError
    queue.capacity = 1
    payload = {k: str(uuid4()) for k in ('job_id', 'flight_id', 'message_id')}
    queue.publish(payload)
    with pytest.raises(QueueError):
        queue.publish(payload)
    assert queue.client.xlen(queue.stream) == 1
    message_id, _ = queue.receive('a')
    queue.ack(message_id)
    queue.publish(payload)


def test_rebuild_deleted_stream_from_mysql_only_unfinished_work(repo, queue):
    job = submit(repo)
    assert publisher(repo, queue).run_once() == 1
    queue.client.delete(queue.stream)  # Only this test's owned stream.
    repo.reconcile(republish=True)
    assert publisher(repo, queue).run_once() == 1
    _, payload = queue.receive('new')
    assert payload['job_id'] == job['job_id']


def test_redis_loss_recovered_from_mysql(repo, queue, gateway_address):
    from apps.flight.clients.inference import InferenceClient
    from tests.integration.test_durable_execution import executor

    job = submit(repo)
    assert publisher(repo, queue).run_once() == 1
    queue.client.delete(queue.stream)
    repo.reconcile(republish=True)
    assert publisher(repo, queue).run_once() == 1
    client = InferenceClient(gateway_address, 1)
    try:
        assert executor(repo, queue, client, 'after-redis-loss').run_once()
        assert repo.get(job['job_id'])['status'] == 'SUCCEEDED'
        assert repo.get(job['job_id'])['success_count'] == 1
    finally:
        client.close()


def test_expired_task_has_terminal_reason(repo, mysql_settings):
    job = submit(repo)
    sql(mysql_settings, 'UPDATE durable_jobs SET deadline_at=UTC_TIMESTAMP(6)-INTERVAL 1 SECOND '
                        'WHERE job_id=%s', (job['job_id'],))
    repo.reconcile()
    final = repo.get(job['job_id'])
    assert final['status'] == 'EXPIRED'
    assert final['results'][0]['error_code'] == 'DEADLINE_EXCEEDED'
    assert repo.pending_outbox() == []


def test_queue_error_and_repr_do_not_include_password(tmp_path):
    from apps.flight.tasks.queue import StreamQueue, QueueError
    queue = StreamQueue(unix_socket=str(tmp_path / 'missing.sock'), password='never-show-me')
    try:
        assert 'never-show-me' not in repr(queue)
        with pytest.raises(QueueError) as caught:
            queue.receive('a')
        assert 'never-show-me' not in str(caught.value)
    finally:
        queue.close()
