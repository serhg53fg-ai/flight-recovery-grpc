from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import pytest

from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.test_prediction_flow import FLIGHT


@pytest.fixture
def repo(mysql_config):
    from apps.flight.tasks.repository import DurableRepository, initialize_durable
    initialize_durable(mysql_config)
    return DurableRepository(mysql_config)


def submit(repo, flights=None, key='sample', **kwargs):
    return repo.submit(flights or [FLIGHT], 'Asia/Shanghai', key,
                       'test-v1', 'TEST', 'test-v1', **kwargs)


def sql(settings, command, values=()):
    import pymysql
    conn = pymysql.connect(**settings, autocommit=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute(command, values)
            return cursor.fetchall()
    finally:
        conn.close()


def expire_lease(settings, flight_id):
    sql(settings, 'UPDATE durable_flights SET lease_until=UTC_TIMESTAMP(6)-INTERVAL 1 SECOND WHERE flight_id=%s', (flight_id,))


def failure(claim, code='UNAVAILABLE'):
    return {**claim['base_result'], 'success': False, 'error_code': code, 'error': 'failed'}


def test_submission_is_idempotent_with_stable_input_and_invalid_rows(repo):
    job = submit(repo, [FLIGHT, {'bad': 'row'}])
    assert job == submit(repo, [FLIGHT, {'bad': 'row'}])
    assert job['status'] == 'QUEUED' and job['completed_count'] == 1
    assert job['flights'][1]['result']['error_code'] == 'INVALID_ARGUMENT'
    assert job['flights'][0]['normalized_input']['planned_total_flow'] == 0
    assert len(repo.pending_outbox(100)) == 1
    with pytest.raises(ValueError, match='幂等'):
        submit(repo, [{**FLIGHT, '航班号': 'CZ2222'}])


def test_zggg_snapshot_round_trips_and_retry_reconstructs_exact_context(repo, mysql_settings):
    payload = {**FLIGHT, 'zggg_context': {
        'prediction_cutoff_time': '2025-05-02 09:30:00', 'data_version': 'snapshot-v1',
        'departure_weather': {'metar': {'kind': 'METAR', 'airport': 'ZGGG',
            'issue_time': '2025-05-02T01:20:00Z', 'features': {'visibility_m': 3000}}},
        'flow_features': [{'horizon_minutes': 15, 'planned_takeoff': 2,
                           'planned_landing': 3, 'completed_takeoff': 1,
                           'completed_landing': 1}],
    }}
    job = submit(repo, [payload], key='zggg-snapshot')
    stored = job['flights'][0]['normalized_context']
    assert stored['data_version'] == 'snapshot-v1'
    first = repo.claim(job['job_id'], job['flights'][0]['flight_id'], 'first', 30)
    assert first['normalized_context'] == stored
    expire_lease(mysql_settings, first['flight_id'])
    second = repo.claim(job['job_id'], first['flight_id'], 'second', 30)
    assert second['normalized_context'] == stored


def test_concurrent_claim_and_result_commit_are_fenced(repo, mysql_settings):
    job = submit(repo)
    flight_id = job['flights'][0]['flight_id']
    with ThreadPoolExecutor(2) as pool:
        claims = list(pool.map(lambda owner: repo.claim(job['job_id'], flight_id, owner, 30), ['a', 'b']))
    a = next(c for c in claims if c)
    assert sum(c is not None for c in claims) == 1
    assert repo.heartbeat(a, 30)
    expire_lease(mysql_settings, flight_id)
    b = repo.claim(job['job_id'], flight_id, 'new', 30)
    assert b['fence'] == a['fence'] + 1 and b['trace_id'] != a['trace_id']
    assert not repo.complete(a, failure(a))
    assert repo.complete(b, failure(b))
    assert not repo.complete(b, failure(b))
    final = repo.get(job['job_id'])
    assert final['status'] == 'FAILED' and final['completed_count'] == 1
    assert final['results'][0]['flight_id'] == flight_id
    events = repo.events(job['job_id'])
    assert [e['event_id'] for e in events] == list(range(1, len(events) + 1))
    assert repo.events(job['job_id'], events[-1]['event_id']) == []
    attempts = sql(mysql_settings, 'SELECT status FROM durable_attempts ORDER BY fence')
    assert attempts == (('LOST',), ('FAILED',))


def test_task_deadline_and_attempt_limit_are_persistent(repo, mysql_settings):
    job = submit(repo, key='max', max_attempts=1)
    claim = repo.claim(job['job_id'], job['flights'][0]['flight_id'], 'a', 30)
    expire_lease(mysql_settings, claim['flight_id'])
    assert repo.reconcile() >= 1
    final = repo.get(job['job_id'])
    assert final['results'][0]['error_code'] == 'EXECUTOR_LOST'
    assert not repo.complete(claim, failure(claim))
    expired = submit(repo, key='expiry')
    sql(mysql_settings, 'UPDATE durable_jobs SET deadline_at=UTC_TIMESTAMP(6)-INTERVAL 1 SECOND WHERE job_id=%s', (expired['job_id'],))
    repo.reconcile()
    assert repo.get(expired['job_id'])['status'] == 'EXPIRED'


def test_admission_limit_is_atomic_and_invalid_rows_do_not_queue(repo, mysql_settings):
    sql(mysql_settings, 'UPDATE queue_admission SET capacity=1 WHERE singleton=1')
    with ThreadPoolExecutor(2) as pool:
        def attempt(key):
            try:
                return submit(repo, key=key)
            except ValueError:
                return None
        jobs = list(pool.map(attempt, ['a', 'b']))
    assert sum(j is not None for j in jobs) == 1
    invalid = submit(repo, [{'invalid': True}], key='invalid')
    assert invalid['status'] == 'FAILED'
    assert len(repo.pending_outbox(10)) == 1


def test_submission_failure_rolls_back_every_table(repo, mysql_settings):
    from apps.flight.storage.repository import StorageError
    sql(mysql_settings, "CREATE TRIGGER reject_outbox BEFORE INSERT ON prediction_outbox FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='failure'")
    with pytest.raises(StorageError):
        submit(repo)
    for table in ['durable_jobs', 'durable_flights', 'durable_events', 'prediction_outbox']:
        assert sql(mysql_settings, f'SELECT COUNT(*) FROM {table}')[0][0] == 0


def test_result_failure_rolls_back_counts_events_and_attempt(repo, mysql_settings):
    from apps.flight.storage.repository import StorageError
    job = submit(repo)
    claim = repo.claim(job['job_id'], job['flights'][0]['flight_id'], 'a', 30)
    before = repo.get(job['job_id'])
    events = repo.events(job['job_id'])
    sql(mysql_settings, "CREATE TRIGGER reject_event BEFORE INSERT ON durable_events FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='failure'")
    with pytest.raises(StorageError):
        repo.complete(claim, failure(claim))
    assert repo.get(job['job_id']) == before
    assert repo.events(job['job_id']) == events


def test_reconcile_rebuilds_unfinished_notifications_only(repo, mysql_settings):
    job = submit(repo)
    messages = repo.pending_outbox(10)
    repo.mark_published(messages[0]['message_id'])
    assert repo.pending_outbox(10) == []
    repo.reconcile(republish=True)
    assert repo.pending_outbox(10)[0]['flight_id'] == job['flights'][0]['flight_id']
    claim = repo.claim(job['job_id'], job['flights'][0]['flight_id'], 'a', 30)
    repo.complete(claim, failure(claim))
    repo.reconcile()
    assert repo.pending_outbox(10) == []


def test_rebuild_is_not_limited_to_first_recovery_page(repo, mysql_settings):
    submit(repo, [FLIGHT] * 4)
    sql(mysql_settings, 'UPDATE prediction_outbox SET published_at=UTC_TIMESTAMP(6)')
    repo.reconcile(limit=1, republish=True)
    assert len(repo.pending_outbox(10)) == 4


def test_heartbeat_checks_current_time_after_waiting_for_lock(repo, mysql_settings):
    import pymysql
    job = submit(repo, ttl_seconds=1)
    claim = repo.claim(job['job_id'], job['flights'][0]['flight_id'], 'a', 30)
    conn = pymysql.connect(**mysql_settings, autocommit=False)
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT job_id FROM durable_jobs WHERE job_id=%s FOR UPDATE', (job['job_id'],))
        started = threading.Event()
        def heartbeat():
            started.set()
            return repo.heartbeat(claim, 30)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(heartbeat)
            assert started.wait(1)
            time.sleep(1.3)
            conn.commit()
            assert future.result(5) is False
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.parametrize('change', [
    'ALTER TABLE durable_schema ENGINE=MyISAM',
    'ALTER TABLE durable_flights DROP INDEX flight_order',
    'ALTER TABLE durable_jobs DROP INDEX submission_key',
    'ALTER TABLE durable_events MODIFY document LONGTEXT NOT NULL',
])
def test_durable_schema_drift_is_rejected(repo, mysql_settings, mysql_config, change):
    from apps.flight.tasks.repository import DurableRepository, initialize_durable
    from apps.flight.storage.repository import StorageError
    if 'DROP INDEX flight_order' in change:
        sql(mysql_settings, 'ALTER TABLE durable_flights ADD INDEX job_lookup (job_id)')
    sql(mysql_settings, change)
    with pytest.raises(StorageError):
        DurableRepository(mysql_config)
    with pytest.raises(StorageError):
        initialize_durable(mysql_config)
