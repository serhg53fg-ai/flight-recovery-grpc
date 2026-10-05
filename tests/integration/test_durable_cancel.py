import pytest

from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.test_durable_storage import repo, submit, sql, failure
from tests.integration.test_prediction_flow import FLIGHT


def test_queued_cancel_is_terminal_and_idempotent(repo):
    job = submit(repo)
    final = repo.cancel(job['job_id'])
    assert final['status'] == 'CANCELLED' and final['cancel_requested'] is True
    assert final['cancelled_count'] == final['completed_count'] == 1
    assert final['failed_count'] == final['expired_count'] == final['success_count'] == 0
    assert final['results'][0]['error_code'] == 'CANCELLED'
    events = repo.events(job['job_id'])
    assert events[-1]['type'] == 'complete'
    assert repo.cancel(job['job_id']) == final
    assert repo.events(job['job_id']) == events
    assert repo.pending_outbox() == []
    assert repo.claim(job['job_id'], job['flights'][0]['flight_id'], 'a') is None


def test_running_cancel_rejects_heartbeat_and_late_result(repo, mysql_settings):
    job = submit(repo)
    claim = repo.claim(job['job_id'], job['flights'][0]['flight_id'], 'a')
    repo.cancel(job['job_id'])
    assert not repo.heartbeat(claim)
    assert not repo.complete(claim, failure(claim))
    assert sql(mysql_settings, 'SELECT status FROM durable_attempts') == (('CANCELLED',),)


def test_cancel_preserves_previously_committed_results(repo):
    job = submit(repo, [FLIGHT, FLIGHT])
    claim = repo.claim(job['job_id'], job['flights'][0]['flight_id'], 'a')
    result = {**claim['base_result'], 'success': True, 'source': 'TEST', 'model_version': 'test-v1'}
    assert repo.complete(claim, result)
    final = repo.cancel(job['job_id'])
    assert final['status'] == 'CANCELLED'
    assert final['success_count'] == final['cancelled_count'] == 1
    assert final['completed_count'] == 2
    assert final['results'][0] == result


def test_cancel_completed_task_does_not_rewrite_history(repo):
    job = submit(repo)
    claim = repo.claim(job['job_id'], job['flights'][0]['flight_id'], 'a')
    repo.complete(claim, failure(claim))
    before = repo.get(job['job_id'])
    assert repo.cancel(job['job_id']) == before
    assert not before.get('cancel_requested', False)


def test_cancel_event_failure_rolls_back_entire_request(repo, mysql_settings):
    from apps.flight.storage.repository import StorageError
    job = submit(repo)
    before = repo.get(job['job_id'])
    sql(mysql_settings, "CREATE TRIGGER reject_cancel_event BEFORE INSERT ON durable_events FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='failure'")
    with pytest.raises(StorageError):
        repo.cancel(job['job_id'])
    assert repo.get(job['job_id']) == before


def test_event_pages_and_terminal_event_match_counters(repo):
    job = submit(repo, [FLIGHT, {'bad': 'input'}])
    claim = repo.claim(job['job_id'], job['flights'][0]['flight_id'], 'a')
    repo.complete(claim, failure(claim))
    events = []
    cursor = 0
    while True:
        page = repo.read_event_page(job['job_id'], cursor, limit=1)
        events.extend(page['events'])
        if not page['events']:
            break
        cursor = page['events'][-1]['event_id']
    assert len(events) == len({e['event_id'] for e in events})
    assert events[-1]['type'] == 'complete' and events[-1]['summary']['total'] == 2
    assert events[-1]['summary']['failed'] == 2
    assert page['last_event_id'] == cursor and page['job']['completed_count'] == 2
    with pytest.raises(ValueError):
        repo.read_event_page(job['job_id'], cursor+1)
    assert repo.list_jobs()[0]['job_id'] == job['job_id']


def test_snapshot_is_consistent_with_read_committed_server(repo, mysql_settings):
    job = submit(repo)
    previous = sql(mysql_settings, 'SELECT @@GLOBAL.transaction_isolation')[0][0]
    sql(mysql_settings, "SET GLOBAL transaction_isolation='READ-COMMITTED'")
    try:
        with repo._session() as cursor:
            cursor.execute('SELECT event_seq FROM durable_jobs WHERE job_id=%s', (job['job_id'],))
            sequence = cursor.fetchone()['event_seq']
            repo.cancel(job['job_id'])
            cursor.execute('SELECT event_id FROM durable_events WHERE job_id=%s', (job['job_id'],))
            assert max(row['event_id'] for row in cursor.fetchall()) == sequence
    finally:
        sql(mysql_settings, 'SET GLOBAL transaction_isolation=%s', (previous,))
