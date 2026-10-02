import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.redis_fixtures import redis_socket, queue
from tests.integration.test_durable_storage import repo
from tests.integration.test_async_api import async_app, HEADERS
from tests.integration.test_durable_queue import publisher
from tests.integration.test_durable_execution import executor
from tests.integration.test_prediction_flow import gateway_address, FLIGHT
from apps.flight.clients.inference import InferenceClient


def submit_http(app, flights=None):
    with app.test_client() as client:
        return client.post('/api/v1/prediction-jobs', json={'flights': flights or [FLIGHT]}, headers=HEADERS).json['job_id']


def blocks(text):
    result = []
    for block in text.split('\n\n'):
        lines = block.splitlines()
        if not any(line.startswith('data: ') for line in lines):
            continue
        fields = dict(line.split(': ', 1) for line in lines if ': ' in line)
        result.append((int(fields['id']) if 'id' in fields else None, fields.get('event'), json.loads(fields['data'])))
    return result


def test_disconnect_does_not_cancel_and_reconnect_reads_only_missing_events(async_app, repo, queue, gateway_address):
    job_id = submit_http(async_app)
    path = '/api/v1/prediction-jobs/' + job_id + '/events'
    with async_app.test_client() as client:
        response = client.get(path, buffered=False)
        assert response.status_code == 200
        first = blocks(next(response.response).decode())[0]
        response.close()
        assert repo.get(job_id)['status'] == 'QUEUED'
        publisher(repo, queue).run_once()
        rpc = InferenceClient(gateway_address, 1)
        try:
            executor(repo, queue, rpc).run_once()
        finally:
            rpc.close()
        resumed = client.get(path, headers={'Last-Event-ID': str(first[0])})
        events = blocks(resumed.data.decode())
        ids = [item[0] for item in events if item[0] is not None]
        assert ids == list(range(first[0]+1, repo.get(job_id)['last_event_id']+1))
        assert any(kind == 'complete' for _, kind, _ in events)
        assert events[-1][1] == 'snapshot' and events[-1][2]['status'] == 'SUCCEEDED'
        assert 'no-cache' in resumed.headers['Cache-Control']
        assert resumed.headers['X-Accel-Buffering'] == 'no'


@pytest.mark.parametrize('cursor', ['-1', 'NaN', '999999'])
def test_invalid_or_future_cursor_returns_400_before_stream(async_app, cursor):
    job_id = submit_http(async_app)
    with async_app.test_client() as client:
        response = client.get('/api/v1/prediction-jobs/' + job_id + '/events', headers={'Last-Event-ID': cursor})
        assert response.status_code == 400
        assert response.is_json


def test_terminal_snapshot_after_last_cursor_and_header_priority(async_app, repo):
    job_id = submit_http(async_app)
    repo.cancel(job_id)
    last = repo.get(job_id)['last_event_id']
    with async_app.test_client() as client:
        response = client.get('/api/v1/prediction-jobs/' + job_id + '/events?after=9999', headers={'Last-Event-ID': str(last)})
        items = blocks(response.data.decode())
        assert len(items) == 1 and items[0][0] is None and items[0][1] == 'snapshot'
        assert items[0][2]['status'] == 'CANCELLED'


def test_terminal_paged_stream_does_not_omit_last_page(async_app, repo):
    job_id = submit_http(async_app, [{'invalid': i} for i in range(121)])
    with async_app.test_client() as client:
        response = client.get('/api/v1/prediction-jobs/' + job_id + '/events')
        items = blocks(response.data.decode())
        ids = [item[0] for item in items if item[0] is not None]
        assert ids == list(range(1, repo.get(job_id)['last_event_id']+1))
        assert items[-2][1] == 'complete' and items[-1][1] == 'snapshot'


def test_stream_storage_failure_does_not_claim_completion(async_app, monkeypatch):
    from apps.flight.storage.repository import StorageError
    job_id = submit_http(async_app)
    repository = async_app.extensions['durable_repository']
    original = repository.read_event_page
    calls = 0
    def fail_after_preflight(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise StorageError('private database details')
        return original(*args, **kwargs)
    monkeypatch.setattr(repository, 'read_event_page', fail_after_preflight)
    with async_app.test_client() as client:
        response = client.get('/api/v1/prediction-jobs/' + job_id + '/events')
        items = blocks(response.data.decode())
        assert items == [(None, 'dependency_error', {'type': 'dependency_error', 'job_id': job_id,
                         'error_code': 'STORAGE_UNAVAILABLE', 'error': '任务存储暂不可用，请从最后游标重连'})]
        assert 'private database details' not in response.data.decode()


def test_persistent_cancel_propagates_to_running_executor(async_app, repo, queue):
    job_id = submit_http(async_app)
    publisher(repo, queue).run_once()
    entered = threading.Event()
    class Waiting:
        timeout = 3
        cancelled = False
        def predict(self, request, cancel, timeout=None):
            from apps.flight.clients.inference import PredictionError
            entered.set()
            self.cancelled = cancel.wait(3)
            raise PredictionError('CANCELLED', 'cancelled')
    client = Waiting()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(executor(repo, queue, client, lease_seconds=1).run_once)
        assert entered.wait(2)
        with async_app.test_client() as web:
            assert web.post('/api/v1/prediction-jobs/' + job_id + '/cancel').json['status'] == 'CANCELLED'
        assert future.result(3) and client.cancelled
    assert repo.get(job_id)['results'][0]['error_code'] == 'CANCELLED'
    assert queue.client.xlen(queue.stream) == 0
