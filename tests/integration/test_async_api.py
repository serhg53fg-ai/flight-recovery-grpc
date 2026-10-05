import pytest

from apps.flight.app import create_app
from apps.flight.clients.inference import InferenceClient
from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.redis_fixtures import redis_socket, queue
from tests.integration.test_durable_storage import repo, sql
from tests.integration.test_durable_queue import publisher
from tests.integration.test_durable_execution import executor
from tests.integration.test_prediction_flow import gateway_address, FLIGHT, excel_file
from tests.integration.test_mysql_app import app_config

HEADERS = {'Idempotency-Key': 'http-submit'}


@pytest.fixture
def async_app(repo, mysql_config, tmp_path, gateway_address):
    app = create_app({**app_config(mysql_config, tmp_path, gateway_address), 'DURABLE_ENABLED': True,
                      'DURABLE_MODEL_VERSION': 'test-v1', 'DURABLE_SOURCE': 'TEST',
                      'DURABLE_PROMPT_VERSION': 'test-v1'})
    yield app
    app.extensions['prediction_client'].close()


def test_202_acceptance_only_writes_database_and_is_idempotent(async_app, repo):
    with async_app.test_client() as client:
        first = client.post('/api/v1/prediction-jobs', json={'flights': [FLIGHT]}, headers=HEADERS)
        assert first.status_code == 202
        assert first.json['accepted'] is True and first.json['status'] == 'QUEUED'
        assert repo.get(first.json['job_id'])['completed_count'] == 0
        assert client.post('/api/v1/prediction-jobs', json={'flights': [FLIGHT]}, headers=HEADERS).json['job_id'] == first.json['job_id']
        conflict = client.post('/api/v1/prediction-jobs', json={'flights': [{**FLIGHT, '航班号': 'CZ2222'}]}, headers=HEADERS)
        assert conflict.status_code == 409
        assert client.get('/api/v1/prediction-jobs').json['jobs'][0]['job_id'] == first.json['job_id']
        assert client.get('/api/jobs').json['jobs'][0]['job_id'] == first.json['job_id']


@pytest.mark.parametrize('body,headers', [
    ({'flights': [FLIGHT]}, {}), ({'flights': []}, HEADERS),
    ({'flights': [FLIGHT], 'source': 'LLM'}, HEADERS),
])
def test_invalid_async_request_rejected(async_app, body, headers):
    with async_app.test_client() as client:
        assert client.post('/api/v1/prediction-jobs', json=body, headers=headers).status_code == 400


def test_queue_backlog_is_http_429(async_app, mysql_settings):
    sql(mysql_settings, 'UPDATE queue_admission SET capacity=1 WHERE singleton=1')
    with async_app.test_client() as client:
        assert client.post('/api/v1/prediction-jobs', json={'flights': [FLIGHT, FLIGHT]}, headers=HEADERS).status_code == 429


def test_spreadsheet_async_execution_and_old_views(async_app, repo, queue, gateway_address):
    with async_app.test_client() as client:
        response = client.post('/api/v1/prediction-jobs/upload', data={'file': excel_file([FLIGHT])}, headers=HEADERS)
        assert response.status_code == 202
        job_id = response.json['job_id']
        publisher(repo, queue).run_once()
        rpc = InferenceClient(gateway_address, 1)
        try:
            executor(repo, queue, rpc).run_once()
        finally:
            rpc.close()
        result = client.get('/api/v1/prediction-jobs/' + job_id)
        assert result.json['status'] == 'SUCCEEDED' and result.json['results'][0]['source'] == 'TEST'
        assert client.get('/api/jobs/' + job_id).json['status'] == 'SUCCEEDED'
        assert client.get('/api/jobs/' + job_id + '/download').status_code == 200
        assert client.get('/api/jobs/' + job_id + '/timeline').status_code == 200
        assert client.get('/api/flight-data?job_id=' + job_id).json['job_id'] == job_id


def test_other_web_instance_reads_and_cancels_job(async_app, mysql_config, tmp_path, gateway_address):
    with async_app.test_client() as client:
        job_id = client.post('/api/v1/prediction-jobs', json={'flights': [FLIGHT]}, headers=HEADERS).json['job_id']
    config = {**app_config(mysql_config, tmp_path / 'b', gateway_address),
              **{k: async_app.config[k] for k in ('DURABLE_ENABLED', 'DURABLE_MODEL_VERSION', 'DURABLE_SOURCE', 'DURABLE_PROMPT_VERSION')}}
    other = create_app(config)
    try:
        with other.test_client() as client:
            assert client.get('/api/jobs/' + job_id).status_code == 200
            cancel = client.post('/api/v1/prediction-jobs/' + job_id + '/cancel')
            assert cancel.status_code == 200 and cancel.json['status'] == 'CANCELLED'
    finally:
        other.extensions['prediction_client'].close()


def test_async_storage_failure_is_safe_503(async_app, mysql_settings):
    sql(mysql_settings, 'RENAME TABLE durable_jobs TO unavailable_durable_jobs')
    with async_app.test_client() as client:
        response = client.post('/api/v1/prediction-jobs', json={'flights': [FLIGHT]}, headers=HEADERS)
        assert response.status_code == 503 and response.json['error_code'] == 'STORAGE_UNAVAILABLE'
        assert mysql_settings['database'] not in response.get_data(as_text=True)
