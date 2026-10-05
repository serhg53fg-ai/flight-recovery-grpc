import pytest

from apps.flight.app import create_app


def test_default_sqlite_reports_async_disabled(tmp_path):
    app = create_app({'RUNTIME_ROOT': str(tmp_path)})
    try:
        client = app.test_client()
        assert client.get('/api/v1/prediction-capabilities').json['enabled'] is False
        response = client.post('/api/v1/prediction-jobs', json={'flights': [{}]}, headers={'Idempotency-Key': 'a'})
        assert response.status_code == 503 and response.json['error_code'] == 'ASYNC_DISABLED'
    finally:
        app.extensions['prediction_client'].close()


@pytest.mark.parametrize('config', [
    {'DURABLE_ENABLED': True},
    {'DURABLE_ENABLED': 'yes'},
    {'DURABLE_ENABLED': True, 'STORAGE_BACKEND': 'mysql'},
    {'DURABLE_ENABLED': True, 'STORAGE_BACKEND': 'mysql', 'DURABLE_MODEL_VERSION': 'test-v1',
     'DURABLE_SOURCE': 'bad', 'DURABLE_PROMPT_VERSION': 'test-v1'},
])
def test_bad_async_configuration_fails_before_database_or_runtime(config, tmp_path):
    with pytest.raises(ValueError):
        create_app({'RUNTIME_ROOT': str(tmp_path / 'unused'), **config})
    assert not (tmp_path / 'unused').exists()
