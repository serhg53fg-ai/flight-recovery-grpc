from pathlib import Path

import pytest


def test_default_factory_uses_sqlite(tmp_path):
    from apps.flight.storage.factory import create_job_store
    from apps.flight.storage.jobs import JobStore
    assert isinstance(create_job_store({}, tmp_path), JobStore)


@pytest.mark.parametrize('config', [
    {'STORAGE_BACKEND': 'unknown'},
    {'STORAGE_BACKEND': 'mysql'},
    {'STORAGE_BACKEND': 'mysql', 'MYSQL_DATABASE': 'flight', 'MYSQL_USER': ''},
])
def test_invalid_storage_configuration_does_not_create_sqlite(tmp_path, config):
    from apps.flight.storage.factory import create_job_store
    with pytest.raises(ValueError):
        create_job_store(config, tmp_path)
    assert not list(tmp_path.rglob('jobs.sqlite3'))


def test_mysql_failure_is_safe_and_does_not_fallback(tmp_path):
    from apps.flight.storage.factory import create_job_store, mysql_config
    from apps.flight.storage.repository import StorageError
    config = {'STORAGE_BACKEND': 'mysql', 'MYSQL_DATABASE': 'flight', 'MYSQL_USER': 'worker',
              'MYSQL_PASSWORD': 'test-secret-should-not-appear',
              'MYSQL_UNIX_SOCKET': str(tmp_path / 'absent.sock')}
    assert config['MYSQL_PASSWORD'] not in repr(mysql_config(config))
    with pytest.raises(StorageError) as captured:
        create_job_store(config, tmp_path)
    assert config['MYSQL_PASSWORD'] not in str(captured.value)
    assert not list(tmp_path.rglob('jobs.sqlite3'))


@pytest.mark.parametrize('field,value', [
    ('port', True), ('port', 0), ('timeout', 0), ('timeout', True),
    ('database', 'flight; DROP TABLE jobs'), ('unix_socket', 'relative.sock'),
])
def test_mysql_config_rejects_invalid_values(field, value):
    from apps.flight.storage.mysql_jobs import MySQLConfig
    with pytest.raises(ValueError):
        MySQLConfig(**{'database': 'flight', 'user': 'worker', field: value})
