import hashlib
import json

import pytest

from apps.flight.app import create_app
from apps.flight.storage.jobs import JobStore
from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.test_prediction_flow import gateway_address, FLIGHT


def app_config(config, root, gateway):
    return {'TESTING': True, 'RUNTIME_ROOT': str(root), 'GATEWAY_ADDRESS': gateway,
            'RPC_TIMEOUT_SECONDS': 1, 'STORAGE_BACKEND': 'mysql',
            'MYSQL_DATABASE': config.database, 'MYSQL_USER': config.user,
            'MYSQL_UNIX_SOCKET': config.unix_socket}


def test_two_apps_share_prediction_download_and_timeline(mysql_config, tmp_path, gateway_address):
    a = create_app(app_config(mysql_config, tmp_path / 'a', gateway_address))
    b = create_app(app_config(mysql_config, tmp_path / 'b', gateway_address))
    try:
        with a.test_client() as client:
            response = client.post('/predict_flight', json=FLIGHT)
            assert response.status_code == 200
            result = response.get_json()
        with b.test_client() as client:
            job = client.get('/api/jobs/' + result['job_id'])
            assert job.status_code == 200
            assert job.get_json()['results'][0]['trace_id'] == result['trace_id']
            assert job.get_json()['llm_success_count'] == 0
            assert client.get('/api/jobs/' + result['job_id'] + '/download').status_code == 200
            assert client.get('/api/jobs/' + result['job_id'] + '/timeline').status_code == 200
        assert not list(tmp_path.rglob('jobs.sqlite3'))
    finally:
        a.extensions['prediction_client'].close()
        b.extensions['prediction_client'].close()


def test_readonly_sqlite_migration_preserves_documents_and_skips_duplicates(mysql_config, tmp_path):
    from apps.flight.storage.mysql_jobs import MySQLJobStore
    from scripts.migrate_jobs import migrate_sqlite
    source = JobStore(tmp_path / 'source')
    j = source.create(2, 'Asia/Shanghai')
    final = source.finish(j['job_id'], [{'success': True, 'source': 'TEST'},
                                      {'success': False, 'error_code': 'UNAVAILABLE'}])
    digest = hashlib.sha256(source.db.read_bytes()).hexdigest()
    original_files = {p.name: p.read_bytes() for p in source.db.parent.iterdir() if p.is_file()}
    target = MySQLJobStore(mysql_config)
    assert migrate_sqlite(source.db, target) == {'migrated': 1, 'skipped': 0}
    assert migrate_sqlite(source.db, target) == {'migrated': 0, 'skipped': 1}
    assert target.get(j['job_id']) == final
    assert hashlib.sha256(source.db.read_bytes()).hexdigest() == digest
    assert {p.name: p.read_bytes() for p in source.db.parent.iterdir() if p.is_file()} == original_files
    source2 = JobStore(tmp_path / 'source2')
    source2.import_job({**final, 'fallback_reason': 'different'})
    with pytest.raises(ValueError, match='冲突'):
        migrate_sqlite(source2.db, target)
    assert target.get(j['job_id']) == final


def test_migration_refuses_running_jobs_and_absent_source(mysql_config, tmp_path):
    from apps.flight.storage.mysql_jobs import MySQLJobStore
    from scripts.migrate_jobs import migrate_sqlite
    source = JobStore(tmp_path / 'source')
    source.create(1, 'UTC')
    target = MySQLJobStore(mysql_config)
    with pytest.raises(ValueError, match='RUNNING'):
        migrate_sqlite(source.db, target)
    absent = tmp_path / 'absent.sqlite3'
    with pytest.raises(ValueError):
        migrate_sqlite(absent, target)
    assert not absent.exists() and target.list_jobs() == []


def test_migration_cli_initializes_and_reports_conflicts(mysql_config, tmp_path, monkeypatch, capsys):
    from scripts.migrate_jobs import main
    for key, value in {'DATABASE': mysql_config.database, 'USER': mysql_config.user,
                       'UNIX_SOCKET': mysql_config.unix_socket}.items():
        monkeypatch.setenv('FLIGHT_MYSQL_' + key, value)
    assert main(['--initialize']) == 0
    assert json.loads(capsys.readouterr().out)['initialized'] is True
    source = JobStore(tmp_path / 'source')
    job = source.create(1, 'UTC')
    final = source.finish(job['job_id'], [{'success': True, 'source': 'TEST'}])
    assert main(['--source-sqlite', str(source.db)]) == 0
    assert json.loads(capsys.readouterr().out)['migrated'] == 1
    conflict = JobStore(tmp_path / 'conflict')
    conflict.import_job({**final, 'fallback_reason': 'conflicting'})
    assert main(['--source-sqlite', str(conflict.db)]) == 2
    assert '不会覆盖' in capsys.readouterr().out


def test_migration_reads_committed_wal_without_touching_source(mysql_config, tmp_path):
    import sqlite3
    from apps.flight.storage.mysql_jobs import MySQLJobStore
    from scripts.migrate_jobs import migrate_sqlite
    source = JobStore(tmp_path / 'source')
    job = source.create(1, 'UTC')
    final = source.finish(job['job_id'], [{'success': True, 'source': 'TEST'}])
    conn = sqlite3.connect(source.db)
    try:
        conn.execute('PRAGMA wal_autocheckpoint=0')
        final['fallback_reason'] = 'committed in WAL'
        conn.execute('UPDATE jobs SET document=? WHERE job_id=?', (json.dumps(final), final['job_id']))
        conn.commit()
        files = {p.name: p.read_bytes() for p in source.db.parent.iterdir()}
        assert files['jobs.sqlite3-wal']
        target = MySQLJobStore(mysql_config)
        assert migrate_sqlite(source.db, target)['migrated'] == 1
        assert target.get(final['job_id']) == final
        assert {p.name: p.read_bytes() for p in source.db.parent.iterdir()} == files
    finally:
        conn.close()


def test_sse_storage_failure_emits_error_instead_of_complete(mysql_config, mysql_settings, tmp_path, gateway_address):
    import pymysql
    app = create_app(app_config(mysql_config, tmp_path, gateway_address))
    conn = pymysql.connect(**mysql_settings, autocommit=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("CREATE TRIGGER fail_detail BEFORE INSERT ON flight_predictions "
                           "FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='failure'")
        with app.test_client() as client:
            response = client.post('/predict_flights_stream', json={'flights': [FLIGHT]})
            events = [json.loads(x[6:]) for x in response.data.decode().split('\n\n') if x.startswith('data: ')]
            assert events[-1]['type'] == 'error'
            assert events[-1]['error_code'] == 'STORAGE_UNAVAILABLE'
            assert not any(event.get('type') == 'complete' for event in events)
            job_id = events[-1]['job_id']
            assert app.extensions['job_store'].get(job_id)['status'] == 'RUNNING'
    finally:
        conn.close()
        app.extensions['prediction_client'].close()


def test_storage_error_is_http_503_and_does_not_leak_driver_details(mysql_config, mysql_settings, tmp_path, gateway_address):
    import pymysql
    app = create_app(app_config(mysql_config, tmp_path, gateway_address))
    conn = pymysql.connect(**mysql_settings, autocommit=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute('RENAME TABLE prediction_jobs TO temporarily_unavailable')
        with app.test_client() as client:
            response = client.get('/api/jobs')
            assert response.status_code == 503
            assert response.get_json()['error_code'] == 'STORAGE_UNAVAILABLE'
            assert mysql_config.database not in response.get_data(as_text=True)
    finally:
        conn.close()
        app.extensions['prediction_client'].close()
