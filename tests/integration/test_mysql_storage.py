import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config


def stores(config):
    from apps.flight.storage.mysql_jobs import MySQLJobStore
    return MySQLJobStore(config), MySQLJobStore(config)


def test_two_repositories_share_jobs_and_reopen_results(mysql_config):
    a, b = stores(mysql_config)
    job = a.create(2, 'Asia/Shanghai')
    result = a.finish(job['job_id'], [
        {'success': True, 'source': 'TEST', 'flight_id': 'A'},
        {'success': False, 'error_code': 'UNAVAILABLE', 'flight_id': 'B'},
    ])
    assert b.get(job['job_id']) == result
    assert (result['status'], result['llm_success_count']) == ('PARTIAL', 0)
    assert stores(mysql_config)[0].get(job['job_id']) == result
    assert 'results' not in b.list_jobs()[0]


def test_terminal_results_are_idempotent_and_conflicts_do_not_overwrite(mysql_config):
    a, b = stores(mysql_config)
    job = a.create(1, 'UTC')
    final = a.finish(job['job_id'], [{'success': True, 'source': 'LLM'}])
    assert b.finish(job['job_id'], final['results']) == final
    with pytest.raises(ValueError, match='终态'):
        b.finish(job['job_id'], [{'success': False}])
    assert a.get(job['job_id']) == final


def test_parallel_create_and_finish_remain_isolated(mysql_config):
    a, b = stores(mysql_config)
    with ThreadPoolExecutor(4) as pool:
        jobs = list(pool.map(lambda _: a.create(1, 'UTC'), range(12)))
    assert len({j['job_id'] for j in jobs}) == 12
    with ThreadPoolExecutor(4) as pool:
        finals = list(pool.map(lambda j: b.finish(j['job_id'], [{'success': True, 'source': 'TEST'}]), jobs))
    assert len(a.list_jobs()) == 12
    assert all(a.get(j['job_id']) == j for j in finals)


def test_detail_failure_rolls_back_snapshot_and_all_details(mysql_config, mysql_settings):
    import pymysql
    from apps.flight.storage.repository import StorageError
    a, b = stores(mysql_config)
    job = a.create(2, 'UTC')
    conn = pymysql.connect(**mysql_settings, autocommit=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("CREATE TRIGGER reject_second BEFORE INSERT ON flight_predictions "
                           "FOR EACH ROW BEGIN IF NEW.ordinal=2 THEN SIGNAL SQLSTATE '45000' "
                           "SET MESSAGE_TEXT='injected failure'; END IF; END")
        with pytest.raises(StorageError):
            a.finish(job['job_id'], [{'success': True}, {'success': True}])
        assert b.get(job['job_id']) == job
        with conn.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM flight_predictions WHERE job_id=%s', (job['job_id'],))
            assert cursor.fetchone()[0] == 0
            cursor.execute('DROP TRIGGER reject_second')
        assert a.finish(job['job_id'], [{'success': True}, {'success': True}])['status'] == 'SUCCEEDED'
    finally:
        conn.close()


def test_schema_requires_explicit_initialization(mysql_settings):
    from apps.flight.storage.mysql_jobs import MySQLConfig, MySQLJobStore
    from apps.flight.storage.repository import StorageError
    with pytest.raises(StorageError):
        MySQLJobStore(MySQLConfig(**mysql_settings))


def test_concurrent_same_finalization_commits_one_result(mysql_config, mysql_settings):
    import pymysql
    a, b = stores(mysql_config)
    job = a.create(1, 'UTC')
    result = [{'success': True, 'flight_id': 'A', 'source': 'TEST'}]
    with ThreadPoolExecutor(2) as pool:
        finals = list(pool.map(lambda store: store.finish(job['job_id'], result), [a, b]))
    assert finals[0] == finals[1]
    conn = pymysql.connect(**mysql_settings)
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT document FROM flight_predictions WHERE job_id=%s', (job['job_id'],))
            rows = cursor.fetchall()
            assert len(rows) == 1 and json.loads(rows[0][0]) == result[0]
    finally:
        conn.close()


@pytest.mark.parametrize('change', [
    'ALTER TABLE storage_schema ENGINE=MyISAM',
    'ALTER TABLE prediction_jobs DROP COLUMN created_at',
    'ALTER TABLE flight_predictions DROP INDEX flight_identity',
    'ALTER TABLE flight_predictions DROP FOREIGN KEY flight_predictions_ibfk_1',
])
def test_reject_incompatible_schema(mysql_config, mysql_settings, change):
    import pymysql
    from apps.flight.storage.mysql_jobs import MySQLJobStore, initialize_schema
    from apps.flight.storage.repository import StorageError
    conn = pymysql.connect(**mysql_settings, autocommit=True)
    try:
        with conn.cursor() as cursor:
            # The created_at index must be removed before its column can be dropped.
            if 'DROP COLUMN' in change:
                cursor.execute('ALTER TABLE prediction_jobs DROP INDEX jobs_created')
            cursor.execute(change)
        with pytest.raises(StorageError):
            MySQLJobStore(mysql_config)
        with pytest.raises(StorageError):
            initialize_schema(mysql_config)
    finally:
        conn.close()
