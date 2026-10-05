from copy import deepcopy

import pytest

from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.test_mysql_app import app_config
from tests.integration.test_prediction_flow import gateway_address, FLIGHT
from tests.unit.test_recovery_service import prediction_job
from tests.unit.test_recovery_engine import scenario


@pytest.fixture
def recovery_repo(mysql_config):
    from apps.flight.recovery.repository import RecoveryRepository, initialize_schema
    initialize_schema(mysql_config)
    return RecoveryRepository(mysql_config)


def test_schema_requires_explicit_initialization(mysql_config):
    from apps.flight.recovery.repository import RecoveryRepository
    from apps.flight.storage.repository import StorageError
    with pytest.raises(StorageError): RecoveryRepository(mysql_config)


def test_recovery_is_shared_idempotent_and_immutable(recovery_repo, mysql_config):
    from apps.flight.recovery.repository import RecoveryRepository, RecoveryConflict
    from apps.flight.recovery.service import build_recovery
    plan = build_recovery(prediction_job(), scenario(), 'v1')
    one = recovery_repo.save(plan, 'submit-a')
    assert one['recovery_id'] and one['status'] == 'SUCCEEDED'
    other = RecoveryRepository(mysql_config)
    assert other.get(one['recovery_id']) == one
    assert other.save(plan, 'submit-a') == one
    plan['scenario']['arrival_capacity'] = 2
    # Repository must re-audit and digest the actual document, not trust supplied digest.
    with pytest.raises(RecoveryConflict): other.save(plan, 'submit-a')


def test_repository_rejects_tampered_valid_plan(recovery_repo):
    from apps.flight.recovery.service import build_recovery
    plan = build_recovery(prediction_job(), scenario(), 'v1')
    plan['assignments'][0]['delay_minutes'] = -999
    with pytest.raises(ValueError): recovery_repo.save(plan, 'submit-a')


def test_recovery_idempotency_and_download_bind_replay_snapshot(recovery_repo):
    from apps.flight.recovery.repository import RecoveryConflict
    from apps.flight.recovery.service import build_recovery

    job = prediction_job()
    job['results'][0]['source'] = 'BASELINE'
    job['submission_context'] = {
        'replay_dataset_id': 'may-v1', 'snapshot_hash': 'snapshot-a',
        'dataset_manifest_hash': 'dataset-a', 'feature_contract_version': 'zggg-airport-flow-v1',
        'release_identity': {'release_version': 'release-a', 'source': 'BASELINE'}}
    first = build_recovery(job, scenario(), 'v1')
    saved = recovery_repo.save(first, 'same-key')
    assert saved['source_context']['snapshot_hash'] == 'snapshot-a'
    assert saved['summary']['scheduled_count'] == 1
    changed = deepcopy(job)
    changed['submission_context']['snapshot_hash'] = 'snapshot-b'
    with pytest.raises(RecoveryConflict):
        recovery_repo.save(build_recovery(changed, scenario(), 'v1'), 'same-key')
    forged = deepcopy(first)
    forged['summary']['scheduled_count'] = 999
    with pytest.raises(ValueError, match='summary'):
        recovery_repo.save(forged, 'tampered-summary')


def test_concurrent_same_key_has_one_effective_recovery(recovery_repo):
    from concurrent.futures import ThreadPoolExecutor
    from apps.flight.recovery.service import build_recovery
    plan = build_recovery(prediction_job(), scenario(), 'v1')
    with ThreadPoolExecutor(max_workers=4) as pool:
        documents = list(pool.map(lambda _: recovery_repo.save(plan, 'concurrent-a'), range(8)))
    assert len({document['recovery_id'] for document in documents}) == 1


@pytest.mark.parametrize('mutation', [
    'ALTER TABLE recovery_jobs MODIFY document LONGTEXT NOT NULL',
    'ALTER TABLE recovery_jobs MODIFY submission_key VARCHAR(128) COLLATE utf8mb4_general_ci NOT NULL',
    'ALTER TABLE recovery_jobs DROP PRIMARY KEY',
    'ALTER TABLE recovery_jobs ENGINE=MyISAM',
])
def test_incompatible_schema_is_rejected_at_startup(recovery_repo, mysql_settings, mysql_config, mutation):
    from apps.flight.recovery.repository import RecoveryRepository
    from apps.flight.storage.repository import StorageError
    from tests.integration.test_durable_storage import sql
    sql(mysql_settings, mutation)
    with pytest.raises(StorageError): RecoveryRepository(mysql_config)


@pytest.fixture
def recovery_app(mysql_config, recovery_repo, tmp_path, gateway_address):
    from apps.flight.app import create_app
    app = create_app({**app_config(mysql_config, tmp_path, gateway_address), 'RECOVERY_ENABLED': True})
    yield app
    app.extensions['prediction_client'].close()


def test_http_fixed_prediction_job_and_download(recovery_app):
    client = recovery_app.test_client()
    job = client.post('/predict_flights_batch', json={'flights': [FLIGHT, {'bad': 'input'}]}).json
    cfg = {**scenario(), 'horizon_start': '2025-05-02T08:00:00+08:00', 'horizon_end': '2025-05-02T16:00:00+08:00'}
    body = dict(source_job_id=job['job_id'], scenario=cfg, scenario_version='v1')
    r = client.post('/api/v1/recovery-jobs', json=body, headers={'Idempotency-Key': 'http-a'})
    assert r.status_code == 201
    plan = r.json
    assert plan['status'] == 'SUCCEEDED' and len(plan['excluded']) == 1
    assert plan['source_job_id'] == job['job_id'] and plan['provenance'][0]['source'] == 'TEST'
    assert client.post('/api/v1/recovery-jobs', json=body, headers={'Idempotency-Key': 'http-a'}).json['recovery_id'] == plan['recovery_id']
    assert client.get(plan['status_url']).json['recovery_id'] == plan['recovery_id']
    download = client.get(plan['download_url'])
    assert download.status_code == 200 and b'recovery-greedy-v1' in download.data
    assert client.get('/api/v1/recovery-capabilities').json['enabled'] is True
    assert client.post('/api/v1/recovery-jobs', json={**body, 'filename': '../bad'}, headers={'Idempotency-Key': 'http-b'}).status_code == 400


def test_disabled_recovery_is_explicit(tmp_path, gateway_address):
    from apps.flight.app import create_app
    app = create_app({'TESTING': True, 'RUNTIME_ROOT': str(tmp_path), 'GATEWAY_ADDRESS': gateway_address})
    try:
        assert app.test_client().get('/api/v1/recovery-capabilities').json['enabled'] is False
        assert app.test_client().post('/api/v1/recovery-jobs', json={}).status_code == 503
    finally: app.extensions['prediction_client'].close()


def test_storage_failure_is_safe_503(recovery_app, mysql_settings):
    from tests.integration.test_durable_storage import sql
    sql(mysql_settings, 'DROP TABLE recovery_jobs')
    r = recovery_app.test_client().get('/api/v1/recovery-jobs/df7280bd-2b63-4c73-89d3-f1ab21486b53')
    assert r.status_code == 503 and r.json['error_code'] == 'STORAGE_UNAVAILABLE'
