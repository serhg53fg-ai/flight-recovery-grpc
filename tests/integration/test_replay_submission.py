import json
import shutil

import pytest

from apps.flight.app import create_app
from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.test_durable_storage import repo
from tests.integration.test_mysql_app import app_config
from tests.integration.test_prediction_flow import gateway_address
from tests.unit.test_context_snapshot import FLIGHT, replay_dataset
from tests.unit.test_release_identity import _historical_input


@pytest.fixture
def replay_app(repo, mysql_config, tmp_path, gateway_address):
    from scripts.zggg_release import stage_candidate

    _, directory = replay_dataset(tmp_path)
    artifacts = _historical_input(tmp_path)
    manifest = tmp_path / 'dataset' / 'manifest.json'
    value = json.loads(manifest.read_text())
    value['manifest_hash'] = 'dataset-hash-v1'
    manifest.write_text(json.dumps(value))
    release = tmp_path / 'release'
    stage_candidate(artifacts, release)
    config = {**app_config(mysql_config, tmp_path / 'web', gateway_address),
              'DURABLE_ENABLED': True,
              'DURABLE_MODEL_VERSION': 'historical-flight-v1+gradient_boosting',
              'DURABLE_SOURCE': 'BASELINE',
              'DURABLE_PROMPT_VERSION': 'historical-duration-v1',
              'REPLAY_ROOT': str(directory.parent), 'REPLAY_DATASET_ID': 'may-v1',
              'RELEASE_MANIFEST_PATH': str(release / 'candidate.json')}
    app = create_app(config)
    yield app, directory
    app.extensions['prediction_client'].close()


def test_replay_submission_persists_context_and_is_idempotent(replay_app, repo):
    app, _ = replay_app
    with app.test_client() as client:
        capabilities = client.get('/api/v1/prediction-capabilities').json
        assert capabilities['replay_dataset_id'] == 'may-v1'
        listing = client.get('/api/v1/replay-datasets')
        assert listing.status_code == 200
        assert listing.json['datasets'][0]['dataset_id'] == 'may-v1'
        assert 'source_file' not in listing.get_data(as_text=True)
        body = {'flights': [FLIGHT], 'replay_dataset_id': 'may-v1'}
        first = client.post('/api/v1/prediction-jobs', json=body, headers={'Idempotency-Key': 'replay-key'})
        assert first.status_code == 202
        job = repo.get(first.json['job_id'])
        assert job['submission_context']['replay_dataset_id'] == 'may-v1'
        assert len(job['submission_context']['snapshot_hash']) == 64
        assert job['submission_context']['release_identity']['source'] == 'BASELINE'
        assert len(job['flights'][0]['normalized_context']['flow_features']) == 3
        assert client.post('/api/v1/prediction-jobs', json=body,
                           headers={'Idempotency-Key': 'replay-key'}).json['job_id'] == first.json['job_id']


def test_replay_export_columns_preserve_snapshot_and_release_identity(replay_app, repo):
    from apps.flight.api.predictions import frame_for_job
    app, _ = replay_app
    with app.test_client() as client:
        accepted = client.post('/api/v1/prediction-jobs', json={'flights': [FLIGHT]},
                               headers={'Idempotency-Key': 'export-context'})
        assert accepted.status_code == 202
        job = repo.get(accepted.json['job_id'])
        row = dict(success=True, prediction_data={
            '实际离港时间': '2025-05-15T09:40:00+08:00',
            '实际到港时间': '2025-05-15T11:45:00+08:00'},
            source='BASELINE', model_version='historical-flight-v1+gradient_boosting',
            airport_flow=None, trace_id='trace', flight_id='flight')
        row['flight_info'] = {'航班号': 'CZ9933', '机尾号': 'B20E8', '机型': 'A320',
                              '计划起飞站四字码': 'ZGGG', '计划到达站四字码': 'ZSPD',
                              '计划离港时间': '2025-05-15T09:30:00+08:00',
                              '计划到港时间': '2025-05-15T11:45:00+08:00'}
        exported = frame_for_job({**job, 'results': [row]}).iloc[0]
        assert exported['快照哈希'] == job['submission_context']['snapshot_hash']
        assert exported['数据集ID'] == 'may-v1'
        assert exported['发布版本'] == job['submission_context']['release_identity']['release_version']


def test_same_key_different_replay_dataset_conflicts(replay_app):
    app, directory = replay_app
    copy = directory.parent / 'may-v2'; shutil.copytree(directory, copy)
    manifest = json.loads((copy / 'replay.json').read_text())
    manifest['dataset_id'] = 'may-v2'
    (copy / 'replay.json').write_text(json.dumps(manifest))
    with app.test_client() as client:
        listed = client.get('/api/v1/replay-datasets').json['datasets']
        assert {entry['dataset_id'] for entry in listed} == {'may-v1', 'may-v2'}
        headers = {'Idempotency-Key': 'same-key'}
        assert client.post('/api/v1/prediction-jobs', json={'flights': [FLIGHT],
            'replay_dataset_id': 'may-v1'}, headers=headers).status_code == 202
        response = client.post('/api/v1/prediction-jobs', json={'flights': [FLIGHT],
            'replay_dataset_id': 'may-v2'}, headers=headers)
        assert response.status_code == 409


def test_bad_replay_input_creates_no_job(replay_app, repo):
    app, _ = replay_app
    with app.test_client() as client:
        before = len(repo.list_jobs())
        invalid = [
            {**FLIGHT, 'zggg_context': {'flow_features': []}},
            {**FLIGHT, '计划离港时间': '2025-05-15T00:30:00Z'},
        ]
        for index, flight in enumerate(invalid):
            result = client.post('/api/v1/prediction-jobs', json={'flights': [flight]},
                                 headers={'Idempotency-Key': f'invalid-{index}'})
            assert result.status_code == 400
        assert len(repo.list_jobs()) == before


def test_legacy_async_job_remains_readable(replay_app, repo):
    app, _ = replay_app
    job = repo.submit([FLIGHT], 'Asia/Shanghai', 'legacy-key',
                      'historical-flight-v1+gradient_boosting', 'BASELINE', 'historical-duration-v1')
    with app.test_client() as client:
        response = client.get('/api/v1/prediction-jobs/' + job['job_id'])
        assert response.status_code == 200
        assert response.json['job_id'] == job['job_id']
