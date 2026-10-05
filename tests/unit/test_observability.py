"""Stable stage metrics and dependency-aware readiness are separate from liveness."""
import json

import pytest


def test_latency_histogram_buckets_include_observations():
    from deploy.metrics.exporter import render_stage_histograms

    text = render_stage_histograms([
        {'stage': 'rpc', 'outcome': 'success', 'seconds': .12},
        {'stage': 'rpc', 'outcome': 'success', 'seconds': .8},
        {'stage': 'rpc', 'outcome': 'failure', 'seconds': 2.4},
    ])
    assert 'flight_stage_duration_seconds_bucket{stage="rpc",outcome="success",le="0.25"} 1\n' in text
    assert 'flight_stage_duration_seconds_bucket{stage="rpc",outcome="success",le="1"} 2\n' in text
    assert 'flight_stage_duration_seconds_bucket{stage="rpc",outcome="success",le="+Inf"} 2\n' in text
    assert 'flight_stage_duration_seconds_count{stage="rpc",outcome="failure"} 1\n' in text


def test_metric_labels_exclude_request_ids():
    from deploy.metrics.exporter import render_stage_histograms

    output = render_stage_histograms([{'stage': 'rpc', 'outcome': 'success',
                                       'seconds': .1, 'job_id': 'unique-job-123',
                                       'flight_id': 'unique-flight-456', 'trace_id': 'unique-trace-789'}])
    assert 'unique-job-123' not in output
    assert 'unique-flight-456' not in output
    assert 'unique-trace-789' not in output
    assert 'stage="rpc",outcome="success"' in output


def test_record_stage_rejects_unbounded_labels_and_duration(capsys):
    from apps.flight.observability import record_stage

    with pytest.raises(ValueError):
        record_stage('job-id-123', .1, 'success')
    with pytest.raises(ValueError):
        record_stage('rpc', -1, 'success')
    record_stage('rpc', .125, 'success')
    event = json.loads(capsys.readouterr().out)
    assert event == {'event': 'flight.stage', 'stage': 'rpc', 'seconds': .125,
                     'outcome': 'success'}


def test_liveness_ok_readiness_fails_when_dependency_down(tmp_path):
    from apps.flight.app import create_app

    app = create_app({'TESTING': True, 'RUNTIME_ROOT': str(tmp_path),
                      'STORAGE_BACKEND': 'sqlite', 'DURABLE_ENABLED': False})
    app.extensions['readiness_probes'] = {'database': lambda: True,
                                          'queue': lambda: False,
                                          'release': lambda: True,
                                          'worker_pool': lambda: True}
    with app.test_client() as client:
        assert client.get('/health').status_code == 200
        response = client.get('/ready')
        assert response.status_code == 503
        assert response.json['checks']['queue'] is False
    app.extensions['prediction_client'].close()


def test_release_readiness_uses_startup_verified_identity_without_rehash(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from apps.flight.observability import _release_ready
    from deploy.distributed.release import release_identity
    import deploy.distributed.release as releases

    artifact = tmp_path / 'large-model.bin'
    artifact.write_bytes(b'artifact already checked at startup')
    release = {'schema_version': 'zggg-release-v2', 'airport': 'ZGGG',
               'flight_mode': 'historical', 'source': 'BASELINE',
               'release_version': 'release-v1', 'flight_model_version': 'historical-flight-v1',
               'flow_model_version': 'schedule', 'model_version': 'historical-flight-v1+schedule',
               'feature_contract_version': 'zggg-airport-flow-v1', 'prompt_version': 'historical-duration-v1',
               'dataset_manifest_hash': 'dataset-v1', 'flight_gate_passed': True,
               'flow_gate_passed': True, 'artifacts': {
                   name: {'path': str(artifact), 'sha256': 'startup-check'}
                   for name in ('historical_model', 'flow_model', 'feature_contract')}}
    manifest = tmp_path / 'release.json'
    manifest.write_text(json.dumps(release))
    monkeypatch.setattr(releases, 'sha256_path', lambda _: (_ for _ in ()).throw(AssertionError('rehash')))
    app = SimpleNamespace(config={'RELEASE_MANIFEST_PATH': str(manifest),
                                  'RELEASE_IDENTITY': release_identity(release)})
    assert _release_ready(app) is True


def test_experimental_release_is_ready_only_when_deployment_explicitly_allows_it(tmp_path):
    from types import SimpleNamespace
    from apps.flight.observability import _release_ready
    from deploy.distributed.release import release_identity

    artifact = tmp_path / 'artifact.bin'
    artifact.write_bytes(b'verified during resident preflight')
    release = {'schema_version': 'zggg-release-v2', 'airport': 'ZGGG',
               'deployment_stage': 'experimental', 'flight_mode': 'qwen', 'source': 'LLM',
               'release_version': 'experimental-v1', 'flight_model_version': 'qwen+adapter',
               'flow_model_version': 'gradient_boosting',
               'model_version': 'qwen+adapter+gradient_boosting',
               'feature_contract_version': 'zggg-airport-flow-v1',
               'prompt_version': 'flight-duration-prompt-v1',
               'dataset_manifest_hash': 'dataset-v1', 'flight_gate_passed': False,
               'flow_gate_passed': True,
               'fallback_identity': {'source': 'BASELINE',
                                     'flight_model_version': 'historical-flight-v1',
                                     'prompt_version': 'historical-duration-v1'},
               'artifacts': {'model': {'path': str(artifact), 'sha256': 'startup-check'}}}
    manifest = tmp_path / 'release.json'
    manifest.write_text(json.dumps(release))
    config = {'RELEASE_MANIFEST_PATH': str(manifest),
              'RELEASE_IDENTITY': release_identity(release),
              'EXPERIMENTAL_MODEL': True}
    assert _release_ready(SimpleNamespace(config=config)) is True
    config['EXPERIMENTAL_MODEL'] = False
    assert _release_ready(SimpleNamespace(config=config)) is False


def test_explicit_durable_test_demo_is_ready_without_model_artifacts():
    from types import SimpleNamespace
    from apps.flight.observability import _release_ready
    app = SimpleNamespace(config={'DURABLE_ENABLED': True, 'DURABLE_SOURCE': 'TEST',
                                  'DURABLE_MODEL_VERSION': 'test-v1',
                                  'DURABLE_PROMPT_VERSION': 'test-v1'})
    assert _release_ready(app) is True


@pytest.mark.parametrize('override', [
    {'DURABLE_ENABLED': False}, {'DURABLE_SOURCE': 'LLM'},
    {'DURABLE_SOURCE': 'BASELINE'}, {'DURABLE_MODEL_VERSION': 'qwen-model'},
    {'DURABLE_PROMPT_VERSION': 'flight-duration-prompt-v1'},
    {'RELEASE_MANIFEST_PATH': '/nonexistent/release.json'},
    {'REPLAY_DATASET_ID': 'historical-replay'},
])
def test_non_demo_identity_cannot_bypass_release_validation(override):
    from types import SimpleNamespace
    from apps.flight.observability import _release_ready
    config = {'DURABLE_ENABLED': True, 'DURABLE_SOURCE': 'TEST',
              'DURABLE_MODEL_VERSION': 'test-v1', 'DURABLE_PROMPT_VERSION': 'test-v1'}
    config.update(override)
    assert _release_ready(SimpleNamespace(config=config)) is False
