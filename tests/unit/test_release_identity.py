from pathlib import Path

import pytest

from tests.training.test_zggg_release import release_input


def _historical_input(tmp_path: Path) -> dict:
    value = release_input(tmp_path)
    value['flight_mode'] = 'historical'
    value['flight_model_version'] = 'historical-flight-v1'
    value['flow_model_version'] = 'gradient_boosting'
    value['feature_contract_version'] = 'zggg-airport-flow-v1'
    value['prompt_version'] = 'historical-duration-v1'
    del value['artifacts']['model']
    del value['artifacts']['adapter']
    return value


def test_historical_release_identity_matches_worker_output(tmp_path):
    from deploy.distributed.release import load_release, release_identity
    from scripts.zggg_release import stage_candidate

    directory = tmp_path / 'release'
    stage_candidate(_historical_input(tmp_path), directory)
    release = load_release(directory / 'candidate.json')
    assert release['schema_version'] == 'zggg-release-v2'
    assert release_identity(release) == {
        'release_version': 'zggg-20260923-01',
        'source': 'BASELINE',
        'model_version': 'historical-flight-v1+gradient_boosting',
        'prompt_version': 'historical-duration-v1',
        'flight_model_version': 'historical-flight-v1',
        'flow_model_version': 'gradient_boosting',
        'feature_contract_version': 'zggg-airport-flow-v1',
        'dataset_manifest_hash': 'dataset-v1',
    }


def test_v1_requires_explicit_migration(tmp_path):
    from deploy.distributed.release import load_release
    from scripts.zggg_release import stage_candidate

    directory = tmp_path / 'release'
    stage_candidate(release_input(tmp_path), directory)
    with pytest.raises(ValueError, match='migration'):
        load_release(directory / 'candidate.json')


def test_v1_can_be_explicitly_migrated_to_historical(tmp_path):
    from deploy.distributed.release import load_release
    from scripts.zggg_release import migrate_candidate, stage_candidate

    old_dir = tmp_path / 'old-release'
    stage_candidate(release_input(tmp_path), old_dir)
    output = tmp_path / 'new-release'
    migrate_candidate(old_dir / 'candidate.json', output, flight_mode='historical',
                      flight_model_version='historical-flight-v1',
                      flow_model_version='gradient_boosting',
                      feature_contract_version='zggg-airport-flow-v1',
                      prompt_version='historical-duration-v1')
    migrated = load_release(output / 'candidate.json')
    assert migrated['flight_mode'] == 'historical'
    assert 'model' not in migrated['artifacts']
    assert 'adapter' not in migrated['artifacts']
    assert (old_dir / 'candidate.json').exists()


def test_release_hash_mismatch_is_rejected_before_deployment(tmp_path):
    from deploy.distributed.release import load_release
    from scripts.zggg_release import stage_candidate

    directory = tmp_path / 'release'
    stage_candidate(_historical_input(tmp_path), directory)
    (tmp_path / 'flow_model.bin').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash'):
        load_release(directory / 'candidate.json')


def test_historical_mode_does_not_require_cuda_or_qwen(tmp_path):
    from deploy.autodl.worker import parse_args, preflight, build_server_argv
    import socket

    flow = tmp_path / 'flow.pkl'; flow.write_bytes(b'flow')
    historical = tmp_path / 'historical.pkl'; historical.write_bytes(b'history')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    args = parse_args(['--backend', 'composite', '--composite-flight-mode', 'historical',
                       '--listen', f'127.0.0.1:{port}', '--worker-id', 'cpu-history',
                       '--flow-model-path', str(flow), '--historical-model-path', str(historical),
                       '--flow-manifest-hash', 'dataset-v1'])
    environment = preflight(args, torch_module=None)
    argv = build_server_argv(args)
    assert environment.gpu_name == 'CPU'
    assert '--model-path' not in argv
    assert argv[argv.index('--composite-flight-mode') + 1] == 'historical'


def test_resident_uses_historical_release_identity(tmp_path):
    from deploy.distributed.resident import commands, environment, validate
    from scripts.zggg_release import stage_candidate
    from tests.unit.test_resident import settings

    release_dir = tmp_path / 'release'
    stage_candidate(_historical_input(tmp_path), release_dir)
    config = validate({**settings(tmp_path / 'runtime'), 'backend': 'composite',
                       'release_manifest_path': str(release_dir / 'candidate.json')})
    env = environment(config)
    worker = commands(config)['worker']
    assert env['FLIGHT_DURABLE_SOURCE'] == 'BASELINE'
    assert env['FLIGHT_DURABLE_MODEL_VERSION'] == 'historical-flight-v1+gradient_boosting'
    assert env['FLIGHT_DURABLE_PROMPT_VERSION'] == 'historical-duration-v1'
    assert worker[worker.index('--composite-flight-mode') + 1] == 'historical'
    assert '--model-path' not in worker
    assert worker[worker.index('--expected-model-version') + 1] == 'historical-flight-v1+gradient_boosting'
    assert worker[worker.index('--expected-source') + 1] == 'BASELINE'


def test_resident_replay_configuration_reaches_web_and_matches_release(tmp_path):
    from deploy.distributed.resident import validate, environment
    from scripts.zggg_release import stage_candidate
    from tests.unit.test_context_snapshot import replay_dataset
    from tests.unit.test_resident import settings

    _, dataset_dir = replay_dataset(tmp_path)
    inputs = _historical_input(tmp_path)
    manifest = tmp_path / 'dataset' / 'manifest.json'
    import json
    value = json.loads(manifest.read_text())
    value['manifest_hash'] = 'dataset-hash-v1'
    manifest.write_text(json.dumps(value))
    release = tmp_path / 'release'
    stage_candidate(inputs, release)
    config = validate({**settings(tmp_path / 'runtime'), 'backend': 'composite',
                       'release_manifest_path': str(release / 'candidate.json'),
                       'replay_root': str(dataset_dir.parent), 'replay_dataset_id': 'may-v1'})
    env = environment(config)
    assert env['FLIGHT_REPLAY_ROOT'] == str(dataset_dir.parent)
    assert env['FLIGHT_REPLAY_DATASET_ID'] == 'may-v1'
    assert env['FLIGHT_RELEASE_MANIFEST_PATH'] == str(release / 'candidate.json')
    with pytest.raises(ValueError, match='dataset is not registered'):
        validate({**settings(tmp_path / 'other'), 'backend': 'composite',
                  'release_manifest_path': str(release / 'candidate.json'),
                  'replay_root': str(dataset_dir.parent), 'replay_dataset_id': 'missing'})
