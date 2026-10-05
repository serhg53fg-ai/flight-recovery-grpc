import json
from pathlib import Path

import pytest

from tests.training.test_zggg_release import release_input
from scripts.zggg_release import sha256_path


def experimental_input(tmp_path):
    value = release_input(tmp_path)
    report = tmp_path / 'accuracy.json'
    report.write_text(json.dumps({'passed': False, 'failures': ['flight_regression']}))
    value['artifacts']['accuracy_report'] = {'path': str(report), 'sha256': sha256_path(report)}
    value.update(flight_mode='qwen', flight_model_version='qwen+adapter-iem-v4',
                 flow_model_version='gradient_boosting', prompt_version='zggg-weather-duration-prompt-v2',
                 feature_contract_version='zggg-airport-flow-v1',
                 flight_gate={'passed': False}, flow_gate={'passed': True},
                 fallback_identity={'source': 'BASELINE', 'flight_model_version': 'historical-flight-v1',
                                    'prompt_version': 'historical-duration-v1'})
    return value


def test_experimental_release_preserves_failed_accuracy_and_requires_opt_in(tmp_path):
    from scripts.zggg_release import stage_candidate, stage_experimental_candidate
    from deploy.distributed.release import load_release, release_identity
    from deploy.distributed.resident import validate
    from tests.unit.test_resident import settings
    value = experimental_input(tmp_path)
    with pytest.raises(ValueError, match='gate'):
        stage_candidate(value, tmp_path / 'ordinary')
    output = tmp_path / 'experimental'
    result = stage_experimental_candidate(value, output)
    assert result['flight_gate_passed'] is False
    assert result['deployment_stage'] == 'experimental'
    release = load_release(output / 'candidate.json')
    assert release_identity(release)['fallback_identity'] == value['fallback_identity']
    config = {**settings(tmp_path / 'runtime'), 'backend': 'composite',
              'release_manifest_path': str(output / 'candidate.json')}
    with pytest.raises(ValueError, match='experimental_model'):
        validate(config)
    assert validate({**config, 'experimental_model': True})['flight_mode'] == 'qwen'


@pytest.mark.parametrize('missing', ['adapter', 'accuracy_report'])
def test_experimental_requires_trained_adapter_and_accuracy_evidence(tmp_path, missing):
    from scripts.zggg_release import stage_experimental_candidate
    value = experimental_input(tmp_path)
    del value['artifacts'][missing]
    with pytest.raises(ValueError):
        stage_experimental_candidate(value, tmp_path / 'release')
    assert not (tmp_path / 'release').exists()


def test_experimental_cannot_claim_accuracy_passed_or_use_historical_primary(tmp_path):
    from scripts.zggg_release import stage_experimental_candidate
    value = experimental_input(tmp_path)
    value['flight_gate'] = {'passed': True}
    with pytest.raises(ValueError, match='accuracy'):
        stage_experimental_candidate(value, tmp_path / 'release')
    value['flight_gate'] = {'passed': False}
    value['flight_mode'] = 'historical'
    with pytest.raises(ValueError):
        stage_experimental_candidate(value, tmp_path / 'release')


def test_experimental_worker_probe_accepts_only_declared_fallback(monkeypatch):
    from deploy.distributed import worker_pool
    from flight.v1 import prediction_pb2 as pb
    class Channel:
        def close(self): pass
    class Stub:
        def __init__(self, channel): pass
        def Predict(self, request, timeout):
            result = pb.PredictResponse(worker_id='worker-a', source=pb.BASELINE,
                model_version='historical-flight-v1+gradient_boosting', flight_degraded=True,
                flight_fallback_reason='PRIMARY_TIMEOUT', prompt_version='historical-duration-v1')
            result.airport_flow.model_version='gradient_boosting'
            return result
    monkeypatch.setattr(worker_pool.grpc, 'insecure_channel', lambda *args, **kwargs: Channel())
    monkeypatch.setattr(worker_pool.rpc, 'InferenceWorkerStub', Stub)
    expected={'source':'LLM','model_version':'qwen3+adapter-iem-v4+gradient_boosting',
              'flow_model_version':'gradient_boosting'}
    fallback={'source':'BASELINE','flight_model_version':'historical-flight-v1',
              'prompt_version':'historical-duration-v1'}
    result=worker_pool.probe_worker_identity('127.0.0.1:50052','worker-a',expected,
                                              timeout=20,fallback_identity=fallback)
    assert result['flight_degraded'] is True
    with pytest.raises(ValueError):
        worker_pool.probe_worker_identity('127.0.0.1:50052','worker-a',expected,timeout=20)


def test_experimental_resident_allocates_gpu_inference_budget(tmp_path):
    from scripts.zggg_release import stage_experimental_candidate
    from deploy.distributed.resident import commands, environment, validate
    from tests.unit.test_resident import settings
    release = tmp_path / 'release'
    stage_experimental_candidate(experimental_input(tmp_path), release)
    config = validate({**settings(tmp_path / 'runtime'), 'backend': 'composite',
                       'release_manifest_path': str(release / 'candidate.json'),
                       'experimental_model': True})
    env = environment(config)
    assert env['FLIGHT_EXPERIMENTAL_MODEL'] == '1'
    assert env['FLIGHT_RPC_TIMEOUT_SECONDS'] == '60'
    worker = commands(config)['worker']
    assert worker[worker.index('--max-seconds') + 1] == '45'


def test_resident_control_can_inspect_preverified_release_without_rehashing_all_weights(tmp_path):
    from scripts.zggg_release import stage_experimental_candidate
    from deploy.distributed.release import load_release
    from deploy.distributed.resident import validate
    from tests.unit.test_resident import settings
    value = experimental_input(tmp_path)
    release_dir = tmp_path / 'release'
    stage_experimental_candidate(value, release_dir)
    config = {**settings(tmp_path / 'runtime'), 'backend': 'composite',
              'release_manifest_path': str(release_dir / 'candidate.json'),
              'experimental_model': True}
    (tmp_path / 'model.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        load_release(release_dir / 'candidate.json')
    with pytest.raises(ValueError, match='hash'):
        validate(config)
    assert load_release(release_dir / 'candidate.json', verify_artifacts=False)['flight_mode'] == 'qwen'
    assert validate(config, verify_artifacts=False)['flight_mode'] == 'qwen'


def test_resident_role_hashes_large_artifacts_only_for_model_worker(tmp_path, monkeypatch):
    import sys
    from deploy.distributed import resident_role
    from tests.unit.test_resident import settings
    path = tmp_path / 'config.json'
    path.write_text(__import__('json').dumps(settings(tmp_path / 'runtime')))
    called = []
    def fake_validate(config, *, verify_artifacts=True):
        called.append(verify_artifacts)
        return config
    monkeypatch.setattr(resident_role, 'validate', fake_validate)
    monkeypatch.setattr(resident_role, 'roles', lambda c: ('worker', 'web-a'))
    monkeypatch.setattr(resident_role, 'wait_dependencies', lambda *args: None)
    monkeypatch.setattr(resident_role, 'environment', lambda c: {})
    monkeypatch.setattr(resident_role, 'commands', lambda c: {'worker': ['/bin/true'], 'web-a': ['/bin/true']})
    monkeypatch.setattr(resident_role.os, 'chdir', lambda *args: None)
    def stop_exec(*args): raise RuntimeError('exec reached')
    monkeypatch.setattr(resident_role.os, 'execve', stop_exec)
    for role in ('web-a', 'worker'):
        monkeypatch.setattr(sys, 'argv', ['resident_role', '--config', str(path), '--role', role])
        with pytest.raises(RuntimeError, match='exec reached'):
            resident_role.main()
    assert called == [False, True]
