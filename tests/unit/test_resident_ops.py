import json

import pytest


class FakeClient:
    def __init__(self, *, ready=True, prediction='SUCCEEDED', recovery='SUCCEEDED'):
        self.ready, self.prediction, self.recovery = ready, prediction, recovery
        self.posts = []

    def get(self, path):
        if path == '/ready':
            return (200 if self.ready else 503), {'ready': self.ready}
        if path == '/api/v1/prediction-jobs/job-a':
            return 200, {'job_id': 'job-a', 'status': self.prediction, 'results': [{
                'success': self.prediction == 'SUCCEEDED', 'source': 'LLM',
                'model_version': 'qwen+adapter+flow', 'worker_id': 'worker-a',
                'flight_degraded': False, 'flow_degraded': False,
            }]}
        raise AssertionError(path)

    def post(self, path, body, key):
        self.posts.append((path, body, key))
        if path == '/api/v1/prediction-jobs':
            return 202, {'job_id': 'job-a', 'status_url': '/api/v1/prediction-jobs/job-a'}
        if path == '/api/v1/recovery-jobs':
            return 201, {'recovery_id': 'recovery-a', 'status': self.recovery,
                         'validation_errors': [],
                         'situation_snapshot': {'risk_level': 'NORMAL', 'snapshot_digest': 'a' * 64},
                         'summary': {'scheduled_count': 1}}
        raise AssertionError(path)


def test_demo_records_prediction_situation_and_recovery(tmp_path):
    from scripts.resident_ops import run_demo

    output = tmp_path / 'demo.json'
    result = run_demo(FakeClient(), output, poll_seconds=.01, timeout_seconds=1)

    assert result['status'] == 'PASS'
    assert result['prediction']['source'] == 'LLM'
    assert result['prediction']['flight_degraded'] is False
    assert result['situation']['risk_level'] == 'NORMAL'
    assert result['recovery']['status'] == 'SUCCEEDED'
    assert json.loads(output.read_text()) == result


def test_demo_rejects_unready_stack_without_submission(tmp_path):
    from scripts.resident_ops import run_demo

    client = FakeClient(ready=False)
    with pytest.raises(ValueError, match='ready'):
        run_demo(client, tmp_path / 'demo.json')
    assert client.posts == []


def test_demo_does_not_create_recovery_for_failed_prediction(tmp_path):
    from scripts.resident_ops import run_demo

    client = FakeClient(prediction='FAILED')
    with pytest.raises(RuntimeError, match='prediction'):
        run_demo(client, tmp_path / 'demo.json', poll_seconds=.01, timeout_seconds=1)
    assert [row[0] for row in client.posts] == ['/api/v1/prediction-jobs']


def test_up_prepares_new_runtime_then_starts(tmp_path):
    from scripts.resident_ops import run_stack

    runtime = tmp_path / 'runtime'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'runtime': str(runtime)}))
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return type('Result', (), {'returncode': 0})()

    assert run_stack('up', config, runner=runner) == 0
    assert [command[3] for command in calls] == ['preflight', 'prepare', 'start']


def test_up_checks_health_when_owned_stack_is_already_active(tmp_path):
    from scripts.resident_ops import run_stack

    runtime = tmp_path / 'runtime'; runtime.mkdir()
    (runtime / 'initialized.json').write_text('{}')
    (runtime / 'supervisor.sock').write_text('owned marker')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'runtime': str(runtime)}))
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return type('Result', (), {'returncode': 0})()

    assert run_stack('up', config, runner=runner) == 0
    assert [command[3] for command in calls] == ['health']
