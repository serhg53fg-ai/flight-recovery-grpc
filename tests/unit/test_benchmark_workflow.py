"""Bounded benchmark records real task outcomes and never overwrites evidence."""
import json

import pytest


class FakeClient:
    def __init__(self, ready=True):
        self.ready = ready
        self.posts = 0

    def get(self, path):
        if path == '/ready':
            return (200 if self.ready else 503), {'ready': self.ready}
        if path == '/api/v1/prediction-capabilities':
            return 200, {'replay_dataset_id': 'may-v1', 'source': 'BASELINE',
                         'model_version': 'history+flow'}
        if path.startswith('/api/v1/prediction-jobs/'):
            return 200, {'status': 'SUCCEEDED', 'results': [{
                'source': 'BASELINE', 'model_version': 'history+flow',
            }]}
        raise AssertionError(path)

    def post(self, path, body, key):
        assert path == '/api/v1/prediction-jobs'
        assert body['replay_dataset_id'] == 'may-v1'
        assert key
        self.posts += 1
        return 202, {'job_id': f'job-{self.posts}',
                     'status_url': f'/api/v1/prediction-jobs/job-{self.posts}'}


def test_benchmark_records_model_batch_concurrency_and_latency(tmp_path):
    from scripts.benchmark_workflow import run_benchmark

    output = tmp_path / 'benchmark.json'
    report = run_benchmark(FakeClient(), 'may-v1', 2, 3, output)
    assert report['requested'] == report['succeeded'] == 3
    assert report['concurrency'] == 2 and report['batch_size'] == 1
    assert report['model_version'] == 'history+flow' and report['source'] == 'BASELINE'
    assert report['latency_seconds']['p95'] >= report['latency_seconds']['p50'] >= 0
    assert report['queue_wait_upper_seconds']['p95'] >= 0
    assert len(report['jobs']) == 3
    assert json.loads(output.read_text()) == report


def test_benchmark_rejects_unready_target_before_submitting(tmp_path):
    from scripts.benchmark_workflow import run_benchmark

    client = FakeClient(ready=False)
    with pytest.raises(ValueError, match='ready'):
        run_benchmark(client, 'may-v1', 1, 2, tmp_path / 'blocked.json')
    assert client.posts == 0


def test_benchmark_never_overwrites_previous_report(tmp_path):
    from scripts.benchmark_workflow import run_benchmark

    path = tmp_path / 'benchmark.json'
    path.write_text('previous')
    client = FakeClient()
    with pytest.raises(ValueError, match='exists'):
        run_benchmark(client, 'may-v1', 1, 1, path)
    assert path.read_text() == 'previous'
    assert client.posts == 0


def test_client_exception_keeps_request_index_and_error(tmp_path):
    from scripts.benchmark_workflow import run_benchmark

    class BrokenClient(FakeClient):
        def post(self, path, body, key):
            raise TimeoutError('request timed out')

    report = run_benchmark(BrokenClient(), 'may-v1', 1, 1, tmp_path / 'error.json')
    assert report['errors'] == 1
    assert report['jobs'][0]['index'] == 0
    assert report['jobs'][0]['status'] == 'CLIENT_ERROR'
    assert report['jobs'][0]['error_code'] == 'TimeoutError'
