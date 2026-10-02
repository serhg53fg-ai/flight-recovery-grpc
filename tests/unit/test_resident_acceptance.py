import json

from tests.unit.test_resident_ops import FakeClient


METRICS = '''
flight_gateway_serving 1
flight_gateway_worker_healthy{worker="worker-a"} 1
flight_gateway_worker_circuit_state{worker="worker-a"} 1
flight_gateway_worker_inflight{worker="worker-a"} 0
flight_gateway_worker_capacity{worker="worker-a"} 1
flight_stage_duration_seconds_bucket{stage="queue",outcome="success",le="10"} 4
flight_stage_duration_seconds_count{stage="queue",outcome="success"} 4
'''


def test_acceptance_combines_monitor_demo_and_delta_report(tmp_path):
    from scripts.resident_acceptance import run_acceptance

    target = tmp_path / 'acceptance'
    report = run_acceptance(FakeClient(), lambda: METRICS, target,
                            poll_seconds=.01, timeout_seconds=1)

    assert report['status'] == 'PASS'
    assert report['before']['status'] == 'PASS'
    assert report['demo']['prediction']['source'] == 'LLM'
    assert report['after']['status'] == 'PASS'
    assert json.loads((target / 'summary.json').read_text()) == report
    assert (target / 'demo.json').is_file()


def test_acceptance_rejects_existing_output_directory(tmp_path):
    import pytest
    from scripts.resident_acceptance import run_acceptance

    target = tmp_path / 'existing'; target.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        run_acceptance(FakeClient(), lambda: METRICS, target)


def test_acceptance_propagates_incremental_warning(tmp_path):
    from scripts.resident_acceptance import run_acceptance

    snapshots = iter((METRICS, METRICS.replace(
        'flight_stage_duration_seconds_count{stage="queue",outcome="success"} 4',
        'flight_stage_duration_seconds_count{stage="queue",outcome="success"} 5')))
    report = run_acceptance(FakeClient(), lambda: next(snapshots), tmp_path / 'warning',
                            poll_seconds=.01, timeout_seconds=1)

    assert report['status'] == 'WARN'
    assert report['after']['queue_over_10_seconds'] == 1
