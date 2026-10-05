import json


METRICS = '''
flight_gateway_serving 1
flight_gateway_worker_healthy{worker="worker-a"} 1
flight_gateway_worker_circuit_state{worker="worker-a"} 1
flight_gateway_worker_inflight{worker="worker-a"} 0
flight_gateway_worker_capacity{worker="worker-a"} 1
flight_gateway_worker_failure_total{worker="worker-a",code="UNAVAILABLE"} 2
flight_stage_duration_seconds_bucket{stage="queue",outcome="success",le="10"} 9
flight_stage_duration_seconds_count{stage="queue",outcome="success"} 10
'''


def test_monitor_passes_healthy_snapshot_and_records_counters(tmp_path):
    from scripts.resident_monitor import evaluate_monitoring

    report, state = evaluate_monitoring(200, {'ready': True}, METRICS)

    assert report['status'] == 'PASS'
    assert report['workers'] == {'worker-a': {'healthy': True, 'circuit': 'CLOSED',
                                               'inflight': 0, 'capacity': 1}}
    assert report['delta_observed'] is False
    assert state['worker_failures']['worker-a:UNAVAILABLE'] == 2


def test_monitor_detects_new_failures_and_queue_over_ten_seconds():
    from scripts.resident_monitor import evaluate_monitoring

    previous = {'worker_failures': {'worker-a:UNAVAILABLE': 1},
                'queue_success_count': 7, 'queue_success_le_10': 7}
    report, _ = evaluate_monitoring(200, {'ready': True}, METRICS, previous)

    assert report['status'] == 'WARN'
    assert report['delta_observed'] is True
    assert report['new_worker_failures'] == {'worker-a:UNAVAILABLE': 1}
    assert report['queue_over_10_seconds'] == 1


def test_monitor_fails_closed_for_unready_or_bad_worker():
    from scripts.resident_monitor import evaluate_monitoring

    bad = METRICS.replace('flight_gateway_worker_healthy{worker="worker-a"} 1',
                          'flight_gateway_worker_healthy{worker="worker-a"} 0')
    report, _ = evaluate_monitoring(503, {'ready': False}, bad)

    assert report['status'] == 'FAIL'
    assert 'deployment_not_ready' in report['reasons']
    assert 'worker_unhealthy:worker-a' in report['reasons']


def test_monitor_writes_new_report_without_overwriting(tmp_path):
    import pytest
    from scripts.resident_monitor import write_report

    output = tmp_path / 'monitor.json'
    write_report(output, {'status': 'PASS'})
    assert json.loads(output.read_text())['status'] == 'PASS'
    with pytest.raises(ValueError, match='already exists'):
        write_report(output, {'status': 'PASS'})
