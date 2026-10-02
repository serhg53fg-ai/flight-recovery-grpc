from copy import deepcopy

import pytest

from tests.unit.test_recovery_engine import scenario


def prediction_job():
    return dict(job_id='bdeee974-cb29-4d9c-a145-c20500075a47', status='SUCCEEDED', results=[dict(
        flight_id='flight-a', success=True, source='TEST', model_version='test-v1', worker_id='worker-a',
        trace_id='trace-a', flight_info={'机尾号': 'B1', '计划起飞站四字码': 'ZSPD', '计划到达站四字码': 'ZGGG',
        '计划离港时间': '2025-05-02T23:05:00+08:00', '计划到港时间': '2025-05-03T00:05:00+08:00'},
        prediction_data={'实际离港时间': '2025-05-02T23:15:00+08:00', '实际起飞时间': '2025-05-02T23:20:00+08:00',
                         '实际落地时间': '2025-05-03T00:10:00+08:00', '实际到港时间': '2025-05-03T00:15:00+08:00'})])


def test_versioned_plan_retains_prediction_provenance_and_stable_digest():
    from apps.flight.recovery.service import build_recovery
    job = prediction_job()
    before = deepcopy(job)
    plan = build_recovery(job, scenario(), 'weather-v1')
    assert plan['status'] == 'SUCCEEDED' and plan['validation_errors'] == []
    assert plan['source_job_id'] == job['job_id'] and plan['scenario_version'] == 'weather-v1'
    assert plan['provenance'][0]['source'] == 'TEST'
    assert plan['provenance'][0]['trace_id'] == 'trace-a'
    assert plan['provenance'][0]['model_version'] == 'test-v1'
    assert len(plan['input_digest']) == 64
    assert build_recovery(job, scenario(), 'weather-v1')['input_digest'] == plan['input_digest']
    assert build_recovery(job, {**scenario(), 'arrival_capacity': 2}, 'weather-v1')['input_digest'] != plan['input_digest']
    assert job == before


def test_failed_predictions_are_explicitly_excluded():
    from apps.flight.recovery.service import build_recovery
    job = prediction_job()
    job['status'] = 'PARTIAL'
    job['results'].append(dict(flight_id='bad', success=False, error_code='INVALID_ARGUMENT'))
    plan = build_recovery(job, scenario(), 'weather-v1')
    assert len(plan['assignments']) == 1
    assert plan['excluded'] == [dict(flight_id='bad', reason='INVALID_ARGUMENT')]


@pytest.mark.parametrize('status', ['QUEUED', 'RUNNING'])
def test_nonterminal_prediction_is_rejected(status):
    from apps.flight.recovery.service import build_recovery
    job = prediction_job()
    job['status'] = status
    with pytest.raises(ValueError, match='终态'): build_recovery(job, scenario(), 'v1')


@pytest.mark.parametrize('bad', ['source', 'model', 'chronology', 'all_failed', 'version'])
def test_unusable_prediction_or_version_is_rejected(bad):
    from apps.flight.recovery.service import build_recovery
    job, version = prediction_job(), 'v1'
    r = job['results'][0]
    if bad == 'source': r['source'] = 'UNKNOWN'
    if bad == 'model': r['model_version'] = ''
    if bad == 'chronology': r['prediction_data']['实际起飞时间'] = '2025-05-03T03:00:00+08:00'
    if bad == 'all_failed': r['success'] = False
    if bad == 'version': version = ''
    with pytest.raises(ValueError): build_recovery(job, scenario(), version)


def test_validator_failure_never_becomes_success(monkeypatch):
    from apps.flight.recovery import service
    monkeypatch.setattr(service, 'validate_plan', lambda *args: ['CAPACITY:arrival'])
    plan = service.build_recovery(prediction_job(), scenario(), 'v1')
    assert plan['status'] == 'INVALID' and plan['validation_errors']


def test_baseline_replay_job_recovery_keeps_snapshot_identity():
    from apps.flight.recovery.service import build_recovery

    job = prediction_job()
    job['results'][0]['source'] = 'BASELINE'
    job['submission_context'] = {
        'replay_dataset_id': 'example-dataset', 'dataset_manifest_hash': 'dataset-hash',
        'feature_contract_version': 'zggg-airport-flow-v1', 'snapshot_hash': 'snapshot-hash',
        'release_identity': {'release_version': 'release-v2', 'source': 'BASELINE',
                             'model_version': 'historical-flight-v1+gradient_boosting'}}
    plan = build_recovery(job, scenario(), 'weather-v1')
    assert plan['source_context']['snapshot_hash'] == 'snapshot-hash'
    assert plan['source_context']['release_identity']['release_version'] == 'release-v2'
    assert plan['summary']['scheduled_count'] == 1
    assert plan['summary']['unscheduled_count'] == 0
    assert plan['summary']['excluded_prediction_count'] == 0
    assert plan['summary']['validation_violation_count'] == 0
    assert plan['summary']['predicted_baseline_feasible'] is True


def test_partial_predictions_report_exclusions_and_no_capacity_inference():
    from apps.flight.recovery.service import build_recovery

    job = prediction_job()
    job['status'] = 'PARTIAL'
    job['results'][0]['airport_flow'] = {'windows': [{'horizon_minutes': 15, 'total': 999}]}
    job['results'].append({'flight_id': 'bad', 'success': False, 'error_code': 'UNAVAILABLE'})
    cfg = scenario()
    plan = build_recovery(job, cfg, 'v1')
    assert plan['summary']['excluded_prediction_count'] == 1
    assert plan['summary']['scheduled_count'] == 1
    assert plan['scenario']['departure_capacity'] == cfg['departure_capacity'] == 1
    assert plan['scenario']['arrival_capacity'] == cfg['arrival_capacity'] == 1


def test_infeasible_scenario_has_no_fake_improvement():
    from apps.flight.recovery.service import build_recovery

    cfg = scenario()
    cfg['departure_capacity'] = 0
    plan = build_recovery(prediction_job(), cfg, 'v1')
    assert plan['status'] == 'INFEASIBLE'
    assert plan['summary']['scheduled_count'] == 0
    assert plan['summary']['unscheduled_count'] == 1
    assert plan['summary']['added_delay_minutes'] == 0
    assert plan['summary']['improvement_claimed'] is False


def test_baseline_comparison_uses_same_normalized_times_as_scheduler():
    from apps.flight.recovery.service import build_recovery

    job = prediction_job()
    job['results'][0]['prediction_data']['实际离港时间'] = '2025-05-02T23:00:00+08:00'
    plan = build_recovery(job, scenario(), 'v1')
    assert plan['status'] == 'SUCCEEDED'
    assert plan['summary']['predicted_baseline_feasible'] is True
    assert plan['summary']['predicted_baseline_validation_errors'] == []


def test_recovery_summary_exposes_multi_objective_metrics_without_fake_score():
    from apps.flight.recovery.service import build_recovery

    job = prediction_job()
    second = deepcopy(job['results'][0])
    second.update(flight_id='flight-b', trace_id='trace-b')
    second['flight_info']['机尾号'] = 'B2'
    job['results'].append(second)
    plan = build_recovery(job, scenario(), 'v1')

    summary = plan['summary']
    assert summary['scheduled_count'] == 2
    assert summary['affected_flight_count'] == 1
    assert summary['completion_rate'] == 1.0
    assert summary['total_delay_minutes'] == 15.0
    assert summary['max_delay_minutes'] == 15.0
    assert summary['average_delay_minutes'] == 7.5
    assert summary['objective_vector'] == {
        'validation_violations': 0,
        'unscheduled_flights': 0,
        'total_delay_minutes': 15.0,
        'max_delay_minutes': 15.0,
        'affected_flights': 1,
    }
