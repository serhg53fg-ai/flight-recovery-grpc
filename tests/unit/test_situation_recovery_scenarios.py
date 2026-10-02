from copy import deepcopy

import pytest

from tests.unit.test_recovery_engine import scenario
from tests.unit.test_recovery_service import prediction_job


def _job_with_two_flights():
    job = prediction_job()
    second = deepcopy(job['results'][0])
    second.update(flight_id='flight-b', trace_id='trace-b')
    second['flight_info']['机尾号'] = 'B2'
    job['results'].append(second)
    return job


@pytest.mark.parametrize('case', ['normal', 'peak', 'adverse_weather', 'llm_fallback', 'worker_failure'])
def test_prediction_situation_recovery_scenario_matrix(case):
    from apps.flight.recovery.service import build_recovery

    job, cfg = _job_with_two_flights(), {**scenario(), 'departure_capacity': 2}
    if case == 'peak':
        job['results'][0]['airport_flow'] = {
            'window_semantics': 'cumulative',
            'windows': [{'horizon_minutes': 15, 'takeoff': 4, 'landing': 0, 'total': 4}],
        }
    elif case == 'adverse_weather':
        job['results'][0]['normalized_context'] = {
            'departure_weather': {'metar': {'features': {
                'thunderstorm': True, 'visibility_m': 1200, 'wind_speed_kt': 25,
            }}},
        }
    elif case == 'llm_fallback':
        job['results'][0].update(source='BASELINE', flight_degraded=True,
                                 flight_fallback_reason='PRIMARY_TIMEOUT')
    elif case == 'worker_failure':
        job['status'] = 'PARTIAL'
        job['results'][1] = {'flight_id': 'flight-b', 'success': False,
                             'error_code': 'UNAVAILABLE'}

    plan = build_recovery(job, cfg, f'{case}-v1')
    snapshot = plan['situation_snapshot']

    assert plan['validation_errors'] == []
    assert plan['summary']['objective_vector']['validation_violations'] == 0
    if case == 'normal':
        assert snapshot['risk_level'] == 'NORMAL'
    elif case == 'peak':
        assert any(item['type'] == 'DEPARTURE_CAPACITY' for item in snapshot['conflicts'])
    elif case == 'adverse_weather':
        assert snapshot['prediction']['adverse_weather'] == 1
        assert snapshot['risk_level'] == 'HIGH'
    elif case == 'llm_fallback':
        assert snapshot['prediction']['flight_fallbacks'] == 1
    elif case == 'worker_failure':
        assert snapshot['prediction']['failed'] == 1
        assert plan['summary']['excluded_prediction_count'] == 1
