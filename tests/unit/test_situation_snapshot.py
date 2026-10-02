from copy import deepcopy

from tests.unit.test_recovery_engine import scenario
from tests.unit.test_recovery_service import prediction_job


def test_snapshot_unifies_prediction_flow_weather_capacity_and_degradation():
    from apps.flight.situation.snapshot import build_situation_snapshot

    job = prediction_job()
    row = job['results'][0]
    row.update(
        flight_degraded=True,
        flight_fallback_reason='PRIMARY_TIMEOUT',
        flow_degraded=False,
        weather_stale=True,
        airport_flow={
            'window_semantics': 'cumulative',
            'model_version': 'gradient-boosting-flow-v1',
            'windows': [
                {'horizon_minutes': 15, 'takeoff': 2, 'landing': 1, 'total': 3},
                {'horizon_minutes': 30, 'takeoff': 3, 'landing': 2, 'total': 5},
            ],
        },
    )
    cfg = scenario()

    before = deepcopy((job, cfg))
    snapshot = build_situation_snapshot(job, cfg)

    assert snapshot['airport'] == 'ZSPD'
    assert snapshot['prediction']['successful'] == 1
    assert snapshot['prediction']['flight_fallbacks'] == 1
    assert snapshot['prediction']['stale_weather'] == 1
    assert snapshot['flow']['windows'][0]['capacity']['takeoff'] == 1
    assert snapshot['flow']['windows'][1]['capacity']['landing'] == 2
    assert snapshot['flow']['windows'][0]['pressure']['takeoff'] == 2.0
    assert {item['type'] for item in snapshot['conflicts']} == {'DEPARTURE_CAPACITY'}
    assert (job, cfg) == before


def test_snapshot_reports_partial_coverage_and_rotation_connection_risk():
    from apps.flight.situation.snapshot import build_situation_snapshot

    job = prediction_job()
    first = job['results'][0]
    first['flight_info']['计划到达站四字码'] = 'ZGGG'
    first['prediction_data']['实际到港时间'] = '2025-05-03T00:50:00+08:00'
    first['prediction_data']['实际落地时间'] = '2025-05-03T00:45:00+08:00'
    job['status'] = 'PARTIAL'
    job['results'].append({
        'flight_id': 'flight-b', 'success': True, 'source': 'LLM', 'model_version': 'qwen-v1',
        'worker_id': 'worker-b', 'trace_id': 'trace-b',
        'flight_info': {'机尾号': 'B1', '计划起飞站四字码': 'ZGGG', '计划到达站四字码': 'ZSPD',
                        '计划离港时间': '2025-05-03T01:00:00+08:00',
                        '计划到港时间': '2025-05-03T02:00:00+08:00'},
        'prediction_data': {'实际离港时间': '2025-05-03T01:00:00+08:00',
                            '实际起飞时间': '2025-05-03T01:05:00+08:00',
                            '实际落地时间': '2025-05-03T01:55:00+08:00',
                            '实际到港时间': '2025-05-03T02:00:00+08:00'},
    })
    job['results'].append({'flight_id': 'failed', 'success': False, 'error_code': 'UNAVAILABLE'})

    cfg = scenario()
    cfg.update(airport='ZGGG', mtt_minutes=30)
    snapshot = build_situation_snapshot(job, cfg)

    assert snapshot['prediction']['requested'] == 3
    assert snapshot['prediction']['successful'] == 2
    assert snapshot['prediction']['failed'] == 1
    risks = [item for item in snapshot['conflicts'] if item['type'] == 'ROTATION_CONNECTION']
    assert risks == [{'type': 'ROTATION_CONNECTION', 'severity': 'HIGH', 'tail_number': 'B1',
                      'predecessor_flight_id': 'flight-a', 'successor_flight_id': 'flight-b',
                      'shortfall_minutes': 20.0}]


def test_recovery_plan_embeds_snapshot_bound_to_same_inputs():
    from apps.flight.recovery.service import build_recovery

    plan = build_recovery(prediction_job(), scenario(), 'weather-v1')

    assert plan['situation_snapshot']['source_job_id'] == plan['source_job_id']
    assert plan['situation_snapshot']['airport'] == plan['scenario']['airport']
    assert plan['situation_snapshot']['prediction']['successful'] == len(plan['inputs'])
    assert len(plan['situation_snapshot']['snapshot_digest']) == 64


def test_snapshot_detects_departure_and_arrival_closure_events():
    from apps.flight.situation.snapshot import build_situation_snapshot

    job = prediction_job()
    arrival = deepcopy(job['results'][0])
    arrival.update(flight_id='flight-b', trace_id='trace-b')
    arrival['flight_info'].update(机尾号='B2', 计划起飞站四字码='ZGGG', 计划到达站四字码='ZSPD')
    job['results'].append(arrival)
    cfg = scenario()
    cfg['closures'] = [
        {'kind': 'departure', 'start': '2025-05-02T23:00:00+08:00',
         'end': '2025-05-02T23:30:00+08:00'},
        {'kind': 'arrival', 'start': '2025-05-03T00:00:00+08:00',
         'end': '2025-05-03T00:30:00+08:00'},
    ]

    snapshot = build_situation_snapshot(job, cfg)

    closures = [item for item in snapshot['conflicts'] if item['type'] == 'AIRPORT_CLOSURE']
    assert closures == [
        {'type': 'AIRPORT_CLOSURE', 'severity': 'HIGH', 'kind': 'departure',
         'flight_id': 'flight-a', 'event_time': '2025-05-02T23:15:00+08:00'},
        {'type': 'AIRPORT_CLOSURE', 'severity': 'HIGH', 'kind': 'arrival',
         'flight_id': 'flight-b', 'event_time': '2025-05-03T00:15:00+08:00'},
    ]


def test_slot_capacity_is_airport_scoped_and_uses_half_open_slots():
    from apps.flight.situation.snapshot import build_situation_snapshot
    job = prediction_job()
    other = deepcopy(job['results'][0])
    other['flight_id'] = 'flight-b'
    other['flight_info']['机尾号'] = 'B2'
    other['prediction_data']['实际离港时间'] = '2025-05-02T23:20:00+08:00'
    job['results'].append(other)
    def slots():
        return [c for c in build_situation_snapshot(job, scenario())['conflicts']
                if c['type'] == 'DEPARTURE_SLOT_CAPACITY']
    assert slots()[0]['flight_ids'] == ['flight-a', 'flight-b']
    assert slots()[0]['slot_start'] == '2025-05-02T23:15:00+08:00'
    other['prediction_data']['实际离港时间'] = '2025-05-02T23:30:00+08:00'
    assert slots() == []
    other['prediction_data']['实际离港时间'] = '2025-05-02T23:20:00+08:00'
    other['flight_info']['计划起飞站四字码'] = 'ZBAA'
    assert slots() == []
    zero = build_situation_snapshot(job, {**scenario(), 'departure_capacity': 0})
    assert any(c['type'] == 'DEPARTURE_SLOT_CAPACITY' and c['excess'] == 1
               for c in zero['conflicts'])


def test_slot_capacity_uses_recovery_no_early_event_lower_bound():
    from apps.flight.situation.snapshot import build_situation_snapshot
    job = prediction_job()
    row = job['results'][0]
    row['flight_info']['计划离港时间'] = '2025-05-02T23:30:00+08:00'
    other = deepcopy(row)
    other['flight_id'] = 'flight-b'
    other['flight_info']['机尾号'] = 'B2'
    other['prediction_data']['实际离港时间'] = '2025-05-02T23:35:00+08:00'
    job['results'].append(other)
    conflicts = build_situation_snapshot(job, scenario())['conflicts']
    slots = [c for c in conflicts if c['type'] == 'DEPARTURE_SLOT_CAPACITY']
    assert slots[0]['slot_start'] == '2025-05-02T23:30:00+08:00'
    assert slots[0]['count'] == 2
