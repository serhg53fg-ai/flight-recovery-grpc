from copy import deepcopy
from datetime import datetime, timedelta

import pytest


def scenario(**updates):
    return dict(airport='ZSPD', horizon_start='2025-05-02T23:00:00+08:00',
                horizon_end='2025-05-03T04:00:00+08:00', slot_minutes=15,
                departure_capacity=1, arrival_capacity=1, mtt_minutes=30,
                closures=[], **updates)


def flight(identity='a', tail='B1', origin='ZSPD', destination='ZGGG', departure='2025-05-02T23:05:00+08:00', arrival='2025-05-03T00:05:00+08:00'):
    return dict(flight_id=identity, tail=tail, origin=origin, destination=destination,
                planned_departure=departure, planned_arrival=arrival,
                predicted_departure=departure, predicted_arrival=arrival)


def solve(flights, config=None):
    from apps.flight.recovery.engine import schedule
    from apps.flight.recovery.validation import validate_plan
    config = config or scenario()
    plan = schedule(flights, config)
    assert validate_plan(flights, config, plan) == []
    return plan


def test_departure_capacity_and_no_early_departure_across_midnight():
    plan = solve([flight(), flight('b', 'B2')])
    assert plan['status'] == 'SUCCEEDED'
    assert plan['assignments'][0]['departure'] == '2025-05-02T23:05:00+08:00'
    assert plan['assignments'][1]['departure'] == '2025-05-02T23:15:00+08:00'
    assert plan['assignments'][1]['arrival'] == '2025-05-03T00:15:00+08:00'


def test_arrival_capacity_is_actually_enforced():
    plan = solve([flight(origin='ZGGG', destination='ZSPD'), flight('b', 'B2', origin='ZBAA', destination='ZSPD')])
    assert plan['assignments'][1]['arrival'] == '2025-05-03T00:15:00+08:00'


def test_non_target_rotation_propagates_mtt():
    flights = [flight(), flight('b', 'B1', 'ZGGG', 'ZBAA', '2025-05-03T00:10:00+08:00', '2025-05-03T01:10:00+08:00'),
               flight('c', 'B1', 'ZBAA', 'ZSPD', '2025-05-03T01:15:00+08:00', '2025-05-03T02:15:00+08:00')]
    plan = solve(flights)
    assert plan['assignments'][1]['departure'] == '2025-05-03T00:35:00+08:00'
    assert plan['assignments'][2]['departure'] == '2025-05-03T02:05:00+08:00'


def test_prediction_is_lower_bound():
    f = flight()
    f['predicted_departure'] = '2025-05-02T23:22:00+08:00'
    f['predicted_arrival'] = '2025-05-03T00:22:00+08:00'
    assert solve([f])['assignments'][0]['departure'] == f['predicted_departure']


@pytest.mark.parametrize('kind', ['departure', 'arrival'])
def test_closure_is_half_open_and_propagates_delay(kind):
    cfg = scenario()
    cfg['closures'] = [dict(kind=kind, start='2025-05-02T23:00:00+08:00', end='2025-05-03T00:30:00+08:00')]
    f = flight() if kind == 'departure' else flight(origin='ZGGG', destination='ZSPD')
    plan = solve([f], cfg)
    assert plan['assignments'][0][kind] == '2025-05-03T00:30:00+08:00'


def test_zero_capacity_and_blocked_successor_are_explicit():
    cfg = scenario()
    cfg['departure_capacity'] = 0
    plan = solve([flight(), flight('b', 'B1', 'ZGGG', 'ZSPD', '2025-05-03T00:45:00+08:00', '2025-05-03T01:45:00+08:00')], cfg)
    assert plan['status'] == 'INFEASIBLE'
    assert [item['reason'] for item in plan['unassigned']] == ['NO_FEASIBLE_SLOT', 'PREDECESSOR_UNSCHEDULED']


def test_discontinuous_rotation_is_not_silently_linked():
    plan = solve([flight(), flight('b', 'B1', 'ZBAA', 'ZSPD', '2025-05-03T01:00:00+08:00', '2025-05-03T02:00:00+08:00')])
    assert plan['status'] == 'PARTIAL'
    assert plan['unassigned'][0]['reason'] == 'ROTATION_DISCONTINUITY'


@pytest.mark.parametrize('change', ['naive', 'duplicate', 'duration', 'capacity', 'horizon'])
def test_invalid_input_rejected(change):
    from apps.flight.recovery.engine import schedule
    flights, cfg = [flight()], scenario()
    if change == 'naive': flights[0]['planned_departure'] = '2025-05-02T23:05:00'
    if change == 'duplicate': flights.append(flight())
    if change == 'duration': flights[0]['predicted_departure'] = '2025-05-03T03:00:00+08:00'
    if change == 'capacity': cfg['departure_capacity'] = True
    if change == 'horizon': cfg['horizon_end'] = '2025-05-05T04:00:00+08:00'
    with pytest.raises(ValueError): schedule(flights, cfg)


def test_deterministic_and_does_not_mutate_inputs():
    from apps.flight.recovery.engine import schedule
    flights, cfg = [flight(), flight('b', 'B2')], scenario()
    before = deepcopy((flights, cfg))
    assert schedule(flights, cfg) == schedule(flights, cfg)
    assert (flights, cfg) == before


def test_rotation_priority_can_reduce_total_delay_without_breaking_constraints():
    from apps.flight.recovery.engine import schedule
    from apps.flight.recovery.validation import validate_plan
    flights = [
        flight('independent', 'B2', departure='2025-05-02T23:01:00+08:00',
               arrival='2025-05-03T00:01:00+08:00'),
        flight('chain-first', 'B1', departure='2025-05-02T23:05:00+08:00',
               arrival='2025-05-03T00:05:00+08:00'),
        flight('chain-next', 'B1', origin='ZGGG', destination='ZSPD',
               departure='2025-05-03T00:35:00+08:00',
               arrival='2025-05-03T01:35:00+08:00'),
    ]
    config = scenario()
    earliest = schedule(flights, config)
    prioritized = schedule(flights, config, priority_policy='rotation_urgency')
    assert validate_plan(flights, config, prioritized) == []
    assert prioritized['algorithm_version'] != earliest['algorithm_version']
    assert sum(x['delay_minutes'] for x in prioritized['assignments']) < sum(
        x['delay_minutes'] for x in earliest['assignments'])
    assert next(x for x in prioritized['assignments'] if x['flight_id'] == 'chain-first')['departure'] == '2025-05-02T23:05:00+08:00'


def test_recovery_rejects_unknown_priority_policy():
    from apps.flight.recovery.engine import schedule
    with pytest.raises(ValueError, match='priority'):
        schedule([flight()], scenario(), priority_policy='unknown')


@pytest.mark.parametrize('tamper', ['capacity', 'early', 'missing', 'duration', 'status', 'reason', 'rotation', 'delay', 'missing_delay', 'boolean_delay', 'nan_delay'])
def test_independent_validator_rejects_tampered_plans(tamper):
    from apps.flight.recovery.validation import validate_plan
    flights = [flight(), flight('b', 'B2')]
    if tamper == 'rotation':
        flights[1] = flight('b', 'B1', 'ZGGG', 'ZSPD', '2025-05-03T00:10:00+08:00', '2025-05-03T01:10:00+08:00')
    cfg = scenario()
    plan = solve(flights, cfg)
    if tamper == 'capacity': plan['assignments'][1].update(departure=plan['assignments'][0]['departure'], arrival=plan['assignments'][0]['arrival'])
    if tamper == 'early': plan['assignments'][0]['departure'] = '2025-05-02T23:00:00+08:00'
    if tamper == 'missing': plan['assignments'].pop()
    if tamper == 'duration': plan['assignments'][0]['arrival'] = '2025-05-03T00:10:00+08:00'
    if tamper == 'status': plan['status'] = 'INFEASIBLE'
    if tamper == 'reason':
        lost = plan['assignments'].pop()
        plan['unassigned'].append(dict(flight_id=lost['flight_id'], reason='NO_FEASIBLE_SLOT'))
        plan['status'] = 'PARTIAL'
    if tamper == 'rotation': plan['assignments'][1].update(departure=flights[1]['planned_departure'], arrival=flights[1]['planned_arrival'])
    if tamper == 'delay': plan['assignments'][0]['delay_minutes'] = -999
    if tamper == 'missing_delay': plan['assignments'][0].pop('delay_minutes')
    if tamper == 'boolean_delay': plan['assignments'][0]['delay_minutes'] = False
    if tamper == 'nan_delay': plan['assignments'][0]['delay_minutes'] = float('nan')
    assert validate_plan(flights, cfg, plan)
