from copy import deepcopy

import pytest
from flight.v1 import prediction_pb2 as pb

from apps.flight.domain.prediction import (
    flow_prediction_to_dict, normalize_flight, normalize_zggg_context, prediction_to_dict,
    validate_request, validate_response,
)


FLIGHT = {
    '航班号': 'cz9933', '机尾号': 'B20E8', '机型': 'A320',
    '计划起飞站四字码': 'zggg', '计划到达站四字码': 'ZSPD',
    '计划离港时间': '2025-05-02 09:30:00',
    '计划到港时间': '2025-05-02 11:45:00',
    '起飞站METAR': 'ZGGG 020900Z 08005MPS 9999',
}


def test_normalize_aliases_to_utc_without_mutating_input():
    before = deepcopy(FLIGHT)
    f = normalize_flight(FLIGHT)
    assert f.flight_number == 'CZ9933'
    assert f.departure_airport == 'ZGGG'
    assert f.planned_off_block.ToJsonString() == '2025-05-02T01:30:00Z'
    assert f.planned_on_block.ToJsonString() == '2025-05-02T03:45:00Z'
    assert f.flight_id
    assert not f.HasField('planned_total_flow')
    assert FLIGHT == before


@pytest.mark.parametrize('change', [
    {'计划离港时间': '09:30'}, {'计划到港时间': '2025-05-02 08:00:00'},
    {'机尾号': ''}, {'计划起飞站四字码': 'PEK'},
    {'航班号': '<img src=x onerror=alert(1)>'},
    {'计划总流量': -1}, {'计划总流量': True}, {'计划总流量': 1.5},
    {'计划地面航程_Mile': float('nan')},
])
def test_reject_incomplete_or_invalid_flight(change):
    with pytest.raises(ValueError):
        normalize_flight({**FLIGHT, **change})


def test_explicit_offset_wins_and_optional_zero_is_present():
    f = normalize_flight({**FLIGHT, '计划离港时间': '2025-05-02T01:30:00Z',
                          '计划总流量': 0}, 'Asia/Shanghai')
    assert f.planned_off_block.ToJsonString() == '2025-05-02T01:30:00Z'
    assert f.HasField('planned_total_flow') and f.planned_total_flow == 0


def test_unknown_timezone_rejected():
    with pytest.raises(ValueError):
        normalize_flight(FLIGHT, 'not/a/timezone')


def test_proto_boundary_rejects_missing_and_invalid_timestamp():
    with pytest.raises(ValueError):
        validate_request(pb.PredictRequest())
    req = pb.PredictRequest(trace_id='trace', flight=normalize_flight(FLIGHT))
    req.flight.planned_off_block.nanos = -1
    with pytest.raises(ValueError):
        validate_request(req)


def response():
    r = pb.PredictResponse(trace_id='trace', source=pb.TEST,
                           model_version='test-v1', worker_id='test-node')
    for key, t in [('off_block','01:40'), ('takeoff','01:50'),
                   ('landing','03:45'), ('on_block','03:55')]:
        getattr(r.prediction, key).FromJsonString(f'2025-05-02T{t}:00Z')
    return r


def test_result_validation_and_display_timezone():
    r = response()
    validate_response(r, 'trace')
    assert prediction_to_dict(r.prediction)['实际离港时间'] == '2025-05-02T09:40:00+08:00'


@pytest.mark.parametrize('fault', ['missing', 'order', 'trace', 'source', 'metadata'])
def test_invalid_backend_response_never_becomes_success(fault):
    r = response()
    if fault == 'missing': r.prediction.ClearField('landing')
    if fault == 'order': r.prediction.landing.CopyFrom(r.prediction.off_block)
    if fault == 'trace': r.trace_id = 'different'
    if fault == 'source': r.source = pb.SOURCE_UNSPECIFIED
    if fault == 'metadata': r.model_version = ''
    with pytest.raises(ValueError):
        validate_response(r, 'trace')

@pytest.mark.parametrize('alias', ['性质', 'flight_nature', 'nature'])
def test_flight_nature_survives_wire_and_enters_shared_prompt(alias):
    from google.protobuf.json_format import MessageToDict
    from training.prompt import render_duration_prompt
    flight = normalize_flight({**FLIGHT, alias: ' j '})
    recovered = pb.FlightFeatures.FromString(flight.SerializeToString())
    assert recovered.flight_nature == 'J'
    assert '"flight_nature":"J"' in render_duration_prompt(
        MessageToDict(recovered, preserving_proto_field_name=True))

@pytest.mark.parametrize('nature', [True, 123, [], {}, 'X' * 33])
def test_reject_invalid_flight_nature(nature):
    with pytest.raises(ValueError):
        normalize_flight({**FLIGHT, '性质': nature})


def test_additive_zggg_contract_keeps_legacy_wire_compatible():
    legacy = pb.PredictRequest(trace_id='legacy', flight=normalize_flight(FLIGHT))
    recovered = pb.PredictRequest.FromString(legacy.SerializeToString())
    assert recovered == legacy and not recovered.HasField('zggg_context')
    legacy_response = response()
    assert pb.PredictResponse.FromString(legacy_response.SerializeToString()) == legacy_response
    validate_response(legacy_response, 'trace')


def zggg_request():
    request = pb.PredictRequest(trace_id='zggg', flight=normalize_flight(FLIGHT))
    request.zggg_context.prediction_cutoff.FromJsonString('2025-05-02T01:30:00Z')
    request.zggg_context.data_version = 'zggg-v1'
    weather = request.zggg_context.departure_weather.metar
    weather.airport = 'ZGGG'; weather.kind = 'METAR'
    weather.issue_time.FromJsonString('2025-05-02T01:20:00Z')
    weather.features.visibility_m = 4000
    window = request.zggg_context.flow_features.add(horizon_minutes=15)
    window.planned_takeoff = 2; window.planned_landing = 3
    window.completed_takeoff = 1; window.completed_landing = 1
    return request


def test_zggg_context_requires_cutoff_and_rejects_future_weather():
    request = zggg_request()
    validate_request(request)
    request.zggg_context.ClearField('prediction_cutoff')
    with pytest.raises(ValueError, match='cutoff'):
        validate_request(request)
    request = zggg_request()
    request.zggg_context.departure_weather.metar.issue_time.FromJsonString(
        '2025-05-02T01:31:00Z')
    with pytest.raises(ValueError, match='天气'):
        validate_request(request)


def test_flow_prediction_is_non_negative_and_legacy_response_remains_valid():
    value = response()
    window = value.airport_flow.windows.add(horizon_minutes=15, takeoff=2,
                                            landing=3, total=5)
    value.data_version = 'zggg-v1'
    validate_response(value, 'trace')
    window.takeoff = -1
    with pytest.raises(ValueError, match='流量'):
        validate_response(value, 'trace')


def test_flow_windows_are_labeled_cumulative_from_prediction_cutoff():
    value = response()
    value.airport_flow.windows.add(horizon_minutes=15, takeoff=3, landing=2, total=5)
    value.airport_flow.windows.add(horizon_minutes=30, takeoff=2, landing=2, total=4)
    flow = flow_prediction_to_dict(value, '2025-05-02T09:30:00+08:00')
    assert flow['window_semantics'] == 'cumulative'
    assert flow['windows'][0]['window_start'] == '2025-05-02T09:30:00+08:00'
    assert flow['windows'][1]['window_start'] == '2025-05-02T09:30:00+08:00'
    assert flow['windows'][0]['window_end'] == '2025-05-02T09:45:00+08:00'
    assert flow['windows'][1]['window_end'] == '2025-05-02T10:00:00+08:00'
    assert flow['flow_consistency_warning'] is True
    assert flow['windows'][1]['total'] == 4


def test_normalizes_zggg_cutoff_weather_and_flow_without_raw_report():
    flight = normalize_flight(FLIGHT)
    context = normalize_zggg_context({**FLIGHT, 'zggg_context': {
        'prediction_cutoff_time': '2025-05-02 09:30:00', 'data_version': 'snapshot-v1',
        'departure_weather': {'metar': {'kind': 'METAR', 'airport': 'ZGGG',
            'issue_time': '2025-05-02T01:20:00Z', 'raw_text': 'must disappear',
            'features': {'visibility_m': 3000, 'precipitation': True}}},
        'flow_features': [{'horizon_minutes': 15, 'planned_takeoff': 2,
                           'planned_landing': 3, 'completed_takeoff': 1,
                           'completed_landing': 1}],
    }}, flight)
    assert context.prediction_cutoff.ToJsonString() == '2025-05-02T01:30:00Z'
    assert context.departure_weather.metar.features.visibility_m == 3000
    assert 'raw_text' not in str(context)


def test_rejects_zggg_context_for_unrelated_airport_or_wrong_cutoff():
    outside = normalize_flight({**FLIGHT, '计划起飞站四字码': 'ZBAA',
                                '计划到达站四字码': 'ZSPD'})
    payload = {**FLIGHT, 'zggg_context': {'prediction_cutoff_time':
        '2025-05-02 09:30:00', 'data_version': 'v1'}}
    with pytest.raises(ValueError, match='ZGGG'):
        normalize_zggg_context(payload, outside)
    with pytest.raises(ValueError, match='cutoff'):
        normalize_zggg_context({**FLIGHT, 'zggg_context': {
            'prediction_cutoff_time': '2025-05-02 09:31:00', 'data_version': 'v1'}},
            normalize_flight(FLIGHT))
