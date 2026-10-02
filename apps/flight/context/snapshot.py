"""Freeze user-visible flight input and cutoff-safe airport context."""

from __future__ import annotations

from datetime import timedelta
import hashlib
import json

from apps.flight.domain.prediction import normalize_flight, normalize_zggg_context
from training.flow_features import FLOW_HORIZONS, build_flow_features
from training.weather import select_weather_snapshot
from training.zggg_dataset import _snapshot_json


def _canonical_flight(flight) -> dict:
    result = {'航班号': flight.flight_number, '机尾号': flight.tail_number,
              '机型': flight.aircraft_type, '计划起飞站四字码': flight.departure_airport,
              '计划到达站四字码': flight.arrival_airport,
              '计划离港时间': flight.planned_off_block.ToJsonString(),
              '计划到港时间': flight.planned_on_block.ToJsonString()}
    if flight.HasField('flight_nature'):
        result['性质'] = flight.flight_nature
    for field, name in (('planned_distance_miles', '计划地面航程_Mile'),
                        ('planned_flight_minutes', '计划航段时间\n（24年同航季平均值）'),
                        ('planned_takeoff_count', '计划起飞数'),
                        ('planned_landing_count', '计划降落数'),
                        ('planned_total_flow', '计划总流量')):
        if flight.HasField(field):
            result[name] = getattr(flight, field)
    return result


def _weather_for(dataset, airport, cutoff, target):
    candidates = [(report, received) for report, received in dataset['weather'].get(airport, ())
                  if report.issue_time <= cutoff and (received is None or received <= cutoff)]
    result, basis = {}, {}
    for kind in ('METAR', 'TAF'):
        selected = select_weather_snapshot([report for report, _ in candidates], cutoff, target, kind)
        result[kind.lower()] = _snapshot_json(selected)
        if selected.report is None:
            basis[kind.lower()] = 'missing'
        else:
            received = next(received for report, received in candidates if report is selected.report)
            basis[kind.lower()] = 'received_time' if received is not None else 'issue_time_assumed'
    return result, basis


def prepare_replay_submission(flights: list[dict], input_timezone: str, dataset: dict) -> dict:
    if not isinstance(flights, list) or not 1 <= len(flights) <= 1000:
        raise ValueError('航班数组长度必须为1～1000')
    normalized = [normalize_flight(value, input_timezone) for value in flights]
    cutoffs = [flight.planned_off_block.ToDatetime(tzinfo=dataset['coverage_start'].tzinfo)
               for flight in normalized]
    maximum = timedelta(minutes=max(FLOW_HORIZONS))
    for flight, cutoff in zip(normalized, cutoffs):
        if ((flight.departure_airport == 'ZGGG') == (flight.arrival_airport == 'ZGGG')):
            raise ValueError('ZGGG 回放必须是单侧白云机场航班')
        if cutoff - maximum < dataset['coverage_start'] or cutoff + maximum > dataset['coverage_end']:
            raise ValueError('机场计划覆盖不足，无法生成完整流量特征')
    features = build_flow_features(dataset['frame'], cutoffs, input_timezone)
    prepared, snapshots = [], []
    for flight, cutoff, flow in zip(normalized, cutoffs, features):
        departure = flight.departure_airport
        arrival = flight.arrival_airport
        arrival_target = flight.planned_on_block.ToDatetime(tzinfo=cutoff.tzinfo)
        departure_weather, departure_basis = _weather_for(dataset, departure, cutoff, cutoff)
        arrival_weather, arrival_basis = _weather_for(dataset, arrival, cutoff, arrival_target)
        windows = [dict(horizon_minutes=horizon,
                        planned_takeoff=flow[f'planned_takeoff_{horizon}m'],
                        planned_landing=flow[f'planned_landing_{horizon}m'],
                        completed_takeoff=flow[f'completed_takeoff_{horizon}m'],
                        completed_landing=flow[f'completed_landing_{horizon}m'])
                   for horizon in FLOW_HORIZONS]
        context = {'prediction_cutoff': flow['prediction_cutoff_time'],
                   'data_version': dataset['dataset_manifest_hash'],
                   'departure_weather': departure_weather,
                   'arrival_weather': arrival_weather,
                   'flow_features': windows}
        canonical = _canonical_flight(flight)
        normalize_zggg_context({'zggg_context': context}, flight, input_timezone)
        prepared.append({**canonical, 'zggg_context': context})
        snapshots.append({'cutoff': flow['prediction_cutoff_time'], 'context': context,
                          'weather_availability_basis': {
                              **{f'departure_{key}': value for key, value in departure_basis.items()},
                              **{f'arrival_{key}': value for key, value in arrival_basis.items()}},
                          'schedule_as_of_known': dataset['schedule_as_of_known'],
                          'observed_event_availability_basis': 'event_time_assumed'})
    identity = {'flights': prepared, 'weather_basis': [value['weather_availability_basis'] for value in snapshots],
                'feature_contract_version': dataset['feature_contract_version']}
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    snapshot_hash = hashlib.sha256(encoded.encode('utf-8')).hexdigest()
    return {'flights': prepared,
            'submission_context': {'schema_version': 'replay-submission-v1',
                                   'replay_dataset_id': dataset['dataset_id'],
                                   'dataset_manifest_hash': dataset['dataset_manifest_hash'],
                                   'source_sha256': dataset['source_sha256'],
                                   'weather_sha256': dataset['weather_sha256'],
                                   'feature_contract_version': dataset['feature_contract_version'],
                                   'snapshot_hash': snapshot_hash,
                                   'snapshots': snapshots}}
