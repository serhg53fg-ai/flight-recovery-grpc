"""Probe actual inference identities before admitting a Worker to a Gateway pool."""
import uuid

import grpc

from apps.flight.domain.prediction import normalize_flight, normalize_zggg_context
from flight.v1 import prediction_pb2 as pb, prediction_pb2_grpc as rpc


def probe_request():
    flight = normalize_flight({
        'flight_number': 'CZ9001', 'tail_number': 'B1234', 'aircraft_type': 'A320',
        'departure_airport': 'ZGGG', 'arrival_airport': 'ZSPD',
        'planned_off_block': '2025-05-15T09:00:00+08:00',
        'planned_on_block': '2025-05-15T11:00:00+08:00',
        'planned_flight_minutes': 120,
    })
    context = normalize_zggg_context({'zggg_context': {
        'prediction_cutoff': '2025-05-15T09:00:00+08:00',
        'data_version': 'resident-identity-probe-v1',
        'flow_features': [{'horizon_minutes': horizon,
                           'planned_takeoff': horizon // 15,
                           'planned_landing': horizon // 15,
                           'completed_takeoff': 0, 'completed_landing': 0}
                          for horizon in (15, 30, 60)],
    }}, flight)
    return pb.PredictRequest(trace_id='resident-probe-' + uuid.uuid4().hex,
                             flight=flight, zggg_context=context)


def probe_worker_identity(address: str, worker_id: str, expected: dict, timeout: float = 5,
                          fallback_identity: dict | None = None) -> dict:
    """Reject a healthy-but-wrong model, source, flow model, or Worker ID."""
    channel = grpc.insecure_channel(address, options=[('grpc.enable_retries', 0)])
    try:
        result = rpc.InferenceWorkerStub(channel).Predict(probe_request(), timeout=timeout)
    finally:
        channel.close()
    actual = {'worker_id': result.worker_id,
              'source': pb.PredictionSource.Name(result.source),
              'model_version': result.model_version,
              'flow_model_version': result.airport_flow.model_version if result.HasField('airport_flow') else '',
              'flow_degraded': result.flow_degraded, 'flight_degraded': result.flight_degraded}
    selected = dict(expected)
    if result.flight_degraded and fallback_identity:
        if (result.flight_fallback_reason not in
                ('PRIMARY_TIMEOUT', 'PRIMARY_INVALID_OUTPUT', 'PRIMARY_UNAVAILABLE') or
                result.prompt_version != fallback_identity['prompt_version']):
            raise ValueError('Worker fallback identity mismatch')
        selected.update(source=fallback_identity['source'],
                        model_version=fallback_identity['flight_model_version'] + '+' + expected['flow_model_version'])
    elif result.flight_degraded:
        raise ValueError('Worker undeclared fallback')
    for key, value in {'worker_id': worker_id, **selected, 'flow_degraded': False}.items():
        if actual[key] != value:
            raise ValueError(f'Worker {worker_id} {key} mismatch: {actual[key]} != {value}')
    return actual
