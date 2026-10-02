from datetime import timedelta, timezone

import pytest
from flight.v1 import prediction_pb2 as pb

from apps.flight.domain.prediction import normalize_flight


def request():
    flight = normalize_flight({
        '航班号': 'CZ1001', '机尾号': 'B1001', '机型': 'A320',
        '计划起飞站四字码': 'ZGGG', '计划到达站四字码': 'ZSPD',
        '计划离港时间': '2025-05-02T01:30:00Z',
        '计划到港时间': '2025-05-02T03:45:00Z',
    })
    value = pb.PredictRequest(trace_id='trace', flight=flight)
    value.zggg_context.prediction_cutoff.CopyFrom(flight.planned_off_block)
    value.zggg_context.data_version = 'snapshot-v1'
    value.zggg_context.departure_weather.metar.missing = False
    value.zggg_context.departure_weather.metar.stale = True
    value.zggg_context.departure_weather.metar.kind = 'METAR'
    value.zggg_context.departure_weather.metar.airport = 'ZGGG'
    value.zggg_context.departure_weather.metar.issue_time.FromJsonString(
        '2025-05-02T01:20:00Z')
    for horizon, takeoff, landing in ((15, 2, 3), (30, 5, 6), (60, 9, 10)):
        value.zggg_context.flow_features.add(
            horizon_minutes=horizon, planned_takeoff=takeoff,
            planned_landing=landing, completed_takeoff=1, completed_landing=1)
    return value


def prediction(offset=10):
    planned = request().flight.planned_off_block.ToDatetime(tzinfo=timezone.utc)
    value = pb.Prediction()
    for field, minutes in (("off_block", offset), ("takeoff", offset + 10),
                           ("landing", offset + 110), ("on_block", offset + 120)):
        getattr(value, field).FromDatetime(planned + timedelta(minutes=minutes))
    return value


class FlightBackend:
    source = pb.LLM
    model_version = 'qwen-v2'
    prompt_version = 'zggg-weather-duration-prompt-v2'
    def __init__(self, error=None): self.error = error; self.budgets = []
    def predict_request(self, request, cancelled, budget):
        self.budgets.append(budget)
        if cancelled(): raise TimeoutError('cancelled')
        if self.error: raise self.error
        return prediction()


class FlowArtifact:
    model_name = 'gradient_boosting'
    def __init__(self, error=None): self.error = error
    def predict(self, rows):
        if self.error: raise self.error
        result = {'record_id': rows[0]['record_id']}
        for horizon, takeoff, landing in ((15, 3, 4), (30, 7, 8), (60, 12, 13)):
            result.update({f'takeoff_{horizon}m': takeoff,
                           f'landing_{horizon}m': landing,
                           f'total_{horizon}m': takeoff + landing})
        return [result]


def test_combines_qwen_flow_weather_and_model_identity():
    from services.inference.adapters.composite import CompositeBackend
    result = CompositeBackend(FlightBackend(), FlowArtifact()).predict_response(
        request(), lambda: False, 10)
    assert result.prediction == prediction()
    assert result.airport_flow.windows[0].total == 7
    assert result.airport_flow.model_version == 'gradient_boosting'
    assert result.weather_stale is True
    assert result.flow_degraded is False and result.flight_degraded is False
    assert result.data_version == 'snapshot-v1'
    assert 'qwen-v2' in result.model_version


def test_missing_zggg_context_is_invalid_argument_at_worker_boundary():
    import grpc
    from services.inference.adapters.composite import CompositeBackend
    from services.inference.server import InferenceServicer
    from tests.unit.test_worker import Aborted, Context

    incoming = request()
    incoming.ClearField('zggg_context')
    worker = InferenceServicer(CompositeBackend(FlightBackend(), FlowArtifact()), 'composite')
    with pytest.raises(Aborted) as error:
        worker.Predict(incoming, Context())
    assert error.value.code == grpc.StatusCode.INVALID_ARGUMENT


def test_independent_historical_and_planned_flow_fallbacks():
    from services.inference.adapters.composite import CompositeBackend
    historical = FlightBackend(); historical.source = pb.TEST
    historical.model_version = 'historical-v1'
    result = CompositeBackend(
        FlightBackend(RuntimeError('qwen unavailable')),
        FlowArtifact(RuntimeError('flow unavailable')),
        historical_backend=historical,
    ).predict_response(request(), lambda: False, 10)
    assert result.flight_degraded is True and result.flow_degraded is True
    assert result.airport_flow.windows[0].takeoff == 2
    assert result.airport_flow.windows[0].landing == 3
    assert result.source == pb.TEST
    assert 'historical-v1' in result.model_version


def test_every_flight_backend_failure_never_fabricates_result():
    from services.inference.adapters.composite import CompositeBackend
    backend = CompositeBackend(
        FlightBackend(RuntimeError('qwen unavailable')), FlowArtifact(),
        historical_backend=FlightBackend(RuntimeError('history unavailable')))
    with pytest.raises(RuntimeError, match='flight inference'):
        backend.predict_response(request(), lambda: False, 10)


def test_cancellation_and_shared_deadline_are_enforced(monkeypatch):
    from services.inference.adapters import composite
    CompositeBackend = composite.CompositeBackend
    clock = iter((100.0, 100.2, 100.4))
    monkeypatch.setattr(composite.time, 'monotonic', lambda: next(clock))
    flight = FlightBackend()
    CompositeBackend(flight, FlowArtifact()).predict_response(request(), lambda: False, 1)
    assert 0 < flight.budgets[0] <= .8
    with pytest.raises(TimeoutError):
        CompositeBackend(FlightBackend(), FlowArtifact()).predict_response(
            request(), lambda: True, 1)


def test_historical_primary_composite_does_not_load_failed_qwen(tmp_path, monkeypatch):
    import argparse
    import pickle
    from services.inference import server
    from services.inference.adapters import qwen
    from training import flow_model
    from training.baselines import HistoricalMedianBaseline

    model_path = tmp_path / 'historical.pkl'
    model_path.write_bytes(pickle.dumps(HistoricalMedianBaseline()))
    monkeypatch.setattr(qwen, 'QwenBackend', lambda *args, **kwargs: pytest.fail('Qwen loaded'))
    monkeypatch.setattr(flow_model, 'load_flow_artifact', lambda *args: FlowArtifact())
    args = argparse.Namespace(flow_model_path=str(tmp_path / 'flow.pkl'),
        flow_manifest_hash='hash', historical_model_path=str(model_path),
        composite_flight_mode='historical', model_path=None, adapter_path=None,
        max_new_tokens=128, max_seconds=20, output_mode=None)
    backend = server.build_composite_backend(args)
    assert backend.flight_backend.source == pb.BASELINE
    assert backend.historical_backend is None
    assert backend.source == pb.BASELINE


def test_composite_startup_rejects_release_identity_mismatch(tmp_path, monkeypatch):
    import argparse
    import pickle
    from services.inference import server
    from training import flow_model
    from training.baselines import HistoricalMedianBaseline

    model_path = tmp_path / 'historical.pkl'
    model_path.write_bytes(pickle.dumps(HistoricalMedianBaseline()))
    monkeypatch.setattr(flow_model, 'load_flow_artifact', lambda *args: FlowArtifact())
    args = argparse.Namespace(flow_model_path=str(tmp_path / 'flow.pkl'),
        flow_manifest_hash='hash', historical_model_path=str(model_path),
        composite_flight_mode='historical', model_path=None, adapter_path=None,
        max_new_tokens=128, max_seconds=20, output_mode=None,
        expected_source='BASELINE', expected_model_version='different-version')
    with pytest.raises(ValueError, match='release identity'):
        server.build_composite_backend(args)


def test_qwen_timeout_leaves_budget_for_historical_fallback(monkeypatch):
    from services.inference.adapters import composite
    clock = [100.0]
    monkeypatch.setattr(composite.time, 'monotonic', lambda: clock[0])
    class TimedPrimary(FlightBackend):
        def predict_request(self, request, cancelled, budget):
            self.budgets.append(budget)
            clock[0] += budget
            raise TimeoutError('private backend details')
    primary = TimedPrimary()
    historical = FlightBackend(); historical.source = pb.BASELINE
    historical.model_version = 'historical-flight-v1'
    result = composite.CompositeBackend(primary, FlowArtifact(), historical).predict_response(
        request(), lambda: False, 10)
    assert 0 < primary.budgets[0] < 10
    assert historical.budgets[0] > 0
    assert result.source == pb.BASELINE and result.flight_degraded
    assert result.flight_fallback_reason == 'PRIMARY_TIMEOUT'


def test_invalid_primary_prediction_falls_back_before_worker_validation():
    from services.inference.adapters.composite import CompositeBackend
    class InvalidPrimary(FlightBackend):
        def predict_request(self, request, cancelled, budget):
            value = prediction()
            value.takeoff.CopyFrom(value.on_block)
            return value
    fallback = FlightBackend(); fallback.source = pb.BASELINE
    result = CompositeBackend(InvalidPrimary(), FlowArtifact(), fallback).predict_response(
        request(), lambda: False, 10)
    assert result.prediction == prediction()
    assert result.flight_fallback_reason == 'PRIMARY_INVALID_OUTPUT'


@pytest.mark.parametrize(('message', 'diagnostic'), [
    ('model output lacks JSON', 'JSON_MISSING'),
    ('invalid model JSON', 'JSON_INVALID'),
    ('duration output fields are invalid', 'FIELDS_INVALID'),
    ('duration values must be finite numbers', 'VALUE_TYPE_INVALID'),
    ('duration is outside range', 'VALUE_RANGE_INVALID'),
    ('响应时间顺序无效', 'TIME_ORDER_INVALID'),
    ('unclassified parser failure', 'OUTPUT_INVALID_OTHER'),
])
def test_primary_invalid_output_logs_bounded_diagnostic_without_raw_output(capsys, message, diagnostic):
    from services.inference.adapters.composite import CompositeBackend

    primary = FlightBackend(ValueError(message + ' SECRET_RAW_OUTPUT'))
    fallback = FlightBackend(); fallback.source = pb.BASELINE
    result = CompositeBackend(primary, FlowArtifact(), fallback).predict_response(
        request(), lambda: False, 10)

    assert result.flight_fallback_reason == 'PRIMARY_INVALID_OUTPUT'
    event = __import__('json').loads(capsys.readouterr().out)
    assert event == {'event': 'worker.primary_fallback', 'trace_id': 'trace',
                     'reason': 'PRIMARY_INVALID_OUTPUT', 'diagnostic': diagnostic}
    assert 'SECRET_RAW_OUTPUT' not in str(event)


def test_flow_fallback_reason_and_late_primary_are_explicit(monkeypatch):
    from services.inference.adapters import composite
    clock = [100.0]
    monkeypatch.setattr(composite.time, 'monotonic', lambda: clock[0])
    class LatePrimary(FlightBackend):
        def predict_request(self, request, cancelled, budget):
            clock[0] += budget + 10
            return prediction()
    with pytest.raises(TimeoutError):
        composite.CompositeBackend(LatePrimary(), FlowArtifact(), FlightBackend()).predict_response(
            request(), lambda: False, 10)
    result = composite.CompositeBackend(FlightBackend(), FlowArtifact(RuntimeError())).predict_response(
        request(), lambda: False, 10)
    assert result.flow_fallback_reason == 'FLOW_MODEL_UNAVAILABLE'
