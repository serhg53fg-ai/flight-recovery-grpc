"""Compose flight-time and airport-flow inference with explicit fallbacks."""

from __future__ import annotations

from datetime import timezone
import json
import time

from flight.v1 import prediction_pb2 as pb

from training.flow import FLOW_SCHEMA_VERSION
from training.prompt import reconstruct_times
from services.inference.errors import InvalidInferenceRequest


def _invalid_output_diagnostic(error):
    message = str(error)
    patterns = (
        ('lacks JSON', 'JSON_MISSING'), ('未返回 JSON', 'JSON_MISSING'),
        ('invalid model JSON', 'JSON_INVALID'),
        ('fields are invalid', 'FIELDS_INVALID'), ('字段或时间顺序无效', 'FIELDS_INVALID'),
        ('finite numbers', 'VALUE_TYPE_INVALID'),
        ('outside range', 'VALUE_RANGE_INVALID'), ('超出', 'VALUE_RANGE_INVALID'),
        ('时间顺序', 'TIME_ORDER_INVALID'), ('time order', 'TIME_ORDER_INVALID'),
    )
    return next((code for fragment, code in patterns if fragment in message),
                'OUTPUT_INVALID_OTHER')


def _record_primary_fallback(trace_id, reason, error):
    diagnostic = (_invalid_output_diagnostic(error) if reason == 'PRIMARY_INVALID_OUTPUT'
                  else 'TIMEOUT' if reason == 'PRIMARY_TIMEOUT' else 'UNAVAILABLE')
    print(json.dumps({'event': 'worker.primary_fallback', 'trace_id': trace_id,
                      'reason': reason, 'diagnostic': diagnostic}), flush=True)


def _flow_row(request):
    features = {
        "airport": "ZGGG",
        "prediction_cutoff_time": request.zggg_context.prediction_cutoff.ToJsonString(),
        "flow_schema_version": FLOW_SCHEMA_VERSION,
    }
    for window in request.zggg_context.flow_features:
        horizon = window.horizon_minutes
        features.update({
            f"planned_takeoff_{horizon}m": window.planned_takeoff,
            f"planned_landing_{horizon}m": window.planned_landing,
            f"planned_total_{horizon}m": window.planned_takeoff + window.planned_landing,
            f"completed_takeoff_{horizon}m": window.completed_takeoff,
            f"completed_landing_{horizon}m": window.completed_landing,
            f"completed_total_{horizon}m": window.completed_takeoff + window.completed_landing,
        })
    return {"record_id": request.trace_id, "features": features, "labels": {}}


def _planned_flow(request, response):
    for source in request.zggg_context.flow_features:
        target = response.airport_flow.windows.add(horizon_minutes=source.horizon_minutes)
        target.takeoff = source.planned_takeoff
        target.landing = source.planned_landing
        target.total = source.planned_takeoff + source.planned_landing
    response.airport_flow.model_version = "planned-flow-fallback-v1"


def _model_flow(request, response, artifact):
    prediction = artifact.predict([_flow_row(request)])[0]
    for horizon in (15, 30, 60):
        target = response.airport_flow.windows.add(horizon_minutes=horizon)
        target.takeoff = max(0, int(round(prediction[f"takeoff_{horizon}m"])))
        target.landing = max(0, int(round(prediction[f"landing_{horizon}m"])))
        target.total = target.takeoff + target.landing
    response.airport_flow.model_version = str(
        getattr(artifact, "model_name", "flow-model"))[:128]


def _weather_stale(request):
    for side in (request.zggg_context.departure_weather,
                 request.zggg_context.arrival_weather):
        for name in ("metar", "taf"):
            if side.HasField(name):
                report = getattr(side, name)
                if report.stale or report.missing:
                    return True
    return False


class HistoricalFlightBackend:
    """Adapt a trusted local duration baseline artifact to the Worker contract."""
    source = pb.BASELINE
    prompt_version = "historical-duration-v1"

    def __init__(self, model, model_version="historical-flight-v1"):
        self.model = model
        self.model_version = str(model_version)[:128]

    def predict_request(self, request, cancelled, budget):
        if budget <= 0 or cancelled():
            raise TimeoutError("historical flight fallback cancelled")
        flight = request.flight
        features = {
            "flight_number": flight.flight_number,
            "tail_number": flight.tail_number,
            "aircraft_type": flight.aircraft_type,
            "flight_nature": flight.flight_nature,
            "departure_airport": flight.departure_airport,
            "arrival_airport": flight.arrival_airport,
            "planned_off_block": flight.planned_off_block.ToJsonString(),
            "planned_on_block": flight.planned_on_block.ToJsonString(),
            "planned_distance_miles": flight.planned_distance_miles,
            "planned_flight_minutes": flight.planned_flight_minutes,
            "planned_takeoff_count": flight.planned_takeoff_count,
            "planned_landing_count": flight.planned_landing_count,
            "planned_total_flow": flight.planned_total_flow,
        }
        output = self.model.predict([{"record_id": request.trace_id,
                                      "features": features, "labels": {}}])[0]
        components = {name: output[name] for name in (
            "off_block_delay_min", "taxi_out_min", "airborne_min", "taxi_in_min")}
        result = pb.Prediction()
        for field, value in zip(("off_block", "takeoff", "landing", "on_block"),
                                reconstruct_times(features["planned_off_block"], components)):
            getattr(result, field).FromJsonString(value)
        return result


class CompositeBackend:
    source = pb.LLM

    def __init__(self, flight_backend, flow_artifact, historical_backend=None):
        self.flight_backend = flight_backend
        self.flow_artifact = flow_artifact
        self.historical_backend = historical_backend
        self.source = flight_backend.source
        self.model_version = f"{flight_backend.model_version}+{getattr(flow_artifact, 'model_name', 'flow')}"[:128]

    def predict_response(self, request, cancelled, time_budget):
        if time_budget <= 0 or cancelled():
            raise TimeoutError("composite inference cancelled")
        if not request.HasField("zggg_context"):
            raise InvalidInferenceRequest("composite backend requires ZGGG context")
        deadline = time.monotonic() + time_budget
        response = pb.PredictResponse(
            trace_id=request.trace_id, source=self.source,
            model_version=self.model_version,
            data_version=request.zggg_context.data_version,
            weather_stale=_weather_stale(request),
        )
        try:
            _model_flow(request, response, self.flow_artifact)
        except Exception:
            if cancelled():
                raise TimeoutError("composite inference cancelled")
            response.ClearField("airport_flow")
            _planned_flow(request, response)
            response.flow_degraded = True
            response.flow_fallback_reason = 'FLOW_MODEL_UNAVAILABLE'
        remaining = deadline - time.monotonic()
        if remaining <= 0 or cancelled():
            raise TimeoutError("composite inference deadline exhausted")
        backend = self.flight_backend
        # Qwen checks its budget during generation. Leave room for CPU fallback.
        reserve = min(1.0, remaining * .2) if self.historical_backend is not None else 0
        primary_budget = remaining - reserve
        try:
            prediction = backend.predict_request(request, cancelled, primary_budget)
            _validate_flight_prediction(prediction, request.trace_id, backend)
        except InvalidInferenceRequest:
            raise
        except Exception as primary_error:
            if cancelled():
                raise TimeoutError("composite inference cancelled") from primary_error
            if self.historical_backend is None:
                raise RuntimeError("flight inference backends unavailable") from primary_error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("composite inference deadline exhausted")
            backend = self.historical_backend
            try:
                prediction = backend.predict_request(request, cancelled, remaining)
                _validate_flight_prediction(prediction, request.trace_id, backend)
            except Exception as fallback_error:
                raise RuntimeError("flight inference backends unavailable") from fallback_error
            response.flight_degraded = True
            response.flight_fallback_reason = (
                'PRIMARY_TIMEOUT' if isinstance(primary_error, TimeoutError) else
                'PRIMARY_INVALID_OUTPUT' if isinstance(primary_error, ValueError) else
                'PRIMARY_UNAVAILABLE')
            _record_primary_fallback(request.trace_id, response.flight_fallback_reason,
                                     primary_error)
        if cancelled() or time.monotonic() >= deadline:
            raise TimeoutError('composite inference deadline exhausted')
        response.prediction.CopyFrom(prediction)
        response.source = backend.source
        response.model_version = f"{backend.model_version}+{response.airport_flow.model_version}"[:128]
        response.prompt_version = str(getattr(backend, "prompt_version", ""))[:128]
        return response


def _validate_flight_prediction(prediction, trace_id, backend):
    from apps.flight.domain.prediction import validate_response
    validate_response(pb.PredictResponse(
        trace_id=trace_id, prediction=prediction, source=backend.source,
        model_version=backend.model_version, worker_id='composite-validation'), trace_id)
