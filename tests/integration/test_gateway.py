from __future__ import annotations

import socket
import json
import subprocess
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc
import pytest
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from google.protobuf.timestamp_pb2 import Timestamp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "generated"))

from flight.v1 import prediction_pb2, prediction_pb2_grpc  # noqa: E402


GATEWAY_BINARY = ROOT / "build" / "phase2-release" / "flight_gateway"


def _free_address() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{sock.getsockname()[1]}"


def _timestamp(seconds: int) -> Timestamp:
    return Timestamp(seconds=seconds)


def _request(trace_id: str = "trace-123") -> prediction_pb2.PredictRequest:
    return prediction_pb2.PredictRequest(
        trace_id=trace_id,
        flight=prediction_pb2.FlightFeatures(
            flight_id="flight-20260912-001",
            flight_number="MU5101",
            tail_number="B-1234",
            aircraft_type="A320",
            departure_airport="ZSSS",
            arrival_airport="ZBAA",
            planned_off_block=_timestamp(1_789_124_400),
            planned_on_block=_timestamp(1_789_131_600),
            departure_metar="ZSSS 120000Z 08003MPS CAVOK 23/18 Q1012",
            arrival_metar="ZBAA 120000Z 02002MPS CAVOK 21/14 Q1015",
            planned_distance_miles=671.25,
            planned_flight_minutes=120.5,
            planned_takeoff_count=8,
            planned_landing_count=7,
            planned_total_flow=15,
        ),
    )


def _valid_response(trace_id: str) -> prediction_pb2.PredictResponse:
    return prediction_pb2.PredictResponse(
        trace_id=trace_id,
        prediction=prediction_pb2.Prediction(
            off_block=_timestamp(1_789_124_700),
            takeoff=_timestamp(1_789_125_600),
            landing=_timestamp(1_789_130_900),
            on_block=_timestamp(1_789_131_900),
        ),
        source=prediction_pb2.TEST,
        model_version="fixture-v1",
        worker_id="worker-fixture-1",
        inference_ms=17,
    )


class TestWorker(prediction_pb2_grpc.InferenceWorkerServicer):
    __test__ = False

    def __init__(self):
        self.calls: list[bytes] = []
        self.handler = lambda request, context: _valid_response(request.trace_id)
        self.lock = threading.Lock()

    def Predict(self, request, context):  # noqa: N802 - generated gRPC API
        with self.lock:
            self.calls.append(request.SerializeToString())
            handler = self.handler
        return handler(request, context)


@pytest.fixture
def worker():
    servicer = TestWorker()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    prediction_pb2_grpc.add_InferenceWorkerServicer_to_server(servicer, server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    address = _free_address()
    assert server.add_insecure_port(address) != 0
    server.start()
    health_servicer.set('', health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set('flight.v1.InferenceWorker', health_pb2.HealthCheckResponse.SERVING)
    try:
        yield servicer, address
    finally:
        server.stop(grace=None).wait()


@pytest.fixture
def start_gateway(tmp_path):
    processes: list[subprocess.Popen[str]] = []

    def start(worker_address: str, *, max_inflight: int = 4, timeout_ms: int = 2_000):
        listen_address = _free_address()
        config_path = tmp_path / f'gateway-{len(processes)}.json'
        config_path.write_text(json.dumps({
            'listen': listen_address,
            'rpc_timeout_ms': timeout_ms,
            'health_interval_ms': 100,
            'health_timeout_ms': 50,
            'failure_threshold': 3,
            'open_cooldown_ms': 1000,
            'minimum_retry_budget_ms': 50,
            'workers': [{'id': 'fixture-worker', 'address': worker_address,
                         'capacity': max_inflight, 'enabled': True}],
        }))
        process = subprocess.Popen(
            [
                str(GATEWAY_BINARY),
                "--config",
                str(config_path),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        channel = grpc.insecure_channel(
            listen_address,
            options=(
                ("grpc.max_receive_message_length", 2 * 1024 * 1024),
                ("grpc.max_send_message_length", 2 * 1024 * 1024),
            ),
        )
        try:
            grpc.channel_ready_future(channel).result(timeout=5)
            health_stub = health_pb2_grpc.HealthStub(channel)
            deadline = time.monotonic() + 5
            while health_stub.Check(health_pb2.HealthCheckRequest(service=''), timeout=1).status != health_pb2.HealthCheckResponse.SERVING:
                if time.monotonic() >= deadline:
                    raise TimeoutError('gateway health did not become serving')
                time.sleep(0.05)
        except Exception:
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(f"gateway did not start\nstdout:\n{stdout}\nstderr:\n{stderr}")
        return prediction_pb2_grpc.PredictionGatewayStub(channel), channel

    try:
        yield start
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


def test_forwards_the_complete_request_and_valid_worker_response(worker, start_gateway):
    servicer, worker_address = worker
    request = _request()
    stub, channel = start_gateway(worker_address)

    response = stub.Predict(request, timeout=2)

    assert servicer.calls == [request.SerializeToString()]
    assert response.trace_id == "trace-123"
    assert response.prediction == _valid_response("trace-123").prediction
    assert response.source == prediction_pb2.TEST
    assert response.model_version == "fixture-v1"
    assert response.worker_id == "worker-fixture-1"
    assert response.inference_ms == 17
    assert response.gateway_ms >= 0
    channel.close()


def test_forwards_additive_zggg_fields_without_alteration(worker, start_gateway):
    servicer, worker_address = worker
    request = _request()
    request.zggg_context.prediction_cutoff.CopyFrom(request.flight.planned_off_block)
    request.zggg_context.data_version = 'zggg-v1'
    request.zggg_context.departure_weather.metar.airport = 'ZSSS'
    request.zggg_context.departure_weather.metar.kind = 'METAR'
    request.zggg_context.departure_weather.metar.issue_time.CopyFrom(
        request.flight.planned_off_block)
    request.zggg_context.flow_features.add(
        horizon_minutes=15, planned_takeoff=2, planned_landing=3)

    def enriched(incoming, _context):
        response = _valid_response(incoming.trace_id)
        response.airport_flow.windows.add(
            horizon_minutes=15, takeoff=2, landing=3, total=5)
        response.data_version = incoming.zggg_context.data_version
        response.weather_stale = False
        response.flow_degraded = False
        return response

    servicer.handler = enriched
    stub, channel = start_gateway(worker_address)
    result = stub.Predict(request, timeout=2)
    assert servicer.calls == [request.SerializeToString()]
    assert result.airport_flow.windows[0].total == 5
    assert result.data_version == 'zggg-v1'
    channel.close()


def test_propagates_worker_unavailable_status_and_details(worker, start_gateway):
    servicer, worker_address = worker

    def unavailable(_request, context):
        context.abort(grpc.StatusCode.UNAVAILABLE, "fixture worker unavailable")

    servicer.handler = unavailable
    stub, channel = start_gateway(worker_address)

    with pytest.raises(grpc.RpcError) as error:
        stub.Predict(_request(), timeout=2)

    assert error.value.code() is grpc.StatusCode.UNAVAILABLE
    assert error.value.details() == "fixture worker unavailable"
    channel.close()


def test_limits_downstream_deadline_to_configured_timeout(worker, start_gateway):
    servicer, worker_address = worker
    observed_remaining: list[float] = []

    def slow(_request, context):
        observed_remaining.append(context.time_remaining())
        while context.is_active():
            time.sleep(0.01)
        context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "worker deadline exceeded")

    servicer.handler = slow
    stub, channel = start_gateway(worker_address, timeout_ms=150)
    started = time.monotonic()

    with pytest.raises(grpc.RpcError) as error:
        stub.Predict(_request(), timeout=3)

    elapsed = time.monotonic() - started
    assert error.value.code() is grpc.StatusCode.DEADLINE_EXCEEDED
    assert observed_remaining and 0 < observed_remaining[0] <= 0.25
    assert elapsed < 1.0
    channel.close()


def test_rejects_overload_without_waiting_for_the_active_request(worker, start_gateway):
    servicer, worker_address = worker
    entered = threading.Event()
    release = threading.Event()

    def blocked(request, context):
        entered.set()
        while context.is_active() and not release.wait(0.01):
            pass
        return _valid_response(request.trace_id)

    servicer.handler = blocked
    stub, channel = start_gateway(worker_address, max_inflight=1)
    first = stub.Predict.future(_request("first"), timeout=2)
    assert entered.wait(timeout=1)
    started = time.monotonic()

    with pytest.raises(grpc.RpcError) as error:
        stub.Predict(_request("second"), timeout=1)

    assert error.value.code() is grpc.StatusCode.RESOURCE_EXHAUSTED
    assert time.monotonic() - started < 0.5
    release.set()
    assert first.result(timeout=1).trace_id == "first"
    channel.close()


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda request: setattr(request, "trace_id", ""), id="missing-trace"),
        pytest.param(
            lambda request: setattr(request.flight, "flight_number", ""),
            id="missing-flight-field",
        ),
        pytest.param(
            lambda request: request.flight.planned_on_block.CopyFrom(
                _timestamp(request.flight.planned_off_block.seconds - 1)
            ),
            id="reversed-planned-times",
        ),
    ],
)
def test_rejects_invalid_input_before_calling_worker(worker, start_gateway, mutate):
    servicer, worker_address = worker
    request = _request()
    mutate(request)
    stub, channel = start_gateway(worker_address)

    with pytest.raises(grpc.RpcError) as error:
        stub.Predict(request, timeout=2)

    assert error.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert servicer.calls == []
    channel.close()


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda response: response.ClearField("prediction"), id="missing-prediction"),
        pytest.param(lambda response: setattr(response, "trace_id", "wrong-trace"), id="wrong-trace"),
        pytest.param(
            lambda response: setattr(response, "source", prediction_pb2.SOURCE_UNSPECIFIED),
            id="unknown-source",
        ),
        pytest.param(lambda response: setattr(response, "model_version", ""), id="missing-model"),
        pytest.param(lambda response: setattr(response, "worker_id", ""), id="missing-worker"),
        pytest.param(
            lambda response: response.prediction.landing.CopyFrom(
                _timestamp(response.prediction.takeoff.seconds - 1)
            ),
            id="reversed-prediction-times",
        ),
    ],
)
def test_rejects_malformed_worker_response(worker, start_gateway, mutate):
    servicer, worker_address = worker

    def malformed(request, _context):
        response = _valid_response(request.trace_id)
        mutate(response)
        return response

    servicer.handler = malformed
    stub, channel = start_gateway(worker_address)

    with pytest.raises(grpc.RpcError) as error:
        stub.Predict(_request(), timeout=2)

    assert error.value.code() is grpc.StatusCode.DATA_LOSS
    channel.close()


def test_propagates_client_cancellation_and_releases_capacity(worker, start_gateway):
    servicer, worker_address = worker
    entered = threading.Event()
    downstream_cancelled = threading.Event()

    def until_cancelled(_request, context):
        entered.set()
        context.add_callback(downstream_cancelled.set)
        while context.is_active():
            time.sleep(0.01)
        context.abort(grpc.StatusCode.CANCELLED, "cancelled")

    servicer.handler = until_cancelled
    stub, channel = start_gateway(worker_address, max_inflight=1, timeout_ms=5_000)
    call = stub.Predict.future(_request("cancel-me"), timeout=5)
    assert entered.wait(timeout=1)

    assert call.cancel()
    with pytest.raises(grpc.FutureCancelledError):
        call.result(timeout=1)
    assert downstream_cancelled.wait(timeout=1)

    servicer.handler = lambda request, _context: _valid_response(request.trace_id)
    deadline = time.monotonic() + 1
    while True:
        try:
            response = stub.Predict(_request("after-cancel"), timeout=1)
            break
        except grpc.RpcError as error:
            if error.code() is not grpc.StatusCode.RESOURCE_EXHAUSTED or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
    assert response.trace_id == "after-cancel"
    channel.close()


def test_rejects_requests_larger_than_one_mib(worker, start_gateway):
    servicer, worker_address = worker
    request = _request()
    request.flight.departure_metar = "x" * (1024 * 1024)
    stub, channel = start_gateway(worker_address)

    with pytest.raises(grpc.RpcError) as error:
        stub.Predict(request, timeout=2)

    assert error.value.code() is grpc.StatusCode.RESOURCE_EXHAUSTED
    assert servicer.calls == []
    channel.close()
