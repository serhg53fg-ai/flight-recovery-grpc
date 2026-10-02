import threading
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
import json
from flight.v1 import prediction_pb2 as pb
from flight.v1 import prediction_pb2_grpc as rpc
from apps.flight.domain.prediction import normalize_flight
from services.inference.server import InferenceServicer
from services.inference.adapters.test_backend import TestBackend
from services.inference.adapters.qwen import duration_prediction, load_adapter_metadata, parse_output, QwenBackend


def test_generated_contract_exposes_gateway_admin_stub():
    assert hasattr(rpc, 'GatewayAdminStub')


def request():
    return pb.PredictRequest(trace_id='trace', flight=normalize_flight({
        'flight_number': 'CZ123', 'tail_number': 'B1234', 'aircraft_type': 'A320',
        'departure_airport': 'ZGGG', 'arrival_airport': 'ZSPD',
        'planned_departure_time': '2025-05-02T01:30:00Z',
        'planned_arrival_time': '2025-05-02T03:45:00Z',
    }))


class Aborted(Exception):
    def __init__(self, code): self.code = code


class Context:
    def abort(self, code, detail): raise Aborted(code)
    def is_active(self): return True
    def time_remaining(self): return 10


def test_worker_returns_deterministic_explicit_test_prediction():
    req = request()
    before = req.SerializeToString()
    worker = InferenceServicer(TestBackend(), 'test-1')
    a, b = worker.Predict(req, Context()), worker.Predict(req, Context())
    assert a.source == pb.TEST and a.worker_id == 'test-1' and a.trace_id == 'trace'
    assert a.prediction == b.prediction
    assert a.prediction.off_block.ToJsonString() == '2025-05-02T01:40:00Z'
    assert a.prediction.takeoff.ToJsonString() == '2025-05-02T01:50:00Z'
    assert a.prediction.landing.ToJsonString() == '2025-05-02T03:45:00Z'
    assert a.prediction.on_block.ToJsonString() == '2025-05-02T03:55:00Z'
    assert req.SerializeToString() == before


def test_invalid_request_rejected_before_backend():
    class NeverCalled(TestBackend):
        def predict(self, *args): raise AssertionError('must not call')
    with pytest.raises(Aborted) as e:
        InferenceServicer(NeverCalled(), 'worker').Predict(pb.PredictRequest(), Context())
    assert e.value.code == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize('fault,expected', [('unavailable',grpc.StatusCode.UNAVAILABLE),
                                          ('malformed',grpc.StatusCode.DATA_LOSS)])
def test_backend_failure_does_not_return_success(fault, expected):
    class Broken(TestBackend):
        def predict(self, *args):
            if fault == 'unavailable': raise RuntimeError('model unavailable')
            return pb.Prediction()
    with pytest.raises(Aborted) as e:
        InferenceServicer(Broken(), 'worker').Predict(request(), Context())
    assert e.value.code == expected


def test_worker_rejects_overload_and_releases_permit():
    entered, release = threading.Event(), threading.Event()
    class Slow(TestBackend):
        def predict(self, *args):
            entered.set()
            assert release.wait(2)
            return super().predict(*args)
    worker = InferenceServicer(Slow(), 'worker', max_inflight=1)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(worker.Predict, request(), Context())
        assert entered.wait(1)
        try:
            with pytest.raises(Aborted) as e: worker.Predict(request(), Context())
            assert e.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED
        finally: release.set()
        assert first.result().source == pb.TEST
    assert worker.Predict(request(), Context()).source == pb.TEST


def test_qwen_parser_checks_json_time_fields():
    text = '''```json
    {"实际离港时间":"2025-05-02T01:40:00Z","实际起飞时间":"2025-05-02T01:50:00Z",
     "实际落地时间":"2025-05-02T03:45:00Z","实际到港时间":"2025-05-02T03:55:00Z"}
    ```'''
    assert parse_output(text).off_block.ToJsonString() == '2025-05-02T01:40:00Z'
    for invalid in ('not json', '{}', text.replace('03:45', '00:45'), text.replace('01:40:00Z', 'bad')):
        with pytest.raises(ValueError): parse_output(invalid)


def test_qwen_missing_local_model_fails_without_download(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        QwenBackend(tmp_path/'missing')
    assert list(tmp_path.iterdir()) == []


def test_duration_adapter_metadata_and_prediction_are_explicit(tmp_path):
    base = tmp_path / "base"; base.mkdir()
    (base / "config.json").write_text('{"model_type":"qwen2"}')
    (base / "model.safetensors").write_text("weights")
    from training.model_identity import model_fingerprint
    adapter = tmp_path / "adapter"; adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    fingerprint = model_fingerprint(base)
    (adapter / "adapter_metadata.json").write_text(json.dumps({"schema_version":"flight-adapter-v1","output_mode":"duration_components","prompt_version":"flight-duration-prompt-v1","dataset_manifest_hash":"hash","base_model_version":"base","base_model_fingerprint":fingerprint,"adapter_version":"adapter-v1"}))
    metadata = load_adapter_metadata(adapter, "base", base)
    prediction = duration_prediction('{"off_block_delay_min":-5,"taxi_out_min":20,"airborne_min":100,"taxi_in_min":10}', request().flight)
    assert metadata["adapter_version"] == "adapter-v1"
    updated = dict(metadata, fit_scope="train+validation")
    (adapter / "adapter_metadata.json").write_text(json.dumps(updated))
    assert load_adapter_metadata(adapter, "base", base)["fit_scope"] == "train+validation"
    (adapter / "adapter_metadata.json").write_text(json.dumps({**updated, "fit_scope": "test"}))
    with pytest.raises(ValueError): load_adapter_metadata(adapter, "base", base)
    (adapter / "adapter_metadata.json").write_text(json.dumps(metadata))
    assert prediction.off_block.ToJsonString() == "2025-05-02T01:25:00Z"
    with pytest.raises(ValueError): load_adapter_metadata(adapter, "other-base", base)
    other = tmp_path / "other" / "base"; other.mkdir(parents=True)
    (other / "config.json").write_text('{"model_type":"qwen2"}')
    (other / "model.safetensors").write_text("changed")
    with pytest.raises(ValueError): load_adapter_metadata(adapter, "base", other)


def test_worker_loads_v2_weather_adapter_and_keeps_v1_compatible(tmp_path):
    base = tmp_path / "base"; base.mkdir()
    (base / "config.json").write_text('{"model_type":"qwen2"}')
    (base / "model.safetensors").write_text("weights")
    from training.model_identity import model_fingerprint
    adapter = tmp_path / "adapter"; adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    metadata = {"schema_version": "flight-adapter-v2", "output_mode": "duration_components",
        "prompt_version": "zggg-weather-duration-prompt-v2", "dataset_manifest_hash": "hash",
        "base_model_version": "base", "base_model_fingerprint": model_fingerprint(base),
        "adapter_version": "zggg-v2", "fit_scope": "train+validation",
        "airport_scope": "ZGGG", "weather_feature_version": "aviation-weather-v1"}
    (adapter / "adapter_metadata.json").write_text(json.dumps(metadata))
    assert load_adapter_metadata(adapter, "base", base)["airport_scope"] == "ZGGG"


def test_worker_accepts_complete_composite_response():
    class Composite:
        source = pb.LLM
        model_version = 'composite-v1'
        def predict_response(self, incoming, _cancelled, _budget):
            result = pb.PredictResponse(
                trace_id=incoming.trace_id, prediction=TestBackend().predict(
                    incoming.flight, lambda: False, 1), source=pb.LLM,
                model_version='qwen+flow', data_version='snapshot-v1',
                prompt_version='zggg-weather-duration-prompt-v2')
            result.airport_flow.model_version = 'flow-v1'
            result.airport_flow.windows.add(horizon_minutes=15, takeoff=2,
                                            landing=3, total=5)
            return result
    result = InferenceServicer(Composite(), 'composite-worker').Predict(request(), Context())
    assert result.worker_id == 'composite-worker'
    assert result.airport_flow.windows[0].total == 5
    assert result.data_version == 'snapshot-v1'
