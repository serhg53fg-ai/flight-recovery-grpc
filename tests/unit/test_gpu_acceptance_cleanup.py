from types import SimpleNamespace
import pytest


def test_gateway_start_failure_stops_allocated_gpu_worker(tmp_path, monkeypatch):
    from tests.integration import test_gpu_business_chain as acceptance
    events = []
    server = SimpleNamespace(stop=lambda grace: SimpleNamespace(wait=lambda: events.append('stop')))
    executor = SimpleNamespace(shutdown=lambda **kwargs: events.append('shutdown'))
    monkeypatch.setenv('FLIGHT_GPU_MODEL_PATH', str(tmp_path))
    monkeypatch.setattr(acceptance, 'OUT', tmp_path)
    monkeypatch.setattr(acceptance, 'QwenBackend', lambda *args, **kwargs: SimpleNamespace(predict=lambda *a: None, model_version='unit'))
    monkeypatch.setattr(acceptance, 'free_address', lambda: '127.0.0.1:12345')
    monkeypatch.setattr(acceptance, 'serve', lambda *args: (server, executor))
    def unavailable(*args, **kwargs):
        raise FileNotFoundError('gateway unavailable')
    monkeypatch.setattr(acceptance.subprocess, 'Popen', unavailable)
    fixture = acceptance.gpu_gateway.__wrapped__()
    with pytest.raises(FileNotFoundError):
        next(fixture)
    assert events == ['stop', 'shutdown']
