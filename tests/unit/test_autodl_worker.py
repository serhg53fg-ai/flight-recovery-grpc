import argparse
import socket
from pathlib import Path

import pytest

from deploy.autodl.worker import build_server_argv, launch, preflight


class FakeCuda:
    def __init__(self, available=True, count=1):
        self.available, self.count = available, count
    def is_available(self): return self.available
    def device_count(self): return self.count
    def get_device_name(self, index): return f"Fake GPU {index}"


class FakeTorch:
    __version__ = "test-torch"
    version = type("Version", (), {"cuda": "test-cuda"})()
    def __init__(self, available=True, count=1): self.cuda = FakeCuda(available, count)


def model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (model / name).write_text("{}", encoding="utf-8")
    return model


def args(tmp_path, **changes):
    values = dict(listen="127.0.0.1:50052", worker_id="gpu-0", model_path=model_dir(tmp_path),
                  gpu_index=0, capacity=1, backend="qwen", max_new_tokens=192,
                  max_seconds=30.0, adapter_path=None)
    values.update(changes)
    return argparse.Namespace(**values)


def test_qwen_preflight_returns_bounded_environment(tmp_path):
    result = preflight(args(tmp_path), FakeTorch())
    assert result.gpu_name == "Fake GPU 0"
    assert result.cuda_version == "test-cuda"
    assert result.torch_version == "test-torch"


@pytest.mark.parametrize("change", [
    {"listen": "0.0.0.0:50052"}, {"listen": "10.0.0.2:50052"},
    {"gpu_index": -1}, {"gpu_index": 2}, {"capacity": 0}, {"capacity": 257},
])
def test_rejects_invalid_worker_settings(tmp_path, change):
    with pytest.raises(ValueError):
        preflight(args(tmp_path, **change), FakeTorch())


def test_rejects_unavailable_cuda(tmp_path):
    with pytest.raises(ValueError, match="CUDA"):
        preflight(args(tmp_path), FakeTorch(available=False))


@pytest.mark.parametrize("missing", ["config.json", "tokenizer_config.json", "model.safetensors"])
def test_rejects_incomplete_model_directory(tmp_path, missing):
    values = args(tmp_path)
    (values.model_path / missing).unlink()
    with pytest.raises(ValueError):
        preflight(values, FakeTorch())


def test_builds_approved_server_arguments(tmp_path):
    values = args(tmp_path, backend="test", model_path=None, capacity=2)
    assert build_server_argv(values)[-10:] == (
        "--listen", "127.0.0.1:50052", "--backend", "test", "--worker-id", "gpu-0",
        "--max-inflight", "2", "--max-seconds", "30.0",
    )


def test_rejects_occupied_worker_port(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        with pytest.raises(ValueError, match="occupied"):
            preflight(args(tmp_path, listen=f"127.0.0.1:{port}"), FakeTorch())


def test_sets_gpu_visibility_before_executing_server(tmp_path):
    values = args(tmp_path, gpu_index=1)
    captured = {}

    def execute(executable, argv, environment):
        captured.update(executable=executable, argv=argv, environment=environment)
        raise RuntimeError("exec intercepted")

    with pytest.raises(RuntimeError, match="intercepted"):
        launch(values, torch_module=FakeTorch(count=2), execve=execute)
    assert captured["environment"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert captured["argv"][3:7] == ("--listen", "127.0.0.1:50052", "--backend", "qwen")


def test_propagates_explicit_adapter_path(tmp_path):
    adapter = tmp_path / "adapter"; adapter.mkdir()
    values = args(tmp_path, adapter_path=adapter)
    argv = build_server_argv(values)
    assert argv[argv.index("--adapter-path") + 1] == str(adapter.resolve())


def test_composite_requires_and_propagates_all_model_artifacts(tmp_path):
    flow = tmp_path / 'flow.pkl'; flow.write_bytes(b'flow')
    historical = tmp_path / 'historical.pkl'; historical.write_bytes(b'history')
    values = args(tmp_path, backend='composite', flow_model_path=flow,
                  flow_manifest_hash='manifest-hash', historical_model_path=historical)
    preflight(values, FakeTorch())
    argv = build_server_argv(values)
    assert argv[argv.index('--flow-model-path') + 1] == str(flow.resolve())
    assert argv[argv.index('--flow-manifest-hash') + 1] == 'manifest-hash'
    assert argv[argv.index('--historical-model-path') + 1] == str(historical.resolve())
    flow.unlink()
    with pytest.raises(ValueError, match='artifacts'):
        preflight(values, FakeTorch())
