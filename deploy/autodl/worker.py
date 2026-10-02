"""Preflight and role entrypoint for an AutoDL inference Worker."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import socket
import sys


@dataclass(frozen=True)
class WorkerEnvironment:
    worker_id: str
    gpu_name: str
    cuda_version: str
    torch_version: str
    model_version: str


def _listen(value: str) -> tuple[str, int]:
    host, separator, port_text = value.rpartition(":")
    if not separator or not port_text.isascii() or not port_text.isdigit():
        raise ValueError("listen must be loopback host:port")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    port = int(port_text)
    if not loopback or not 1 <= port <= 65535:
        raise ValueError("listen must use a valid loopback address and port")
    return host, port


def _require_free(host: str, port: int) -> None:
    bind_host = "127.0.0.1" if host == "localhost" else host
    try:
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as probe:
            probe.bind((bind_host, port))
    except OSError as error:
        raise ValueError("worker listen port is occupied") from error


def _validate_model(path: Path) -> Path:
    model = path.expanduser().resolve()
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError("model directory lacks config.json")
    if not any((model / name).is_file() for name in ("tokenizer_config.json", "tokenizer.json", "tokenizer.model")):
        raise ValueError("model directory lacks tokenizer metadata")
    weights = list(model.glob("*.safetensors")) + list(model.glob("pytorch_model*.bin"))
    if not weights or any(not path.is_file() for path in weights):
        raise ValueError("model directory lacks regular weight files")
    return model


def preflight(args, torch_module=None) -> WorkerEnvironment:
    host, port = _listen(args.listen)
    if not args.worker_id or len(args.worker_id) > 128 or args.worker_id.strip() != args.worker_id:
        raise ValueError("worker-id is invalid")
    if type(args.capacity) is not int or not 1 <= args.capacity <= 256:
        raise ValueError("capacity must be 1..256")
    _require_free(host, port)
    if args.backend == "test":
        return WorkerEnvironment(args.worker_id, "TEST", "none", "none", "test-backend")
    historical_mode = args.backend == 'composite' and getattr(args, 'composite_flight_mode', 'qwen') == 'historical'
    model = None if historical_mode else _validate_model(Path(args.model_path))
    adapter_path = getattr(args, "adapter_path", None)
    if adapter_path is not None:
        adapter = Path(adapter_path).expanduser().resolve()
        if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_metadata.json").is_file():
            raise ValueError("adapter directory is incomplete")
    if args.backend == "composite":
        flow_path = Path(args.flow_model_path).expanduser().resolve()
        history_path = Path(args.historical_model_path).expanduser().resolve()
        if not flow_path.is_file() or not history_path.is_file() or not args.flow_manifest_hash:
            raise ValueError("composite model artifacts are incomplete")
    if historical_mode:
        return WorkerEnvironment(args.worker_id, "CPU", "none", "none", history_path.name)
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if type(args.gpu_index) is not int or not 0 <= args.gpu_index < torch_module.cuda.device_count():
        raise ValueError("gpu-index is unavailable")
    return WorkerEnvironment(
        args.worker_id,
        str(torch_module.cuda.get_device_name(args.gpu_index))[:128],
        str(torch_module.version.cuda)[:64],
        str(torch_module.__version__)[:64],
        model.name[:128],
    )


def build_server_argv(args) -> tuple[str, ...]:
    argv = (
        sys.executable, "-m", "services.inference.server",
        "--listen", args.listen, "--backend", args.backend,
        "--worker-id", args.worker_id, "--max-inflight", str(args.capacity),
        "--max-seconds", str(args.max_seconds),
    )
    if args.backend in ("qwen", "composite"):
        historical_mode = args.backend == 'composite' and getattr(args, 'composite_flight_mode', 'qwen') == 'historical'
        if not historical_mode:
            argv += ("--model-path", str(Path(args.model_path).expanduser().resolve()),
                     "--max-new-tokens", str(args.max_new_tokens))
            if getattr(args, "adapter_path", None) is not None:
                argv += ("--adapter-path", str(Path(args.adapter_path).expanduser().resolve()))
            if getattr(args, 'output_mode', None) is not None:
                argv += ('--output-mode', args.output_mode)
        if args.backend == "composite":
            argv += ("--flow-model-path", str(Path(args.flow_model_path).expanduser().resolve()),
                     "--flow-manifest-hash", args.flow_manifest_hash,
                     "--historical-model-path", str(Path(args.historical_model_path).expanduser().resolve()),
                     '--composite-flight-mode', getattr(args, 'composite_flight_mode', 'qwen'))
            if getattr(args, 'expected_model_version', None):
                argv += ('--expected-model-version', args.expected_model_version)
            if getattr(args, 'expected_source', None):
                argv += ('--expected-source', args.expected_source)
    return argv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Start one private AutoDL inference Worker")
    parser.add_argument("--listen", default="127.0.0.1:50052")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--backend", choices=("qwen", "composite", "test"), default="qwen")
    parser.add_argument('--composite-flight-mode', choices=('qwen', 'historical'), default='qwen')
    parser.add_argument('--expected-model-version')
    parser.add_argument('--expected-source', choices=('BASELINE', 'LLM'))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument('--output-mode', choices=['absolute_times', 'duration_components'])
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--flow-model-path", type=Path)
    parser.add_argument("--flow-manifest-hash")
    parser.add_argument("--historical-model-path", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if (args.backend == 'qwen' or (args.backend == 'composite' and args.composite_flight_mode == 'qwen')) and args.model_path is None:
        parser.error("--model-path is required for qwen/composite qwen mode")
    if args.backend == "composite" and (
            args.flow_model_path is None or args.historical_model_path is None or
            not args.flow_manifest_hash):
        parser.error("composite requires flow and historical artifacts")
    return args


def launch(args, torch_module=None, execve=os.execve) -> None:
    environment = preflight(args, torch_module)
    env = os.environ.copy()
    if args.backend == 'qwen' or (args.backend == 'composite' and getattr(args, 'composite_flight_mode', 'qwen') == 'qwen'):
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    print(
        f"worker ready for launch: id={environment.worker_id} gpu={environment.gpu_name} "
        f"cuda={environment.cuda_version} torch={environment.torch_version} "
        f"model={environment.model_version}",
        flush=True,
    )
    execve(sys.executable, build_server_argv(args), env)


def main() -> int:
    args = parse_args()
    try:
        launch(args)
    except ValueError as error:
        print(f"worker preflight failed: {error}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
