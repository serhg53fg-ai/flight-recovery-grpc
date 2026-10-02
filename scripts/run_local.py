#!/usr/bin/env python3
"""Launch the local Web -> C++ gateway -> Python worker prediction chain."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-worker flight prediction chain")
    parser.add_argument("--backend", choices=("qwen", "test"), default="qwen")
    parser.add_argument("--model-path")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=5000)
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=50051)
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-port", type=int, default=50052)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--runtime-root", type=Path, default=ROOT / "runtime")
    parser.add_argument(
        "--gateway-binary",
        type=Path,
        default=ROOT / "build" / "phase2" / "flight_gateway",
    )
    parser.add_argument("--rpc-timeout-seconds", type=float, default=20)
    parser.add_argument("--gateway-max-inflight", type=int, default=4)
    parser.add_argument("--worker-max-inflight", type=int, default=1)
    parser.add_argument("--batch-workers", type=int, default=1)
    parser.add_argument("--input-timezone", default="Asia/Shanghai")
    parser.add_argument("--worker-id", default="local-worker")
    parser.add_argument("--startup-timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    for name in ("web_port", "gateway_port", "worker_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"{name.replace('_', '-')} must be 1..65535")
    if args.rpc_timeout_seconds <= 0:
        parser.error("rpc-timeout-seconds must be positive")
    if not 1 <= args.startup_timeout_seconds <= 3600:
        parser.error("startup-timeout-seconds must be 1..3600")
    if not 1 <= args.gateway_max_inflight <= 256:
        parser.error("gateway-max-inflight must be 1..256")
    if not 1 <= args.worker_max_inflight <= 256:
        parser.error("worker-max-inflight must be 1..256")
    if not 1 <= args.worker_count <= 256:
        parser.error("worker-count must be 1..256")
    if args.worker_port + args.worker_count - 1 > 65535:
        parser.error("worker port range exceeds 65535")
    if not 1 <= args.batch_workers <= args.gateway_max_inflight:
        parser.error("batch-workers must be 1..gateway-max-inflight")
    if args.backend == "qwen":
        if args.worker_count != 1:
            parser.error("qwen launcher supports one worker; use an explicit gateway config for multi-GPU")
        if not args.model_path:
            parser.error("--model-path is required for the qwen backend")
        model_path = Path(args.model_path).expanduser().resolve()
        if not model_path.is_dir():
            parser.error("--model-path must be an existing local directory")
        args.model_path = str(model_path)
    return args


def terminate_children(children: list[subprocess.Popen[bytes]], grace_seconds: float = 5) -> None:
    for child in reversed(children):
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + grace_seconds
    for child in reversed(children):
        if child.poll() is not None:
            continue
        try:
            child.wait(timeout=max(0.05, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def port_is_open(host: str, port: int) -> bool:
    connect_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        with socket.create_connection((connect_host, port), timeout=0.2):
            return True
    except OSError:
        return False


def grpc_is_serving(address: str, service: str = "") -> bool:
    channel = grpc.insecure_channel(address)
    try:
        response = health_pb2_grpc.HealthStub(channel).Check(
            health_pb2.HealthCheckRequest(service=service), timeout=0.3
        )
        return response.status == health_pb2.HealthCheckResponse.SERVING
    except grpc.RpcError:
        return False
    finally:
        channel.close()


def main() -> int:
    args = parse_args()
    gateway_binary = args.gateway_binary.expanduser().resolve()
    if not gateway_binary.is_file() or not os.access(gateway_binary, os.X_OK):
        print(
            f"Gateway binary is missing or not executable: {gateway_binary}\n"
            "Build it with: cmake -S gateway -B build/phase2 && "
            "cmake --build build/phase2 --parallel 2",
            file=sys.stderr,
        )
        return 2

    runtime_root = args.runtime_root.expanduser().resolve()
    logs_dir = runtime_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    worker_addresses = [f"{args.worker_host}:{args.worker_port + index}"
                        for index in range(args.worker_count)]
    gateway_address = f"{args.gateway_host}:{args.gateway_port}"
    # Keep the virtual-environment entry point. Resolving this symlink would
    # execute the system interpreter and lose the venv's installed packages.
    python = Path(sys.executable)
    env = os.environ.copy()
    python_path = [str(ROOT), str(ROOT / "generated")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env.update(
        PYTHONPATH=os.pathsep.join(python_path),
        PYTHONUNBUFFERED="1",
        FLIGHT_RUNTIME_ROOT=str(runtime_root),
        FLIGHT_GATEWAY_ADDRESS=gateway_address,
        FLIGHT_RPC_TIMEOUT_SECONDS=str(args.rpc_timeout_seconds),
        FLIGHT_BATCH_WORKERS=str(args.batch_workers),
        FLIGHT_INPUT_TIMEZONE=args.input_timezone,
    )

    worker_commands = []
    for index, worker_address in enumerate(worker_addresses):
        worker_id = args.worker_id if args.worker_count == 1 else f"{args.worker_id}-{index}"
        command = [
            str(python), "-m", "services.inference.server", "--listen", worker_address,
            "--backend", args.backend, "--worker-id", worker_id,
            "--max-inflight", str(args.worker_max_inflight),
            "--max-seconds", str(args.rpc_timeout_seconds),
        ]
        if args.backend == "qwen":
            command.extend(("--model-path", args.model_path))
        worker_commands.append(command)
    gateway_config_path = runtime_root / "gateway.generated.json"
    gateway_config_path.write_text(json.dumps({
        "listen": gateway_address,
        "rpc_timeout_ms": round(args.rpc_timeout_seconds * 1000),
        "health_interval_ms": 500,
        "health_timeout_ms": 200,
        "failure_threshold": 3,
        "open_cooldown_ms": 10000,
        "minimum_retry_budget_ms": min(500, max(1, round(args.rpc_timeout_seconds * 500))),
        "workers": [
            {"id": args.worker_id if args.worker_count == 1 else f"{args.worker_id}-{index}",
             "address": address, "capacity": args.worker_max_inflight, "enabled": True}
            for index, address in enumerate(worker_addresses)
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gateway_command = [
        str(gateway_binary), "--config", str(gateway_config_path),
    ]
    web_command = [
        str(python), "-m", "apps.flight.run_app",
        "--host", args.web_host,
        "--port", str(args.web_port),
    ]

    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    children: list[subprocess.Popen[bytes]] = []
    log_files = []
    try:
        process_specs = [
            (f"worker-{index}", command, address, "flight.v1.InferenceWorker")
            for index, (command, address) in enumerate(zip(worker_commands, worker_addresses))
        ]
        process_specs.append(("gateway", gateway_command, gateway_address, ""))
        for name, command, address, health_service in process_specs:
            log_file = (logs_dir / f"{name}.log").open("ab", buffering=0)
            log_files.append(log_file)
            children.append(
                subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            )
            deadline = time.monotonic() + args.startup_timeout_seconds
            while not stopping and not grpc_is_serving(address, health_service):
                if children[-1].poll() is not None:
                    print(f"{name} exited during startup; see {logs_dir / (name + '.log')}", file=sys.stderr)
                    return 1
                if time.monotonic() >= deadline:
                    print(f"{name} startup timed out; see {logs_dir / (name + '.log')}", file=sys.stderr)
                    return 1
                time.sleep(0.1)
            if stopping:
                return 0
        log_file = (logs_dir / "web.log").open("ab", buffering=0)
        log_files.append(log_file)
        children.append(subprocess.Popen(web_command, cwd=ROOT, env=env,
                                         stdout=log_file, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + args.startup_timeout_seconds
        while not stopping and not port_is_open(args.web_host, args.web_port):
            if children[-1].poll() is not None or time.monotonic() >= deadline:
                print(f"web startup failed; see {logs_dir / 'web.log'}", file=sys.stderr)
                return 1
            time.sleep(0.1)
        backend_label = "TEST" if args.backend == "test" else "LLM/Qwen"
        print(
            f"Flight prediction chain ready at http://{args.web_host}:{args.web_port} "
            f"(backend={backend_label}, runtime={runtime_root})",
            flush=True,
        )
        while not stopping:
            names = [f"worker-{index}" for index in range(args.worker_count)] + ["gateway", "web"]
            for name, child in zip(names, children):
                if child.poll() is not None:
                    print(f"{name} exited unexpectedly with code {child.returncode}", file=sys.stderr)
                    return 1
            time.sleep(0.1)
        return 0
    finally:
        terminate_children(children)
        for log_file in log_files:
            log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
