"""AutoDL control-instance process orchestration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Callable

from deploy.distributed.release import load_release, release_identity
from deploy.distributed.worker_pool import probe_worker_identity

from .health import wait_for_serving
from .manifest import LocalWorker, SshWorker, load_manifest, to_gateway_config
from .tunnels import autossh_argv, validate_ssh_files


ROOT = Path(__file__).resolve().parents[2]
WORKER_SERVICE = "flight.v1.InferenceWorker"


def load_release_identity(path: Path) -> dict:
    return release_identity(load_release(path))


def _port_free(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "localhost"} else host
    try:
        with socket.socket() as probe:
            probe.bind((bind_host, port))
        return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, timeout: float, stop_event: threading.Event) -> bool:
    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    deadline = time.monotonic() + timeout
    while not stop_event.is_set() and time.monotonic() < deadline:
        try:
            with socket.create_connection((connect_host, port), timeout=min(0.2, timeout)):
                return True
        except OSError:
            stop_event.wait(0.05)
    return False


class ControlPlane:
    def __init__(self, args, *, spawn=None, health_waiter=wait_for_serving,
                 port_waiter=_wait_for_port,
                 release_identity_loader=load_release_identity,
                 identity_prober=probe_worker_identity):
        self.args = args
        self.spawn = spawn or self._spawn
        self.health_waiter = health_waiter
        self.port_waiter = port_waiter
        self.release_identity_loader = release_identity_loader
        self.identity_prober = identity_prober
        self.stop_event = threading.Event()
        self.processes: dict[str, list[tuple[str, subprocess.Popen]]] = {
            "tunnel": [], "worker": [], "gateway": [], "web": []
        }
        self.logs = []
        self.manifest = None
        self.release_identity = None
        self.runtime_root = Path(args.runtime_root).expanduser().resolve()

    def _spawn(self, role, argv, **kwargs):
        return subprocess.Popen(argv, shell=False, **kwargs)

    def _start(self, group: str, name: str, argv: tuple[str, ...] | list[str], env: dict[str, str]):
        logs = self.runtime_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace(":", "-")
        output = (logs / f"{safe_name}.log").open("ab", buffering=0)
        self.logs.append(output)
        process = self.spawn(name, argv, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        self.processes[group].append((name, process))
        return process

    def _require_health(self, process, address: str, service: str, required: bool, label: str) -> bool:
        deadline = time.monotonic() + self.args.startup_timeout_seconds
        serving = False
        while not serving and not self.stop_event.is_set() and time.monotonic() < deadline:
            if process.poll() is not None:
                if required:
                    raise RuntimeError(f"{label} exited during startup")
                return False
            serving = self.health_waiter(
                address, service, min(0.25, deadline - time.monotonic()), self.stop_event
            )
        if process.poll() is not None:
            if required:
                raise RuntimeError(f"{label} exited during startup")
            return False
        if not serving and required:
            raise RuntimeError(f"required {label} failed health check")
        return serving

    def _drop_optional(self, group: str, process) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            self.processes[group] = [
                item for item in self.processes[group] if item[1] is not process
            ]

    def _require_identity(self, process, group: str, worker, address: str) -> bool:
        if self.args.backend != "composite":
            return True
        expected = {key: self.release_identity[key] for key in (
            "source", "model_version", "flow_model_version")}
        try:
            self.identity_prober(
                address, worker.id, expected,
                timeout=max(0.1, min(60.0, self.args.startup_timeout_seconds)),
                fallback_identity=self.release_identity.get("fallback_identity"),
            )
            return True
        except Exception as error:
            if worker.required:
                raise RuntimeError(
                    f"required {worker.id} failed identity probe: {error}") from error
            self._drop_optional(group, process)
            return False

    def validate(self):
        self.manifest = load_manifest(Path(self.args.manifest))
        if self.args.backend == "composite":
            if self.manifest.release_manifest_path is None:
                raise ValueError("composite backend requires release_manifest_path")
            self.release_identity = self.release_identity_loader(
                self.manifest.release_manifest_path)
        binary = Path(self.args.gateway_binary).expanduser().resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("gateway binary must be an executable regular file")
        self.args.gateway_binary = binary
        for worker in self.manifest.workers:
            if isinstance(worker, SshWorker):
                validate_ssh_files(worker)
            elif self.args.backend in {"qwen", "composite"}:
                model_path = worker.model_path or getattr(self.args, "model_path", None)
                if model_path is None or not Path(model_path).expanduser().is_dir():
                    raise ValueError(f"{self.args.backend} backend requires a model directory for {worker.id}")
                if self.args.backend == "composite":
                    paths = (worker.flow_model_path, worker.historical_model_path,
                             worker.feature_contract_path)
                    if any(path is None or not Path(path).expanduser().is_file() for path in paths):
                        raise ValueError(f"composite backend requires flow, historical and contract files for {worker.id}")
                    try:
                        contract = json.loads(Path(worker.feature_contract_path).read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as error:
                        raise ValueError(f"invalid feature contract for {worker.id}") from error
                    if not isinstance(contract, dict) or not contract.get("dataset_manifest_hash"):
                        raise ValueError(f"feature contract lacks dataset_manifest_hash for {worker.id}")
        control = self.manifest.control
        checked = [(control.web_host, control.web_port), (control.gateway_host, control.gateway_port)]
        checked.extend(
            ("127.0.0.1", worker.local_forward_port) if isinstance(worker, SshWorker)
            else tuple(worker.gateway_address.rsplit(":", 1))
            for worker in self.manifest.workers
        )
        if any(not _port_free(host, int(port)) for host, port in checked):
            raise ValueError("a configured local port is already occupied")

    def start(self):
        self.validate()
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "generated"), env.get("PYTHONPATH", "")))
        env["PYTHONUNBUFFERED"] = "1"
        healthy = []
        for worker in self.manifest.workers:
            if not isinstance(worker, SshWorker):
                continue
            process = self._start("tunnel", f"tunnel:{worker.id}", autossh_argv(worker), env)
            address = f"127.0.0.1:{worker.local_forward_port}"
            if self._require_health(process, address, WORKER_SERVICE, worker.required, worker.id):
                if self._require_identity(process, "tunnel", worker, address):
                    healthy.append(worker)
            else:
                self._drop_optional("tunnel", process)
        for worker in self.manifest.workers:
            if not isinstance(worker, LocalWorker):
                continue
            argv = [sys.executable, "-m", "deploy.autodl.worker", "--listen", worker.gateway_address,
                    "--backend", self.args.backend, "--worker-id", worker.id,
                    "--gpu-index", str(worker.gpu_index), "--capacity", str(worker.capacity)]
            if self.args.backend in {"qwen", "composite"}:
                model_path = worker.model_path or getattr(self.args, "model_path", None)
                argv.extend(("--model-path", str(Path(model_path).expanduser().resolve())))
                if getattr(self.args, 'output_mode', None) is not None:
                    argv.extend(('--output-mode', self.args.output_mode))
                adapter_path = worker.adapter_path or getattr(self.args, "adapter_path", None)
                if adapter_path is not None:
                    argv.extend(("--adapter-path", str(Path(adapter_path).expanduser().resolve())))
                if self.args.backend == "composite":
                    contract = json.loads(worker.feature_contract_path.read_text(encoding="utf-8"))
                    argv.extend((
                        "--flow-model-path", str(worker.flow_model_path.resolve()),
                        "--flow-manifest-hash", str(contract["dataset_manifest_hash"]),
                        "--historical-model-path", str(worker.historical_model_path.resolve()),
                        "--expected-model-version", self.release_identity["model_version"],
                        "--expected-source", self.release_identity["source"],
                    ))
            process = self._start("worker", f"worker:{worker.id}", argv, env)
            if self._require_health(process, worker.gateway_address, WORKER_SERVICE, worker.required, worker.id):
                if self._require_identity(process, "worker", worker, worker.gateway_address):
                    healthy.append(worker)
            else:
                self._drop_optional("worker", process)
        if not healthy:
            raise RuntimeError("no healthy workers are available")
        healthy_ids = {worker.id for worker in healthy}
        active_manifest = type(self.manifest)(
            release_manifest_path=self.manifest.release_manifest_path,
            control=self.manifest.control,
            workers=tuple(worker for worker in self.manifest.workers if worker.id in healthy_ids),
        )
        config_path = self.runtime_root / "gateway.generated.json"
        config_path.write_text(json.dumps(to_gateway_config(active_manifest), indent=2) + "\n", encoding="utf-8")
        gateway_address = f"{self.manifest.control.gateway_host}:{self.manifest.control.gateway_port}"
        gateway = self._start("gateway", "gateway", [str(self.args.gateway_binary), "--config", str(config_path)], env)
        if not self._require_health(gateway, gateway_address, "", True, "gateway"):
            raise RuntimeError("gateway failed health check")
        env.update(
            FLIGHT_RUNTIME_ROOT=str(self.runtime_root),
            FLIGHT_GATEWAY_ADDRESS=gateway_address,
            FLIGHT_RPC_TIMEOUT_SECONDS=str(self.manifest.control.rpc_timeout_ms / 1000),
        )
        web = self._start("web", "web", [sys.executable, "-m", "apps.flight.run_app", "--host",
                          self.manifest.control.web_host, "--port", str(self.manifest.control.web_port)], env)
        if web.poll() is not None:
            raise RuntimeError("web exited during startup")
        if not self.port_waiter(
            self.manifest.control.web_host, self.manifest.control.web_port,
            self.args.startup_timeout_seconds, self.stop_event,
        ):
            raise RuntimeError("web failed readiness check")

    def stop(self):
        for group in ("web", "gateway", "tunnel", "worker"):
            for _, process in reversed(self.processes[group]):
                if process.poll() is None:
                    process.terminate()
        deadline = time.monotonic() + 5
        for group in ("web", "gateway", "tunnel", "worker"):
            for _, process in reversed(self.processes[group]):
                if process.poll() is not None:
                    continue
                try:
                    process.wait(timeout=max(0.05, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for output in self.logs:
            output.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the AutoDL flight control plane")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--gateway-binary", type=Path, required=True)
    parser.add_argument("--backend", choices=("test", "qwen", "composite"), default="qwen")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument('--output-mode', choices=['absolute_times', 'duration_components'])
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--startup-timeout-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    if not 0 < args.startup_timeout_seconds <= 3600:
        parser.error("startup timeout must be in (0, 3600]")
    return args


def main() -> int:
    plane = ControlPlane(parse_args())
    try:
        signal.signal(signal.SIGINT, lambda *_: plane.stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: plane.stop_event.set())
        plane.start()
        print("AutoDL control plane is ready", flush=True)
        while not plane.stop_event.wait(0.2):
            if any(process.poll() is not None for group in plane.processes.values() for _, process in group):
                raise RuntimeError("a managed process exited unexpectedly")
        return 0
    except (ValueError, RuntimeError) as error:
        print(f"control plane failed: {error}", file=sys.stderr)
        return 1
    finally:
        plane.stop()


if __name__ == "__main__":
    raise SystemExit(main())
