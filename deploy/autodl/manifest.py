"""Strict parser for the committed AutoDL cluster-manifest format."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class ControlConfig:
    web_host: str
    web_port: int
    gateway_host: str
    gateway_port: int
    rpc_timeout_ms: int


@dataclass(frozen=True)
class LocalWorker:
    id: str
    gateway_address: str
    gpu_index: int
    capacity: int
    required: bool
    model_path: Path | None = None
    adapter_path: Path | None = None
    flow_model_path: Path | None = None
    historical_model_path: Path | None = None
    feature_contract_path: Path | None = None
    mode: str = "local"


@dataclass(frozen=True)
class SshWorker:
    id: str
    ssh_host: str
    ssh_port: int
    ssh_user: str
    ssh_key_file: Path
    known_hosts_file: Path
    remote_worker_host: str
    remote_worker_port: int
    local_forward_port: int
    capacity: int
    required: bool
    model_path: Path | None = None
    adapter_path: Path | None = None
    flow_model_path: Path | None = None
    historical_model_path: Path | None = None
    feature_contract_path: Path | None = None
    mode: str = "ssh"


Worker = Union[LocalWorker, SshWorker]


@dataclass(frozen=True)
class ClusterManifest:
    release_manifest_path: Path | None
    control: ControlConfig
    workers: tuple[Worker, ...]


_CONTROL_FIELDS = {"web_host", "web_port", "gateway_host", "gateway_port", "rpc_timeout_ms"}
_ARTIFACT_FIELDS = {
    "model_path", "adapter_path", "flow_model_path", "historical_model_path",
    "feature_contract_path",
}
_LOCAL_FIELDS = ({"id", "mode", "gateway_address", "gpu_index", "capacity", "required"}
                 | _ARTIFACT_FIELDS)
_SSH_FIELDS = {
    "id", "mode", "ssh_host", "ssh_port", "ssh_user", "ssh_key_file",
    "known_hosts_file", "remote_worker_host", "remote_worker_port",
    "local_forward_port", "capacity", "required",
    *_ARTIFACT_FIELDS,
}


def _object(value: object, scope: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{scope} must be an object")
    return value


def _exact(value: dict[str, object], expected: set[str], scope: str) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        raise ValueError(f"{scope} fields are invalid")


def _string(value: object, scope: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or value.strip() != value:
        raise ValueError(f"{scope} must be a non-empty trimmed string")
    return value


def _integer(value: object, scope: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{scope} is outside its integer range")
    return value


def _boolean(value: object, scope: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{scope} must be boolean")
    return value


def _loopback_host(value: object, scope: str) -> str:
    host = _string(value, scope)
    if host == "localhost":
        return host
    try:
        if ipaddress.ip_address(host).is_loopback:
            return host
    except ValueError:
        pass
    raise ValueError(f"{scope} must be a loopback address")


def _endpoint(value: object, scope: str) -> tuple[str, int]:
    endpoint = _string(value, scope)
    host, separator, port_text = endpoint.rpartition(":")
    if not separator or not port_text.isascii() or not port_text.isdigit():
        raise ValueError(f"{scope} must be host:port")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    _loopback_host(host, scope)
    return endpoint, _integer(int(port_text), scope, 1, 65535)


def _absolute_path(value: object, scope: str) -> Path:
    path = Path(_string(value, scope))
    if not path.is_absolute():
        raise ValueError(f"{scope} must be absolute")
    return path


def _optional_absolute_path(value: object, scope: str) -> Path | None:
    return None if value is None else _absolute_path(value, scope)


def _parse_control(value: object) -> ControlConfig:
    raw = _object(value, "control")
    _exact(raw, _CONTROL_FIELDS, "control")
    web_host = _string(raw["web_host"], "control.web_host")
    if web_host not in {"0.0.0.0", "127.0.0.1", "localhost"}:
        raise ValueError("control.web_host is unsupported")
    return ControlConfig(
        web_host=web_host,
        web_port=_integer(raw["web_port"], "control.web_port", 1, 65535),
        gateway_host=_loopback_host(raw["gateway_host"], "control.gateway_host"),
        gateway_port=_integer(raw["gateway_port"], "control.gateway_port", 1, 65535),
        rpc_timeout_ms=_integer(raw["rpc_timeout_ms"], "control.rpc_timeout_ms", 1, 600000),
    )


def _parse_worker(value: object, index: int) -> Worker:
    scope = f"workers[{index}]"
    raw = _object(value, scope)
    mode = raw.get("mode")
    fields = _LOCAL_FIELDS if mode == "local" else _SSH_FIELDS if mode == "ssh" else None
    if fields is None:
        raise ValueError(f"{scope}.mode is invalid")
    _exact(raw, fields, scope)
    common = {
        "id": _string(raw["id"], f"{scope}.id", 128),
        "capacity": _integer(raw["capacity"], f"{scope}.capacity", 1, 256),
        "required": _boolean(raw["required"], f"{scope}.required"),
        "model_path": _optional_absolute_path(raw["model_path"], f"{scope}.model_path"),
        "adapter_path": _optional_absolute_path(raw["adapter_path"], f"{scope}.adapter_path"),
        "flow_model_path": _optional_absolute_path(raw["flow_model_path"], f"{scope}.flow_model_path"),
        "historical_model_path": _optional_absolute_path(
            raw["historical_model_path"], f"{scope}.historical_model_path"),
        "feature_contract_path": _optional_absolute_path(
            raw["feature_contract_path"], f"{scope}.feature_contract_path"),
    }
    if mode == "local":
        address, _ = _endpoint(raw["gateway_address"], f"{scope}.gateway_address")
        return LocalWorker(
            gateway_address=address,
            gpu_index=_integer(raw["gpu_index"], f"{scope}.gpu_index", 0, 255),
            **common,
        )
    return SshWorker(
        ssh_host=_string(raw["ssh_host"], f"{scope}.ssh_host"),
        ssh_port=_integer(raw["ssh_port"], f"{scope}.ssh_port", 1, 65535),
        ssh_user=_string(raw["ssh_user"], f"{scope}.ssh_user", 128),
        ssh_key_file=_absolute_path(raw["ssh_key_file"], f"{scope}.ssh_key_file"),
        known_hosts_file=_absolute_path(raw["known_hosts_file"], f"{scope}.known_hosts_file"),
        remote_worker_host=_loopback_host(raw["remote_worker_host"], f"{scope}.remote_worker_host"),
        remote_worker_port=_integer(raw["remote_worker_port"], f"{scope}.remote_worker_port", 1, 65535),
        local_forward_port=_integer(raw["local_forward_port"], f"{scope}.local_forward_port", 1, 65535),
        **common,
    )


def load_manifest(path: Path) -> ClusterManifest:
    try:
        root = _object(json.loads(path.read_text(encoding="utf-8")), "root")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load cluster manifest: {error}") from error
    _exact(root, {"release_manifest_path", "control", "workers"}, "root")
    release_manifest_path = _optional_absolute_path(
        root["release_manifest_path"], "release_manifest_path")
    control = _parse_control(root["control"])
    raw_workers = root["workers"]
    if not isinstance(raw_workers, list) or not 1 <= len(raw_workers) <= 256:
        raise ValueError("workers must contain 1..256 entries")
    workers = tuple(_parse_worker(value, index) for index, value in enumerate(raw_workers))
    if not any(worker.required for worker in workers):
        raise ValueError("at least one worker must be required")
    ids = [worker.id for worker in workers]
    if len(ids) != len(set(ids)):
        raise ValueError("worker ids must be unique")
    gpu_indexes = [worker.gpu_index for worker in workers if isinstance(worker, LocalWorker)]
    if len(gpu_indexes) != len(set(gpu_indexes)):
        raise ValueError("local worker gpu_index values must be unique")
    reserved_ports = {control.web_port, control.gateway_port}
    local_ports = []
    for worker in workers:
        if isinstance(worker, LocalWorker):
            _, port = _endpoint(worker.gateway_address, "local worker address")
            local_ports.append(port)
        else:
            local_ports.append(worker.local_forward_port)
    if len(local_ports) != len(set(local_ports)) or reserved_ports.intersection(local_ports):
        raise ValueError("worker ports must be unique and not conflict with control ports")
    return ClusterManifest(release_manifest_path=release_manifest_path,
                           control=control, workers=workers)


def to_gateway_config(manifest: ClusterManifest) -> dict[str, object]:
    workers = []
    for worker in manifest.workers:
        address = worker.gateway_address if isinstance(worker, LocalWorker) else f"127.0.0.1:{worker.local_forward_port}"
        workers.append({"id": worker.id, "address": address, "capacity": worker.capacity, "enabled": True})
    return {
        "listen": f"{manifest.control.gateway_host}:{manifest.control.gateway_port}",
        "rpc_timeout_ms": manifest.control.rpc_timeout_ms,
        "health_interval_ms": 2000,
        "health_timeout_ms": 500,
        "failure_threshold": 3,
        "open_cooldown_ms": 10000,
        "minimum_retry_budget_ms": 500,
        "workers": workers,
    }
