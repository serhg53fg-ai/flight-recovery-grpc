import argparse
import json
import os
from pathlib import Path

import pytest

from deploy.autodl.control import ControlPlane

from tests.unit.test_autodl_manifest import valid_manifest, write_manifest


class FakeProcess:
    next_pid = 100

    def __init__(self, argv, events, role, **_kwargs):
        self.argv = tuple(argv)
        self.events = events
        self.role = role
        self.returncode = None
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        events.append(f"start:{role}")

    def poll(self):
        return self.returncode

    def terminate(self):
        self.events.append(f"stop:{self.role}")
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_control_plane_starts_health_gated_roles_and_stops_in_owned_order(tmp_path, monkeypatch):
    monkeypatch.setattr("deploy.autodl.control._port_free", lambda *_: True)
    value = valid_manifest()
    manifest_path = write_manifest(tmp_path, value)
    key = Path(value["workers"][1]["ssh_key_file"])
    hosts = Path(value["workers"][1]["known_hosts_file"])
    # Replace example root paths with isolated test files.
    key = tmp_path / "id_cluster"
    hosts = tmp_path / "known_hosts"
    key.write_text("test-only", encoding="utf-8")
    hosts.write_text("fingerprint", encoding="utf-8")
    os.chmod(key, 0o600)
    os.chmod(hosts, 0o644)
    value["workers"][1]["ssh_key_file"] = str(key)
    value["workers"][1]["known_hosts_file"] = str(hosts)
    manifest_path = write_manifest(tmp_path, value)
    gateway = tmp_path / "flight_gateway"
    gateway.write_text("binary", encoding="utf-8")
    os.chmod(gateway, 0o700)

    events = []

    def spawn(role, argv, **kwargs):
        return FakeProcess(argv, events, role, **kwargs)

    def serving(address, service, timeout, stop):
        events.append(f"health:{service or 'gateway'}@{address}")
        return True

    args = argparse.Namespace(
        manifest=manifest_path, runtime_root=tmp_path / "runtime",
        gateway_binary=gateway, backend="test", model_path=None,
        startup_timeout_seconds=1.0,
    )
    plane = ControlPlane(args, spawn=spawn, health_waiter=serving, port_waiter=lambda *_: True)
    plane.start()
    generated = json.loads((args.runtime_root / "gateway.generated.json").read_text())
    assert [worker["address"] for worker in generated["workers"]] == [
        "127.0.0.1:50052", "127.0.0.1:15053"
    ]
    assert events[:6] == [
        "start:tunnel:autodl-b-gpu0",
        "health:flight.v1.InferenceWorker@127.0.0.1:15053",
        "start:worker:autodl-a-gpu0",
        "health:flight.v1.InferenceWorker@127.0.0.1:50052",
        "start:gateway",
        "health:gateway@127.0.0.1:50051",
    ]
    assert events[6] == "start:web"
    plane.stop()
    assert [event for event in events if event.startswith("stop:")] == [
        "stop:web", "stop:gateway", "stop:tunnel:autodl-b-gpu0", "stop:worker:autodl-a-gpu0"
    ]


def make_plane(tmp_path, monkeypatch, health, *, optional_remote=False, fail_role=None):
    monkeypatch.setattr("deploy.autodl.control._port_free", lambda *_: True)
    value = valid_manifest()
    value["workers"][1]["required"] = not optional_remote
    key, hosts = tmp_path / "key", tmp_path / "hosts"
    key.write_text("test", encoding="utf-8")
    hosts.write_text("fingerprint", encoding="utf-8")
    os.chmod(key, 0o600)
    os.chmod(hosts, 0o644)
    value["workers"][1]["ssh_key_file"] = str(key)
    value["workers"][1]["known_hosts_file"] = str(hosts)
    gateway = tmp_path / "gateway"
    gateway.write_text("binary", encoding="utf-8")
    os.chmod(gateway, 0o700)
    events = []

    def spawn(role, argv, **kwargs):
        process = FakeProcess(argv, events, role, **kwargs)
        if role == fail_role:
            process.returncode = 7
        return process

    args = argparse.Namespace(
        manifest=write_manifest(tmp_path, value), runtime_root=tmp_path / "run",
        gateway_binary=gateway, backend="test", model_path=None,
        startup_timeout_seconds=0.1,
    )
    return ControlPlane(
        args, spawn=spawn, health_waiter=health, port_waiter=lambda *_: True
    ), events


def test_required_worker_health_failure_aborts_before_gateway(tmp_path, monkeypatch):
    plane, events = make_plane(tmp_path, monkeypatch, lambda *_: False)
    try:
        with pytest.raises(RuntimeError, match="required"):
            plane.start()
        assert "start:gateway" not in events
    finally:
        plane.stop()


def test_optional_remote_failure_degrades_to_required_local_worker(tmp_path, monkeypatch):
    def health(address, *_):
        return address != "127.0.0.1:15053"

    plane, events = make_plane(tmp_path, monkeypatch, health, optional_remote=True)
    try:
        plane.start()
        generated = json.loads((plane.runtime_root / "gateway.generated.json").read_text())
        assert [worker["id"] for worker in generated["workers"]] == ["autodl-a-gpu0"]
        assert "start:web" in events
    finally:
        plane.stop()


def test_tunnel_early_exit_aborts_before_health_and_gateway(tmp_path, monkeypatch):
    plane, events = make_plane(tmp_path, monkeypatch, lambda *_: True, fail_role="tunnel:autodl-b-gpu0")
    try:
        with pytest.raises(RuntimeError, match="exited"):
            plane.start()
        assert not any(event.startswith("health:") for event in events)
        assert "start:gateway" not in events
    finally:
        plane.stop()


def test_gateway_early_exit_aborts_before_web(tmp_path, monkeypatch):
    plane, events = make_plane(tmp_path, monkeypatch, lambda *_: True, fail_role="gateway")
    try:
        with pytest.raises(RuntimeError, match="exited"):
            plane.start()
        assert "start:web" not in events
    finally:
        plane.stop()


def test_occupied_port_is_rejected_before_any_process_starts(tmp_path, monkeypatch):
    plane, events = make_plane(tmp_path, monkeypatch, lambda *_: True)
    monkeypatch.setattr("deploy.autodl.control._port_free", lambda *_: False)
    with pytest.raises(ValueError, match="occupied"):
        plane.start()
    assert events == []


def test_optional_exited_tunnel_is_removed_from_managed_processes(tmp_path, monkeypatch):
    plane, events = make_plane(
        tmp_path, monkeypatch, lambda *_: True,
        optional_remote=True, fail_role="tunnel:autodl-b-gpu0",
    )
    try:
        plane.start()
        assert plane.processes["tunnel"] == []
        assert all(process.poll() is None for group in plane.processes.values() for _, process in group)
    finally:
        plane.stop()


def test_health_wait_detects_process_exit_without_full_timeout(tmp_path, monkeypatch):
    import threading
    import time

    plane, _ = make_plane(tmp_path, monkeypatch, lambda address, service, timeout, stop: (time.sleep(timeout), False)[1])
    process = FakeProcess(("worker",), [], "worker")
    timer = threading.Timer(0.05, lambda: setattr(process, "returncode", 9))
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="exited"):
            plane._require_health(process, "127.0.0.1:50052", "service", True, "worker")
        assert time.monotonic() - started < 0.5
    finally:
        timer.cancel()


def test_local_workers_launch_through_gpu_preflight_entrypoint(tmp_path, monkeypatch):
    plane, events = make_plane(tmp_path, monkeypatch, lambda *_: True)
    try:
        plane.start()
        _, process = plane.processes["worker"][0]
        assert process.argv[1:3] == ("-m", "deploy.autodl.worker")
        assert "--gpu-index" in process.argv
    finally:
        plane.stop()


def test_composite_local_worker_uses_manifest_artifact_contract(tmp_path, monkeypatch):
    plane, _ = make_plane(tmp_path, monkeypatch, lambda *_: True)
    model = tmp_path / "qwen"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    flow = tmp_path / "flow.pkl"
    historical = tmp_path / "historical.pkl"
    contract = tmp_path / "flow-metrics.json"
    flow.write_bytes(b"flow")
    historical.write_bytes(b"history")
    contract.write_text(json.dumps({"dataset_manifest_hash": "dataset-hash-v1"}), encoding="utf-8")
    value = valid_manifest()
    value["release_manifest_path"] = str(tmp_path / "candidate.json")
    value["workers"] = [value["workers"][0]]
    value["workers"][0].update(
        model_path=str(model), adapter_path=str(adapter), flow_model_path=str(flow),
        historical_model_path=str(historical), feature_contract_path=str(contract),
    )
    plane.args.manifest = write_manifest(tmp_path, value)
    plane.args.backend = "composite"
    expected = {"source": "LLM", "model_version": "qwen3+adapter-v1+flow-v1",
                "flow_model_version": "flow-v1"}
    probes = []
    plane.release_identity_loader = lambda path: expected
    plane.identity_prober = lambda address, worker_id, identity, **kwargs: probes.append(
        (address, worker_id, identity, kwargs)) or {"source": "LLM"}
    try:
        plane.start()
        argv = plane.processes["worker"][0][1].argv
        assert argv[argv.index("--model-path") + 1] == str(model)
        assert argv[argv.index("--flow-model-path") + 1] == str(flow)
        assert argv[argv.index("--historical-model-path") + 1] == str(historical)
        assert argv[argv.index("--flow-manifest-hash") + 1] == "dataset-hash-v1"
        assert argv[argv.index("--expected-model-version") + 1] == expected["model_version"]
        assert argv[argv.index("--expected-source") + 1] == "LLM"
        assert probes == [("127.0.0.1:50052", "autodl-a-gpu0", expected,
                           {"timeout": 0.1, "fallback_identity": None})]
    finally:
        plane.stop()


def test_required_remote_identity_mismatch_aborts_before_local_worker_and_gateway(tmp_path, monkeypatch):
    plane, events = make_plane(tmp_path, monkeypatch, lambda *_: True)
    value = json.loads(Path(plane.args.manifest).read_text())
    value["release_manifest_path"] = str(tmp_path / "candidate.json")
    model = tmp_path / "qwen"; model.mkdir()
    flow = tmp_path / "flow.pkl"; flow.write_bytes(b"flow")
    historical = tmp_path / "historical.pkl"; historical.write_bytes(b"history")
    contract = tmp_path / "metrics.json"
    contract.write_text(json.dumps({"dataset_manifest_hash": "dataset-hash-v1"}))
    value["workers"][0].update(
        model_path=str(model), adapter_path=None, flow_model_path=str(flow),
        historical_model_path=str(historical), feature_contract_path=str(contract))
    plane.args.manifest = write_manifest(tmp_path, value)
    plane.args.backend = "composite"
    plane.release_identity_loader = lambda path: {
        "source": "LLM", "model_version": "qwen3+adapter-v1+flow-v1",
        "flow_model_version": "flow-v1"}
    plane.identity_prober = lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("Worker autodl-b-gpu0 model_version mismatch"))

    try:
        with pytest.raises(RuntimeError, match="identity"):
            plane.start()
        assert "start:worker:autodl-a-gpu0" not in events
        assert "start:gateway" not in events
    finally:
        plane.stop()


def test_composite_requires_release_manifest_before_starting_processes(tmp_path, monkeypatch):
    plane, events = make_plane(tmp_path, monkeypatch, lambda *_: True)
    plane.args.backend = "composite"
    with pytest.raises(ValueError, match="release_manifest_path"):
        plane.start()
    assert events == []


def test_optional_remote_identity_mismatch_is_removed_from_gateway_pool(tmp_path, monkeypatch):
    plane, events = make_plane(
        tmp_path, monkeypatch, lambda *_: True, optional_remote=True)
    value = json.loads(Path(plane.args.manifest).read_text())
    value["release_manifest_path"] = str(tmp_path / "candidate.json")
    model = tmp_path / "qwen"; model.mkdir()
    flow = tmp_path / "flow.pkl"; flow.write_bytes(b"flow")
    historical = tmp_path / "historical.pkl"; historical.write_bytes(b"history")
    contract = tmp_path / "metrics.json"
    contract.write_text(json.dumps({"dataset_manifest_hash": "dataset-hash-v1"}))
    value["workers"][0].update(
        model_path=str(model), adapter_path=None, flow_model_path=str(flow),
        historical_model_path=str(historical), feature_contract_path=str(contract))
    plane.args.manifest = write_manifest(tmp_path, value)
    plane.args.backend = "composite"
    expected = {"source": "LLM", "model_version": "qwen3+adapter-v1+flow-v1",
                "flow_model_version": "flow-v1"}
    plane.release_identity_loader = lambda path: expected

    def probe(_address, worker_id, *_args, **_kwargs):
        if worker_id == "autodl-b-gpu0":
            raise ValueError("model_version mismatch")
        return expected

    plane.identity_prober = probe
    try:
        plane.start()
        generated = json.loads((plane.runtime_root / "gateway.generated.json").read_text())
        assert [worker["id"] for worker in generated["workers"]] == ["autodl-a-gpu0"]
        assert plane.processes["tunnel"] == []
        assert "start:web" in events
    finally:
        plane.stop()


def test_optional_process_ignoring_terminate_is_killed_and_removed(tmp_path, monkeypatch):
    class StubbornProcess(FakeProcess):
        def terminate(self):
            self.events.append(f"stop:{self.role}")
        def wait(self, timeout=None):
            if self.returncode is None:
                raise __import__("subprocess").TimeoutExpired(self.argv, timeout)
            return self.returncode
        def kill(self):
            self.events.append(f"kill:{self.role}")
            self.returncode = -9

    plane, events = make_plane(tmp_path, monkeypatch, lambda *_: True)
    process = StubbornProcess(("tunnel",), events, "tunnel:optional")
    plane.processes["tunnel"].append(("tunnel:optional", process))
    plane._drop_optional("tunnel", process)
    assert plane.processes["tunnel"] == []
    assert "kill:tunnel:optional" in events
