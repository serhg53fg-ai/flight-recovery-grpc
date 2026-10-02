import json
from pathlib import Path

import pytest

from deploy.autodl.manifest import load_manifest, to_gateway_config


def valid_manifest():
    return {
        "release_manifest_path": None,
        "control": {
            "web_host": "0.0.0.0",
            "web_port": 6006,
            "gateway_host": "127.0.0.1",
            "gateway_port": 50051,
            "rpc_timeout_ms": 30000,
        },
        "workers": [
            {
                "id": "autodl-a-gpu0",
                "mode": "local",
                "gateway_address": "127.0.0.1:50052",
                "gpu_index": 0,
                "model_path": None,
                "adapter_path": None,
                "flow_model_path": None,
                "historical_model_path": None,
                "feature_contract_path": None,
                "capacity": 1,
                "required": True,
            },
            {
                "id": "autodl-b-gpu0",
                "mode": "ssh",
                "ssh_host": "connect.example.autodl.com",
                "ssh_port": 12345,
                "ssh_user": "root",
                "ssh_key_file": "/root/.ssh/autodl_cluster",
                "known_hosts_file": "/root/.ssh/known_hosts",
                "remote_worker_host": "127.0.0.1",
                "remote_worker_port": 50052,
                "local_forward_port": 15053,
                "model_path": "/srv/flight-example/models/Qwen",
                "adapter_path": "/srv/flight-example/project/runtime/training/adapter",
                "flow_model_path": "/srv/flight-example/project/runtime/training/flow/flow_model.pkl",
                "historical_model_path": "/srv/flight-example/project/runtime/training/baseline.pkl",
                "feature_contract_path": "/srv/flight-example/project/runtime/training/flow/metrics.json",
                "capacity": 2,
                "required": False,
            },
        ],
    }


def write_manifest(tmp_path: Path, value) -> Path:
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_loads_valid_manifest_and_generates_existing_gateway_schema(tmp_path):
    manifest = load_manifest(write_manifest(tmp_path, valid_manifest()))

    assert manifest.control.gateway_port == 50051
    assert manifest.release_manifest_path is None
    assert manifest.workers[0].required is True
    assert manifest.workers[0].gpu_index == 0
    assert manifest.workers[1].local_forward_port == 15053
    assert to_gateway_config(manifest) == {
        "listen": "127.0.0.1:50051",
        "rpc_timeout_ms": 30000,
        "health_interval_ms": 2000,
        "health_timeout_ms": 500,
        "failure_threshold": 3,
        "open_cooldown_ms": 10000,
        "minimum_retry_budget_ms": 500,
        "workers": [
            {"id": "autodl-a-gpu0", "address": "127.0.0.1:50052", "capacity": 1, "enabled": True},
            {"id": "autodl-b-gpu0", "address": "127.0.0.1:15053", "capacity": 2, "enabled": True},
        ],
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra=True),
        lambda value: value.pop("release_manifest_path"),
        lambda value: value["control"].pop("web_host"),
        lambda value: value["control"].update(extra=True),
        lambda value: value["workers"][0].update(extra=True),
        lambda value: value["workers"][0].update(mode="other"),
        lambda value: value["workers"][1].pop("ssh_user"),
        lambda value: value["workers"].append(dict(value["workers"][0])),
        lambda value: value["workers"][1].update(local_forward_port=50051),
        lambda value: value["workers"][1].update(local_forward_port=50052),
        lambda value: value["workers"].append({**value["workers"][1], "id": "third", "local_forward_port": 15053}),
        lambda value: value["workers"][0].update(capacity=0),
        lambda value: value["workers"][0].update(capacity=257),
        lambda value: value["workers"][0].update(gpu_index=-1),
        lambda value: value["workers"][0].update(gpu_index=True),
        lambda value: value["workers"][1].update(remote_worker_host="0.0.0.0"),
        lambda value: value["workers"][1].update(remote_worker_host="10.0.0.2"),
        lambda value: value["workers"][1].update(ssh_key_file="relative/key"),
        lambda value: value["workers"][1].update(known_hosts_file="known_hosts"),
        lambda value: value["workers"][0].update(feature_contract_path="relative.json"),
        lambda value: value["workers"][1].update(ssh_user=""),
        lambda value: value["workers"][1].update(ssh_host=""),
        lambda value: value["workers"][0].update(required=1),
        lambda value: value["control"].update(web_port=True),
        lambda value: value["workers"].clear(),
        lambda value: [worker.update(required=False) for worker in value["workers"]],
    ],
)
def test_rejects_invalid_manifest(tmp_path, mutate):
    value = valid_manifest()
    mutate(value)
    with pytest.raises(ValueError):
        load_manifest(write_manifest(tmp_path, value))


def test_rejects_more_than_256_workers(tmp_path):
    value = valid_manifest()
    template = value["workers"][0]
    value["workers"] = [{**template, "id": f"worker-{index}", "gateway_address": f"127.0.0.1:{20000 + index}"} for index in range(257)]
    with pytest.raises(ValueError):
        load_manifest(write_manifest(tmp_path, value))


def test_rejects_duplicate_local_gpu_index(tmp_path):
    value = valid_manifest()
    value["workers"][1] = {
        **value["workers"][0],
        "id": "autodl-a-gpu0-copy",
        "gateway_address": "127.0.0.1:50053",
    }
    with pytest.raises(ValueError, match="gpu"):
        load_manifest(write_manifest(tmp_path, value))


def test_loads_absolute_release_manifest_path(tmp_path):
    value = valid_manifest()
    value["release_manifest_path"] = "/root/releases/candidate.json"
    manifest = load_manifest(write_manifest(tmp_path, value))
    assert manifest.release_manifest_path == Path("/root/releases/candidate.json")


def test_rejects_relative_release_manifest_path(tmp_path):
    value = valid_manifest()
    value["release_manifest_path"] = "runtime/release.json"
    with pytest.raises(ValueError, match="absolute"):
        load_manifest(write_manifest(tmp_path, value))
