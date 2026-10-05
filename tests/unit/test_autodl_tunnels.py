import os
from pathlib import Path

import pytest

from deploy.autodl.manifest import SshWorker
from deploy.autodl.tunnels import autossh_argv, validate_ssh_files


def ssh_worker(tmp_path: Path) -> SshWorker:
    return SshWorker(
        id="autodl-b-gpu0",
        ssh_host="connect.example.autodl.com",
        ssh_port=12345,
        ssh_user="root",
        ssh_key_file=tmp_path / "id_cluster",
        known_hosts_file=tmp_path / "known_hosts",
        remote_worker_host="127.0.0.1",
        remote_worker_port=50052,
        local_forward_port=15053,
        capacity=1,
        required=True,
    )


def test_generates_exact_safe_autossh_arguments(tmp_path):
    worker = ssh_worker(tmp_path)
    assert autossh_argv(worker) == (
        "autossh", "-M", "0", "-N",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={worker.known_hosts_file}",
        "-i", str(worker.ssh_key_file),
        "-p", "12345",
        "-L", "127.0.0.1:15053:127.0.0.1:50052",
        "root@connect.example.autodl.com",
    )
    assert all("password" not in argument.lower() for argument in autossh_argv(worker))
    assert "StrictHostKeyChecking=no" not in autossh_argv(worker)


def test_accepts_secure_regular_ssh_files(tmp_path):
    worker = ssh_worker(tmp_path)
    worker.ssh_key_file.write_text("test-only-key", encoding="utf-8")
    worker.known_hosts_file.write_text("example fingerprint", encoding="utf-8")
    os.chmod(worker.ssh_key_file, 0o600)
    os.chmod(worker.known_hosts_file, 0o644)
    validate_ssh_files(worker)


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o777])
def test_rejects_group_or_world_accessible_private_key(tmp_path, mode):
    worker = ssh_worker(tmp_path)
    worker.ssh_key_file.write_text("test-only-key", encoding="utf-8")
    worker.known_hosts_file.write_text("example fingerprint", encoding="utf-8")
    os.chmod(worker.ssh_key_file, mode)
    with pytest.raises(ValueError):
        validate_ssh_files(worker)


@pytest.mark.parametrize("missing", ["key", "known_hosts"])
def test_rejects_missing_ssh_files(tmp_path, missing):
    worker = ssh_worker(tmp_path)
    if missing != "key":
        worker.ssh_key_file.write_text("test-only-key", encoding="utf-8")
        os.chmod(worker.ssh_key_file, 0o600)
    if missing != "known_hosts":
        worker.known_hosts_file.write_text("example fingerprint", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_ssh_files(worker)


@pytest.mark.parametrize("target", ["key", "known_hosts"])
def test_rejects_symlinked_ssh_files(tmp_path, target):
    worker = ssh_worker(tmp_path)
    real_key = tmp_path / "real_key"
    real_hosts = tmp_path / "real_hosts"
    real_key.write_text("test-only-key", encoding="utf-8")
    real_hosts.write_text("example fingerprint", encoding="utf-8")
    os.chmod(real_key, 0o600)
    worker.ssh_key_file.symlink_to(real_key)
    worker.known_hosts_file.symlink_to(real_hosts)
    if target == "key":
        worker.known_hosts_file.unlink()
        worker.known_hosts_file.write_text("example fingerprint", encoding="utf-8")
    else:
        worker.ssh_key_file.unlink()
        worker.ssh_key_file.write_text("test-only-key", encoding="utf-8")
        os.chmod(worker.ssh_key_file, 0o600)
    with pytest.raises(ValueError):
        validate_ssh_files(worker)


def test_rejects_nonregular_ssh_files(tmp_path):
    worker = ssh_worker(tmp_path)
    worker.ssh_key_file.mkdir()
    worker.known_hosts_file.write_text("example fingerprint", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_ssh_files(worker)


@pytest.mark.parametrize("mode", [0o666, 0o664, 0o646])
def test_rejects_group_or_world_writable_known_hosts(tmp_path, mode):
    worker = ssh_worker(tmp_path)
    worker.ssh_key_file.write_text("test-only-key", encoding="utf-8")
    worker.known_hosts_file.write_text("example fingerprint", encoding="utf-8")
    os.chmod(worker.ssh_key_file, 0o600)
    os.chmod(worker.known_hosts_file, mode)
    with pytest.raises(ValueError):
        validate_ssh_files(worker)
