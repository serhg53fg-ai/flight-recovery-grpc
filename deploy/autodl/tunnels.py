"""Build safe autossh commands and validate their local identity files."""

from __future__ import annotations

import os
from pathlib import Path
import re
import stat

from .manifest import SshWorker


_SSH_USER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SSH_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,252}\Z")


def _safe_connection_fields(worker: SshWorker) -> None:
    if not _SSH_USER.fullmatch(worker.ssh_user):
        raise ValueError("ssh_user contains unsupported characters")
    if not _SSH_HOST.fullmatch(worker.ssh_host) or worker.ssh_host.startswith("-"):
        raise ValueError("ssh_host contains unsupported characters")
    for name, path in (("ssh_key_file", worker.ssh_key_file), ("known_hosts_file", worker.known_hosts_file)):
        if any(character in str(path) for character in ("\n", "\r", "\x00")):
            raise ValueError(f"{name} contains unsupported characters")


def autossh_argv(worker: SshWorker) -> tuple[str, ...]:
    """Return an argv tuple intended for ``Popen(..., shell=False)``."""
    _safe_connection_fields(worker)
    return (
        "autossh", "-M", "0", "-N",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={worker.known_hosts_file}",
        "-i", str(worker.ssh_key_file),
        "-p", str(worker.ssh_port),
        "-L", (
            f"127.0.0.1:{worker.local_forward_port}:"
            f"{worker.remote_worker_host}:{worker.remote_worker_port}"
        ),
        f"{worker.ssh_user}@{worker.ssh_host}",
    )


def _regular_without_symlink(path: Path, name: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    if not os.access(path, os.R_OK):
        raise ValueError(f"{name} must be readable")
    return metadata


def validate_ssh_files(worker: SshWorker) -> None:
    """Validate files without reading or returning their contents."""
    _safe_connection_fields(worker)
    key = _regular_without_symlink(worker.ssh_key_file, "ssh_key_file")
    known_hosts = _regular_without_symlink(worker.known_hosts_file, "known_hosts_file")
    if key.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError("ssh_key_file permissions must not grant group or world access")
    if not key.st_mode & stat.S_IRUSR:
        raise ValueError("ssh_key_file must be owner-readable")
    if known_hosts.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("known_hosts_file must not be group or world writable")
