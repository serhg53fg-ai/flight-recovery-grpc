"""Stable local identity for a base model without hashing multi-gigabyte weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


IDENTITY_FILES = ("config.json", "tokenizer_config.json", "model.safetensors.index.json")


def model_fingerprint(model_path: Path) -> str:
    root = Path(model_path).expanduser().resolve()
    entries = []
    for name in IDENTITY_FILES:
        path = root / name
        if path.is_file():
            entries.append({"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in sorted(root.glob("*.safetensors")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append({"name": path.name, "size": path.stat().st_size,
                        "sha256": digest.hexdigest()})
    if not entries or not (root / "config.json").is_file():
        raise ValueError("base model identity files are incomplete")
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
