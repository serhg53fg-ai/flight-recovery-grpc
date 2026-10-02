"""Export split-isolated Qwen SFT records from a dataset manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .prompt import LEGACY_PROMPT_VERSION, PROMPT_VERSION, render_duration_prompt
from .schema import LABEL_FIELDS
from .dataset import verify_dataset


def export_sft(dataset_dir: Path, output_dir: Path):
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    try:
        raw_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid dataset manifest") from error
    is_zggg = raw_manifest.get("schema_version") == "zggg-flight-weather-v1"
    if is_zggg:
        from .zggg_dataset import verify_zggg_dataset
        manifest = verify_zggg_dataset(dataset_dir)
        prompt_version = PROMPT_VERSION
    else:
        manifest = verify_dataset(dataset_dir)
        prompt_version = LEGACY_PROMPT_VERSION
    manifest_hash = manifest.get("manifest_hash")
    if not isinstance(manifest_hash, str) or not manifest_hash: raise ValueError("dataset manifest lacks hash")
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    fit_audits = {}
    for split in ("train", "validation", "test"):
        output = []
        rows = [json.loads(line) for line in
                (dataset_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        if is_zggg and split in ("train", "validation"):
            from .temporal_audit import audit_temporal_rows, eligible_training_rows
            next_split = "validation" if split == "train" else "test"
            dates = manifest["splits"][next_split]["dates"]
            if not dates:
                raise ValueError(f"{next_split} has no boundary date for temporal SFT filtering")
            timezone_name = manifest["config"]["input_timezone"]
            boundary = datetime.fromisoformat(dates[0]).replace(
                tzinfo=ZoneInfo(timezone_name)).isoformat()
            audit = audit_temporal_rows(rows, boundary, input_timezone=timezone_name)
            fit_audits[split] = {key: audit[key] for key in
                                  ("cutoff", "rows", "eligible_rows", "excluded_rows",
                                   "reasons", "limitations")}
            rows = eligible_training_rows(rows, boundary)
        for row in rows:
            answer = {name: row["labels"][name] for name in LABEL_FIELDS}
            output.append({"record_id": row["record_id"], "prompt": render_duration_prompt(row["features"], prompt_version),
                           "response": json.dumps(answer, sort_keys=True, separators=(",", ":")),
                           "prompt_version": prompt_version})
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in output)
        temporary = output_dir / f"{split}.jsonl.tmp"
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(output_dir / f"{split}.jsonl")
        hashes[split] = hashlib.sha256(text.encode()).hexdigest()
    result = {"dataset_manifest_hash": manifest_hash, "prompt_version": prompt_version,
              "splits": hashes}
    if is_zggg:
        result.update({"airport_scope": "ZGGG",
                       "weather_feature_version": manifest["weather_feature_version"],
                       "temporal_fit_filter": fit_audits})
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")
    return result
