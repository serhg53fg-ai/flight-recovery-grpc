"""Frozen random and chronological splits for original-style model comparison."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from random import Random
from zoneinfo import ZoneInfo

from .temporal_audit import audit_temporal_rows, eligible_training_rows
from .zggg_dataset import verify_zggg_dataset


def random_split(rows, *, seed=42, test_fraction=.2):
    if not 0 < test_fraction < 1 or len(rows) < 2:
        raise ValueError("invalid random split configuration")
    ids = [row.get("record_id") for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("duplicate or missing record ID")
    indices = list(range(len(rows)))
    Random(seed).shuffle(indices)
    test_count = max(1, min(len(rows) - 1, round(len(rows) * test_fraction)))
    test_ids = {ids[index] for index in indices[:test_count]}
    return ([row for row in rows if row["record_id"] not in test_ids],
            [row for row in rows if row["record_id"] in test_ids])


def eligible_before(rows, boundary, *, input_timezone="Asia/Shanghai"):
    audit = audit_temporal_rows(rows, boundary, input_timezone=input_timezone)
    return list(eligible_training_rows(rows, boundary)), audit


def temporal_split(dataset_dir):
    dataset_dir = Path(dataset_dir)
    manifest = verify_zggg_dataset(dataset_dir)
    splits = {name: [json.loads(line) for line in
                     (dataset_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
              for name in ("train", "validation", "test")}
    dates = manifest["splits"]["validation"]["dates"]
    if not dates:
        raise ValueError("validation split has no boundary date")
    timezone_name = manifest["config"]["input_timezone"]
    boundary = datetime.fromisoformat(dates[0]).replace(tzinfo=ZoneInfo(timezone_name)).isoformat()
    eligible, audit = eligible_before(splits["train"], boundary, input_timezone=timezone_name)
    return {"manifest": manifest, "train": eligible,
            "validation": splits["validation"], "test": splits["test"],
            "train_audit": audit, "boundary": boundary}
