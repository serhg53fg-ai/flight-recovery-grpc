"""Retrospective random-vs-temporal comparison of an original-style ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .baselines import HistoricalMedianBaseline
from .error_diagnostics import compare_flight_predictions
from .original_style_split import random_split, temporal_split


def arm_metrics(rows, baseline, candidate, manifest_hash):
    return compare_flight_predictions(rows, baseline, candidate, manifest_hash)


def operational_metrics(paired):
    """Use the ZGGG release report's direction-specific departure/arrival scope."""
    groups = paired["groups"]
    return {"departure_off_block": groups["direction=DEPARTURE"]["off_block"],
            "arrival_on_block": groups["direction=ARRIVAL"]["on_block"]}


def temporal_gate(result):
    if result.get("evaluation_scope") != "temporal_test":
        return False
    checked = [result["overall"]]
    for group in ("severe_weather=true", "peak_period=true"):
        if group in result["groups"]:
            checked.append(result["groups"][group])
    return all(item[field]["paired_delta_mae"] <= 0
               for item in checked for field in ("off_block", "on_block"))


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def evaluate_original_style(dataset_dir, output, *, model_python=sys.executable):
    dataset_dir, output = Path(dataset_dir), Path(output)
    artifacts = output.with_name(output.stem + "-artifacts")
    if output.exists() or output.is_symlink() or artifacts.exists() or artifacts.is_symlink():
        raise ValueError("comparison output must be a new file and artifact directory")
    split = temporal_split(dataset_dir)
    # The random arm intentionally restores rows excluded from temporal fitting;
    # it is a retrospective reconstruction, never an operational release test.
    all_rows = [*_read_jsonl(dataset_dir / "train.jsonl"),
                *_read_jsonl(dataset_dir / "validation.jsonl"),
                *_read_jsonl(dataset_dir / "test.jsonl")]
    random_train, random_test = random_split(all_rows, seed=42, test_fraction=.2)
    arms = (("random", random_train, random_test),
            ("temporal", split["train"], split["test"]))
    artifacts.mkdir(parents=True)
    configuration = {"seed": 42, "random_test_fraction": .2,
                     "ensemble": "0.7_two_layer_128_relu_huber_nn+0.3_rf_100_depth10",
                     "nn_epochs": 30, "nn_batch_size": 256,
                     "feature_pipeline": "zggg-cutoff-weather-features-v1",
                     "temporal_train_boundary": split["boundary"]}
    result = {"dataset_manifest_hash": split["manifest"]["manifest_hash"],
              "configuration": configuration,
              "configuration_hash": hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest(),
              "model_python": str(model_python), "temporal_train_audit": split["train_audit"],
              "arms": {}, "release_configuration_changed": False}
    for name, train, test in arms:
        folder = artifacts / name
        folder.mkdir()
        train_file, test_file = folder / "train.jsonl", folder / "test.jsonl"
        prediction_file, metadata_file = folder / "predictions.jsonl", folder / "model-run.json"
        _write_jsonl(train_file, train)
        _write_jsonl(test_file, test)
        command = [str(model_python), "-m", "training.original_style_fit",
                   "--train", str(train_file), "--test", str(test_file),
                   "--output", str(prediction_file), "--metadata", str(metadata_file)]
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent.parent, timeout=7200)
        baseline = HistoricalMedianBaseline().fit(train).predict(test)
        candidate = _read_jsonl(prediction_file)
        paired = arm_metrics(test, baseline, candidate, split["manifest"]["manifest_hash"])
        paired["evaluation_scope"] = ("random_split_retrospective" if name == "random" else "temporal_test")
        result["arms"][name] = {"train_rows": len(train), "test_rows": len(test),
                                "train_id_sha256": hashlib.sha256("\n".join(sorted(row["record_id"] for row in train)).encode()).hexdigest(),
                                "test_id_sha256": hashlib.sha256("\n".join(sorted(row["record_id"] for row in test)).encode()).hexdigest(),
                                "model_run": json.loads(metadata_file.read_text(encoding="utf-8")),
                                "operational_metrics": operational_metrics(paired),
                                "paired": paired}
    result["retrospective_temporal_non_regression"] = temporal_gate(result["arms"]["temporal"]["paired"])
    result["publication_decision"] = "not_eligible_without_new_blind_dates_and_grpc_acceptance"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-python", default=sys.executable)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("comparison output must be a new file")
    report = evaluate_original_style(args.dataset, args.output, model_python=args.model_python)
    print(json.dumps({"output": str(args.output), "random_rows": report["arms"]["random"]["test_rows"],
                      "temporal_rows": report["arms"]["temporal"]["test_rows"],
                      "temporal_gate": report["retrospective_temporal_non_regression"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
