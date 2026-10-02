"""Train-only rolling comparison of historical and cutoff-weather table candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .baselines import HistoricalMedianBaseline, ZgggWeatherGradientBoostingBaseline
from .error_diagnostics import compare_flight_predictions
from .rolling_weather import expanding_folds
from .schema import LABEL_FIELDS
from .zggg_dataset import verify_zggg_dataset


def evaluate_fold_predictions(rows, baseline, candidate, manifest_hash):
    return compare_flight_predictions(rows, baseline, candidate, manifest_hash)


def fold_gate(paired):
    groups = paired["groups"]
    checked = [paired["overall"]]
    for name in ("severe_weather=true", "peak_period=true"):
        if name in groups:
            checked.append(groups[name])
    return all(item[metric]["paired_delta_mae"] <= 0
               for item in checked for metric in ("off_block", "on_block"))


def _valid(predictions):
    if not all(all(row[name] >= 0 for name in LABEL_FIELDS[1:]) for row in predictions):
        raise ValueError("candidate predicted negative duration")
    return predictions


def evaluate_original_method_rolling(dataset_dir: Path) -> dict:
    dataset_dir = Path(dataset_dir)
    manifest = verify_zggg_dataset(dataset_dir)
    rows = [json.loads(line) for line in (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    folds = expanding_folds(rows, input_timezone=manifest["config"]["input_timezone"])
    configuration = {"scope": "train_split_expanding_dates_only", "models": ["historical", "weather_histogram_gbdt"],
                     "initial_train_dates": 9, "validation_dates": 3, "step_dates": 3,
                     "gate": "non_regression_overall_severe_peak_off_and_on_block"}
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    results = []
    for fold in folds:
        predictions = {}
        latency = {}
        for name, model in (("historical", HistoricalMedianBaseline()),
                            ("weather_histogram_gbdt", ZgggWeatherGradientBoostingBaseline())):
            start = perf_counter()
            predictions[name] = _valid(model.fit(fold.train_rows).predict(fold.validation_rows))
            latency[name] = perf_counter() - start
        paired = evaluate_fold_predictions(fold.validation_rows, predictions["historical"],
                                           predictions["weather_histogram_gbdt"], manifest["manifest_hash"])
        results.append({"train_dates": [fold.train_dates[0], fold.train_dates[-1]],
                        "validation_dates": [fold.validation_dates[0], fold.validation_dates[-1]],
                        "train_rows": len(fold.train_rows), "train_audit": fold.train_audit,
                        "validation_rows": len(fold.validation_rows), "paired": paired,
                        "fit_predict_seconds": latency, "gate_pass": fold_gate(paired)})
    return {"dataset_manifest_hash": manifest["manifest_hash"], "configuration": configuration,
            "configuration_hash": config_hash, "folds": results,
            "passes_all_folds": all(fold["gate_pass"] for fold in results)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("experiment output must be a new file")
    result = evaluate_original_method_rolling(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "folds": len(result["folds"]),
                      "passes_all_folds": result["passes_all_folds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
