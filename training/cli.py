"""Command-line orchestration for local dataset and CPU baseline evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle

from .baselines import GradientBoostingBaseline, HistoricalMedianBaseline, ScheduleZeroBaseline
from .dataset import DatasetConfig, build_dataset, verify_dataset
from .evaluate import evaluate_predictions, render_markdown


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8"); temporary.replace(path)


def build_command(args):
    manifest = build_dataset(args.source, args.output, DatasetConfig(
        min_test_rows=args.min_test_rows, minimum_dates=args.minimum_dates,
        input_timezone=args.input_timezone,
    ))
    print(json.dumps({"accepted_rows": manifest.accepted_rows, "publishable": manifest.publishable,
                      "manifest_hash": manifest.manifest_hash}, sort_keys=True))
    return 0


def build_zggg_command(args):
    from .zggg_dataset import build_zggg_dataset
    manifest = build_zggg_dataset(
        args.source, args.output,
        DatasetConfig(min_test_rows=args.min_test_rows,
                      minimum_dates=args.minimum_dates,
                      input_timezone=args.input_timezone),
        weather_archive=args.weather_archive,
        weather_feature_version=args.weather_feature_version,
    )
    print(json.dumps({"airport": manifest.airport,
                      "accepted_rows": manifest.accepted_rows,
                      "publishable": manifest.publishable,
                      "manifest_hash": manifest.manifest_hash}, sort_keys=True))
    return 0


def verify_zggg_command(args):
    from .zggg_dataset import verify_zggg_dataset
    manifest = verify_zggg_dataset(args.dataset)
    print(json.dumps({"airport": manifest["airport"],
                      "accepted_rows": manifest["accepted_rows"],
                      "manifest_hash": manifest["manifest_hash"]}, sort_keys=True))
    return 0


def train_flow_command(args):
    from .flow_model import train_flow_candidate
    result = train_flow_candidate(args.dataset, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


def predict_flow_command(args):
    from .flow_model import predict_flow_file
    from .zggg_dataset import verify_zggg_dataset
    dataset = args.dataset.resolve()
    manifest = verify_zggg_dataset(dataset)
    count = predict_flow_file(args.model, dataset / f"flow_{args.split}.jsonl",
                              args.output, manifest["manifest_hash"])
    print(json.dumps({"prediction_rows": count, "split": args.split,
                      "output": str(args.output)}, sort_keys=True))
    return 0


def evaluate_zggg_command(args):
    from .zggg_dataset import verify_zggg_dataset
    from .zggg_evaluate import evaluate_zggg_predictions
    dataset = args.dataset.resolve()
    manifest = verify_zggg_dataset(dataset)
    report = evaluate_zggg_predictions(
        _rows(dataset / f"{args.split}.jsonl"), _rows(args.flight_predictions),
        _rows(dataset / f"flow_{args.split}.jsonl"), _rows(args.flow_predictions),
        manifest["manifest_hash"], args.prediction_manifest_hash,
    )
    _write(args.output, json.dumps(report, ensure_ascii=False, indent=2,
                                  sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"flight_samples": report["flight"]["samples"],
                      "flow_metrics": len(report["flow"]),
                      "output": str(args.output)}, sort_keys=True))
    return 0


def gate_zggg_command(args):
    from .zggg_evaluate import gate_zggg_release
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    result = gate_zggg_release(candidate, reference)
    _write(args.output, json.dumps(result, ensure_ascii=False, indent=2,
                                  sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 3


def train_command(args):
    dataset, output = args.dataset.resolve(), args.output.resolve()
    manifest = verify_dataset(dataset)
    train, test = _rows(dataset / "train.jsonl"), _rows(dataset / "test.jsonl")
    if not train or not test: raise ValueError("train and test splits must be non-empty")
    models = {
        "schedule_zero": ScheduleZeroBaseline(),
        "historical_median": HistoricalMedianBaseline(),
        "hist_gradient_boosting": GradientBoostingBaseline(),
    }
    summaries = {}
    for name, model in models.items():
        model.fit(train); predictions = model.predict(test)
        _write(output / f"predictions/{name}.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions))
        report = evaluate_predictions(test, predictions, label_manifest_hash=manifest["manifest_hash"],
                                      prediction_label_hash=manifest["manifest_hash"],
                                      group_fields=("departure_airport", "arrival_airport"), minimum_group_size=20)
        summaries[name] = asdict(report)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{name}.pkl").write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
        _write(output / f"{name}.md", render_markdown(report))
    result = {"dataset_manifest_hash": manifest["manifest_hash"], "test_rows": len(test), "models": summaries}
    _write(output / "metrics.json", json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"test_rows": len(test), "models": list(models)}, sort_keys=True))
    return 0


def evaluate_command(args):
    dataset = args.dataset.resolve()
    manifest = verify_dataset(dataset)
    predictions = _rows(args.predictions)
    labels = _rows(dataset / 'test.jsonl')
    report = evaluate_predictions(
        labels, predictions, label_manifest_hash=manifest['manifest_hash'],
        prediction_label_hash=args.prediction_label_hash,
        group_fields=('departure_airport', 'arrival_airport'), minimum_group_size=20,
        failed_record_ids=_failed_ids(args.failures),
    )
    result = {'dataset_manifest_hash': manifest['manifest_hash'], 'evaluation': asdict(report)}
    _write(args.output / 'metrics.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    _write(args.output / 'evaluation.md', render_markdown(report))
    print(json.dumps({'sample_count': report.sample_count,
                      'structured_success_rate': report.structured_success_rate}))
    return 0


def _failed_ids(path):
    return [item['record_id'] for item in _rows(path)] if path else []


def gate_command(args):
    from .candidate import compare_candidate
    dataset = args.dataset.resolve()
    manifest = verify_dataset(dataset)
    model = ScheduleZeroBaseline()
    model.fit(_rows(dataset / 'train.jsonl'))
    schedule_predictions = model.predict(_rows(dataset / 'test.jsonl'))
    supplied = _rows(args.reference)
    def components(items):
        from .schema import LABEL_FIELDS
        return {item['record_id']: {key: item[key] for key in LABEL_FIELDS} for item in items}
    if len(supplied) != len(schedule_predictions) or components(supplied) != components(schedule_predictions):
        raise ValueError('reference does not match recomputed schedule baseline')
    result = compare_candidate(_rows(dataset / 'test.jsonl'), _rows(args.predictions),
        schedule_predictions, _failed_ids(args.failures), manifest['manifest_hash'], args.prediction_label_hash)
    _write(args.output / 'gate.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['gate']))
    return 0 if result['gate']['passed'] else 3


def predict_qwen_command(args):
    from .batch import OfflineQwen, export_predictions, verify_prediction_dataset
    manifest, prompt_version = verify_prediction_dataset(args.dataset.resolve())
    if args.output.exists():
        raise ValueError('prediction output must be a new directory')
    if args.adapter_path:
        from services.inference.adapters.qwen import load_adapter_metadata
        model_path = args.model_path.resolve()
        metadata = load_adapter_metadata(args.adapter_path, model_path.name, model_path)
        if metadata['dataset_manifest_hash'] != manifest['manifest_hash']:
            raise ValueError('adapter dataset manifest mismatch')
        if metadata.get('prompt_version') != prompt_version:
            raise ValueError('adapter prompt version mismatch')
    predictor = OfflineQwen(args.model_path, args.adapter_path, args.max_new_tokens,
                            prompt_version=prompt_version)
    print(json.dumps(export_predictions(args.dataset, args.output, predictor, predictor.identity)))
    return 0


def compose_hybrid_command(args):
    from .hybrid import compose_predictions
    if args.output.exists():
        raise ValueError('hybrid prediction output must be a new file')
    predictions = compose_predictions(
        _rows(args.candidate), _rows(args.baseline), args.baseline_fields,
    )
    _write(args.output, ''.join(
        json.dumps(row, sort_keys=True) + '\n' for row in predictions
    ))
    print(json.dumps({
        'prediction_rows': len(predictions),
        'baseline_fields': args.baseline_fields,
        'output': str(args.output),
    }, sort_keys=True))
    return 0


def predict_grpc_command(args):
    from apps.flight.clients.inference import InferenceClient
    from .rpc_batch import export_rpc_predictions
    dataset = args.dataset.resolve()
    verify_dataset(dataset)
    client = InferenceClient(args.gateway, args.timeout)
    try:
        result = export_rpc_predictions(
            dataset / f'{args.split}.jsonl', args.output.resolve(), client,
            timeout=args.timeout, resume=args.resume,
        )
    finally:
        client.close()
    print(json.dumps(result, sort_keys=True))
    return 0 if result['complete'] else 3


def _split_baselines(dataset, split):
    train, rows = _rows(dataset / 'train.jsonl'), _rows(dataset / f'{split}.jsonl')
    historical = HistoricalMedianBaseline().fit(train).predict(rows)
    schedule = ScheduleZeroBaseline().fit(train).predict(rows)
    return rows, historical, schedule


def fit_guard_policy_command(args):
    from .guarded import fit_policy
    from .hybrid import compose_predictions
    dataset, output = args.dataset.resolve(), args.output.resolve()
    manifest = verify_dataset(dataset)
    if output.exists():
        raise ValueError('guard policy output must be a new directory')
    rows, historical, schedule = _split_baselines(dataset, 'validation')
    hybrid = compose_predictions(_rows(args.candidate), historical, ('off_block_delay_min',))
    policy = fit_policy(rows, hybrid, schedule, args.minimum_group_size, args.maximum_regression)
    policy['dataset_manifest_hash'] = manifest['manifest_hash']
    _write(output / 'policy.json', json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    _write(output / 'validation_candidate.jsonl', ''.join(json.dumps(row, sort_keys=True) + '\n' for row in hybrid))
    _write(output / 'validation_reference.jsonl', ''.join(json.dumps(row, sort_keys=True) + '\n' for row in schedule))
    print(json.dumps({'validation_rows': len(rows),
                      'eligible_departure_airports': len(policy['eligible_departure_airports']),
                      'eligible_arrival_airports': len(policy['eligible_arrival_airports'])}, sort_keys=True))
    return 0


def apply_guard_policy_command(args):
    from .guarded import apply_policy
    from .hybrid import compose_predictions
    dataset = args.dataset.resolve()
    manifest = verify_dataset(dataset)
    policy = json.loads(args.policy.read_text(encoding='utf-8'))
    if policy.get('dataset_manifest_hash') != manifest['manifest_hash']:
        raise ValueError('guard policy dataset manifest mismatch')
    if args.output.exists():
        raise ValueError('guarded prediction output must be a new file')
    rows, historical, schedule = _split_baselines(dataset, 'test')
    hybrid = compose_predictions(_rows(args.candidate), historical, ('off_block_delay_min',))
    guarded = apply_policy(rows, hybrid, schedule, policy)
    _write(args.output, ''.join(json.dumps(row, sort_keys=True) + '\n' for row in guarded))
    print(json.dumps({'prediction_rows': len(guarded), 'output': str(args.output)}, sort_keys=True))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build and evaluate flight prediction models")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "build"):
        command = commands.add_parser(name); command.add_argument("--source", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True); command.add_argument("--min-test-rows", type=int, default=500)
        command.add_argument("--minimum-dates", type=int, default=10); command.add_argument("--input-timezone", default="Asia/Shanghai")
        command.set_defaults(run=build_command)
    command = commands.add_parser("build-zggg")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--weather-archive", type=Path)
    command.add_argument("--weather-feature-version", choices=("aviation-weather-v1", "aviation-weather-v2"),
                         default="aviation-weather-v1")
    command.add_argument("--min-test-rows", type=int, default=500)
    command.add_argument("--minimum-dates", type=int, default=10)
    command.add_argument("--input-timezone", default="Asia/Shanghai")
    command.set_defaults(run=build_zggg_command)
    command = commands.add_parser("verify-zggg")
    command.add_argument("--dataset", type=Path, required=True)
    command.set_defaults(run=verify_zggg_command)
    command = commands.add_parser("train-flow")
    command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(run=train_flow_command)
    command = commands.add_parser("predict-flow")
    command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--split", choices=("validation", "test"), default="test")
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(run=predict_flow_command)
    command = commands.add_parser("evaluate-zggg")
    command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--split", choices=("validation", "test"), default="test")
    command.add_argument("--flight-predictions", type=Path, required=True)
    command.add_argument("--flow-predictions", type=Path, required=True)
    command.add_argument("--prediction-manifest-hash", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(run=evaluate_zggg_command)
    command = commands.add_parser("gate-zggg")
    command.add_argument("--candidate", type=Path, required=True)
    command.add_argument("--reference", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(run=gate_zggg_command)
    command = commands.add_parser("train-baseline"); command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True); command.set_defaults(run=train_command)
    for name, handler in (('evaluate', evaluate_command), ('gate', gate_command)):
        command = commands.add_parser(name)
        command.add_argument('--dataset', type=Path, required=True)
        command.add_argument('--predictions', type=Path, required=True)
        command.add_argument('--failures', type=Path)
        command.add_argument('--prediction-label-hash', required=True)
        command.add_argument('--output', type=Path, required=True)
        if name == 'gate':
            command.add_argument('--reference', type=Path, required=True)
        command.set_defaults(run=handler)
    command = commands.add_parser('predict-qwen')
    command.add_argument('--dataset', type=Path, required=True)
    command.add_argument('--model-path', type=Path, required=True)
    command.add_argument('--adapter-path', type=Path)
    command.add_argument('--max-new-tokens', type=int, default=192)
    command.add_argument('--output', type=Path, required=True)
    command.set_defaults(run=predict_qwen_command)
    command = commands.add_parser('compose-hybrid')
    command.add_argument('--candidate', type=Path, required=True)
    command.add_argument('--baseline', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--baseline-fields', nargs='+', default=['off_block_delay_min'])
    command.set_defaults(run=compose_hybrid_command)
    command = commands.add_parser('predict-grpc')
    command.add_argument('--dataset', type=Path, required=True)
    command.add_argument('--split', choices=('validation', 'test'), default='validation')
    command.add_argument('--gateway', required=True)
    command.add_argument('--timeout', type=float, default=20)
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--resume', action='store_true')
    command.set_defaults(run=predict_grpc_command)
    command = commands.add_parser('fit-guard-policy')
    command.add_argument('--dataset', type=Path, required=True)
    command.add_argument('--candidate', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--minimum-group-size', type=int, default=20)
    command.add_argument('--maximum-regression', type=float, default=0.0)
    command.set_defaults(run=fit_guard_policy_command)
    command = commands.add_parser('apply-guard-policy')
    command.add_argument('--dataset', type=Path, required=True)
    command.add_argument('--candidate', type=Path, required=True)
    command.add_argument('--policy', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    command.set_defaults(run=apply_guard_policy_command)
    return parser.parse_args(argv)


def main(argv=None):
    try:
        args = parse_args(argv)
        return args.run(args)
    except (ValueError, OSError, json.JSONDecodeError, KeyError) as error:
        print(f"training command failed: {error}")
        return 2


if __name__ == "__main__": raise SystemExit(main())
