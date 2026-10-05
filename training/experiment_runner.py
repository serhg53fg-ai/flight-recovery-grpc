"""Run a fixed, train-split-only ZGGG flight and flow comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import pickle
from zoneinfo import ZoneInfo

import numpy as np

from .baselines import (GradientBoostingBaseline, HistoricalMedianBaseline,
                        ZgggWeatherGradientBoostingBaseline)
from .evaluate import evaluate_predictions
from .flow_model import (FLOW_LABELS, GradientBoostingFlowModel,
                         ScheduleFlowBaseline)
from .rolling_weather import expanding_folds
from .zggg_dataset import verify_zggg_dataset
from .zggg_evaluate import evaluate_zggg_predictions, _peak, _severe


def candidate_specs():
    common = {'split_policy': 'same_train_split_expanding_dates',
              'label_policy': 'same_fold_validation_ids',
              'parameters': {'random_state': 42, 'rounds': 40,
                             'learning_rate': .1, 'bins': 16}}
    return {
        'historical_median': {**common, 'flight_model': 'historical_median',
                              'flight_feature_pipeline': 'route_hour_median',
                              'flow_model': 'schedule'},
        'plain_gradient_boosting': {**common, 'flight_model': 'gradient_boosting',
                                    'flight_feature_pipeline': 'features-v1',
                                    'flow_model': 'gradient_boosting'},
        'weather_gradient_boosting': {**common, 'flight_model': 'gradient_boosting',
                                      'flight_feature_pipeline': 'zggg-cutoff-weather-features-v1',
                                      'flow_model': 'gradient_boosting'},
    }


def validate_prediction_ids(labels, predictions):
    expected = [row['record_id'] for row in labels]
    actual = [row['record_id'] for row in predictions]
    if (len(expected) != len(set(expected)) or len(actual) != len(set(actual)) or
            set(expected) != set(actual)):
        raise ValueError('prediction IDs do not match shared labels')


def metric_deltas(candidate, baseline):
    if set(candidate) != set(FLOW_LABELS) or set(baseline) != set(FLOW_LABELS):
        raise ValueError('all flow horizons must be retained')
    return {name: float(candidate[name]['mae'] - baseline[name]['mae'])
            for name in FLOW_LABELS}


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def _date(value, zone):
    timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if timestamp.tzinfo is None:
        raise ValueError('fold timestamps must include timezone')
    return timestamp.astimezone(zone).date().isoformat()


def _flow_fold(rows, fold, zone):
    train_dates, validation_dates = set(fold.train_dates), set(fold.validation_dates)
    boundary = datetime.fromisoformat(fold.validation_dates[0]).replace(tzinfo=zone)
    train, validation = [], []
    unavailable = 0
    for row in rows:
        stamp = datetime.fromisoformat(row['features']['prediction_cutoff_time'].replace('Z', '+00:00'))
        local = stamp.astimezone(zone)
        date = local.date().isoformat()
        if date in validation_dates:
            validation.append(row)
        elif date in train_dates:
            if local + timedelta(minutes=60) <= boundary:
                train.append(row)
            else:
                unavailable += 1
    return train, validation, unavailable


def _flight_model(name):
    return {'historical_median': HistoricalMedianBaseline,
            'plain_gradient_boosting': GradientBoostingBaseline,
            'weather_gradient_boosting': ZgggWeatherGradientBoostingBaseline}[name]()


def _flow_model(name):
    return {'schedule': ScheduleFlowBaseline,
            'gradient_boosting': GradientBoostingFlowModel}[name]()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _ids_hash(rows):
    return _digest(json.dumps(sorted(row['record_id'] for row in rows),
                              separators=(',', ':')).encode())


def _groups(rows, predictions):
    by_id = {row['record_id']: row for row in predictions}
    values = {}
    for row in rows:
        feature = row['features']
        prediction = by_id[row['record_id']]
        names = ('off_block_delay_min', 'taxi_out_min', 'airborne_min', 'taxi_in_min')
        error = abs(sum(float(prediction[name]) - float(row['labels'][name]) for name in names))
        labels = (f"direction={'DEPARTURE' if feature['departure_airport'] == 'ZGGG' else 'ARRIVAL'}",
                  f'peak={str(_peak(feature)).lower()}',
                  f'severe_weather={str(_severe(feature)).lower()}')
        for label in labels:
            values.setdefault(label, []).append(error)
    return {name: {'samples': len(errors), 'on_block_mae': float(np.mean(errors))}
            for name, errors in sorted(values.items())}


def run_experiment(config_path: Path, output: Path) -> dict:
    config_path, output = Path(config_path), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('experiment output already exists')
    if not output.parent.is_dir():
        raise ValueError('experiment output parent is missing')
    raw_config = config_path.read_bytes()
    config = json.loads(raw_config)
    if not isinstance(config.get('experiment_id'), str) or not config['experiment_id']:
        raise ValueError('experiment ID is required')
    specs = candidate_specs()
    if (config.get('candidates') != list(specs) or config.get('development_split') != 'train' or
            config.get('locked_splits') != ['validation', 'test'] or
            config.get('random_seed') != 42 or
            set(config.get('targets', [])) != set(('off_block_delay_min', 'taxi_out_min',
                'airborne_min', 'taxi_in_min', *FLOW_LABELS))):
        raise ValueError('experiment candidate, target or split contract changed')
    dataset = Path(config['dataset_dir'])
    manifest = verify_zggg_dataset(dataset)
    for key, field in (('dataset_manifest_hash', 'manifest_hash'),
                       ('source_sha256', 'source_sha256'),
                       ('weather_archive_sha256', 'weather_archive_sha256')):
        if manifest.get(field) != config.get(key):
            raise ValueError(f'{key} does not match verified dataset')
    flight_rows = _read_jsonl(dataset / 'train.jsonl')
    flow_rows = _read_jsonl(dataset / 'flow_train.jsonl')
    folds = expanding_folds(flight_rows, **config['folds'],
                            input_timezone=config['timezone'])
    zone = ZoneInfo(config['timezone'])
    reports = []
    for index, fold in enumerate(folds, 1):
        train_flow, validation_flow, excluded_flow = _flow_fold(flow_rows, fold, zone)
        if not fold.train_rows or not fold.validation_rows or not train_flow or not validation_flow:
            raise ValueError('a fold has no usable flight or flow rows')
        flow_models = {}
        for name in ('schedule', 'gradient_boosting'):
            model = _flow_model(name).fit(train_flow)
            predictions = model.predict(validation_flow)
            validate_prediction_ids(validation_flow, predictions)
            flow_models[name] = (predictions, _digest(pickle.dumps(model, protocol=4)))
        scores = {}
        for name, spec in specs.items():
            model = _flight_model(name).fit(fold.train_rows)
            predictions = model.predict(fold.validation_rows)
            validate_prediction_ids(fold.validation_rows, predictions)
            flow_predictions, flow_hash = flow_models[spec['flow_model']]
            metrics = evaluate_zggg_predictions(
                fold.validation_rows, predictions, validation_flow, flow_predictions,
                manifest['manifest_hash'], manifest['manifest_hash'])
            extended = evaluate_predictions(
                fold.validation_rows, predictions,
                label_manifest_hash=manifest['manifest_hash'],
                prediction_label_hash=manifest['manifest_hash'])
            metrics['flight']['absolute_times'] = extended.absolute_times
            metrics['flight']['groups_complete'] = _groups(fold.validation_rows, predictions)
            scores[name] = {'metrics': metrics,
                            'flight_model_sha256': _digest(pickle.dumps(model, protocol=4)),
                            'flow_model_sha256': flow_hash}
        baseline_flow = scores['historical_median']['metrics']['flow']
        for name in specs:
            scores[name]['flow_mae_delta_vs_historical'] = metric_deltas(
                scores[name]['metrics']['flow'], baseline_flow)
        reports.append({'fold': index, 'train_dates': [fold.train_dates[0], fold.train_dates[-1]],
                        'validation_dates': [fold.validation_dates[0], fold.validation_dates[-1]],
                        'flight_train_rows': len(fold.train_rows),
                        'flight_train_audit': fold.train_audit,
                        'flight_validation_rows': len(fold.validation_rows),
                        'flight_validation_ids_sha256': _ids_hash(fold.validation_rows),
                        'flow_train_rows': len(train_flow),
                        'flow_train_labels_unavailable': excluded_flow,
                        'flow_validation_rows': len(validation_flow),
                        'flow_validation_ids_sha256': _ids_hash(validation_flow),
                        'scores': scores})
    report = {'experiment_id': config['experiment_id'],
              'config_sha256': _digest(raw_config),
              'dataset_manifest_hash': manifest['manifest_hash'],
              'source_sha256': manifest['source_sha256'],
              'weather_archive_sha256': manifest['weather_archive_sha256'],
              'selection_scope': 'train_split_expanding_dates_only',
              'candidate_specs': specs, 'folds': reports}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_experiment(args.config, args.output)
    print(json.dumps({'experiment_id': result['experiment_id'],
                      'config_sha256': result['config_sha256'],
                      'folds': len(result['folds']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
