"""Expanding-date, train-only comparison for the ZGGG weather candidate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from .baselines import (GradientBoostingBaseline, HistoricalMedianBaseline,
                        ZgggWeatherGradientBoostingBaseline)
from .evaluate import evaluate_predictions
from .weather_residual import WeatherResidualBaseline
from .zggg_dataset import verify_zggg_dataset
from .temporal_audit import audit_temporal_rows, eligible_training_rows


@dataclass(frozen=True)
class Fold:
    train_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    train_rows: tuple[dict, ...]
    validation_rows: tuple[dict, ...]
    train_audit: dict


def expanding_folds(rows, *, initial_train_dates=9, validation_dates=3,
                    step_dates=3, input_timezone='Asia/Shanghai'):
    if min(initial_train_dates, validation_dates, step_dates) < 1:
        raise ValueError('fold date counts must be positive')
    timezone = ZoneInfo(input_timezone)
    grouped = {}
    for row in rows:
        value = row['features']['planned_off_block']
        timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError('planned_off_block must include timezone')
        date = timestamp.astimezone(timezone).date().isoformat()
        grouped.setdefault(date, []).append(row)
    dates = sorted(grouped)
    if len(dates) < initial_train_dates + validation_dates:
        raise ValueError('insufficient dates for expanding folds')
    result = []
    for end in range(initial_train_dates, len(dates) - validation_dates + 1, step_dates):
        train_dates = dates[:end]
        future_dates = dates[end:end + validation_dates]
        candidates = tuple(row for date in train_dates for row in grouped[date])
        boundary = datetime.fromisoformat(future_dates[0]).replace(tzinfo=timezone).isoformat()
        train_audit = audit_temporal_rows(candidates, boundary, input_timezone=input_timezone)
        result.append(Fold(tuple(train_dates), tuple(future_dates),
                           eligible_training_rows(candidates, boundary),
                           tuple(row for date in future_dates for row in grouped[date]),
                           train_audit))
    return tuple(result)


def evaluate_weather_rolling(dataset_dir, *, initial_train_dates=9,
                             validation_dates=3, step_dates=3):
    """Read the verified train split only; never fit or select on validation/test."""
    dataset_dir = Path(dataset_dir)
    manifest = verify_zggg_dataset(dataset_dir)
    rows = [json.loads(line) for line in (dataset_dir / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
    folds = expanding_folds(rows, initial_train_dates=initial_train_dates,
                            validation_dates=validation_dates, step_dates=step_dates,
                            input_timezone=manifest['config']['input_timezone'])
    reports = []
    for fold in folds:
        scores = {}
        for name, model in (('historical', HistoricalMedianBaseline()),
                            ('plain_gradient_boosting', GradientBoostingBaseline()),
                            ('weather', ZgggWeatherGradientBoostingBaseline()),
                            ('weather_residual', WeatherResidualBaseline()),
                            ('adverse_only_residual', WeatherResidualBaseline(adverse_only=True))):
            predictions = model.fit(fold.train_rows).predict(fold.validation_rows)
            result = evaluate_predictions(fold.validation_rows, predictions,
                                          label_manifest_hash=manifest['manifest_hash'],
                                          prediction_label_hash=manifest['manifest_hash'])
            scores[name] = {
                'off_block_mae': result.absolute_times['off_block']['mae'],
                'on_block_mae': result.absolute_times['on_block']['mae'],
                'time_order_rate': result.time_order_rate,
            }
        reports.append({'train_dates': [fold.train_dates[0], fold.train_dates[-1]],
                        'validation_dates': [fold.validation_dates[0], fold.validation_dates[-1]],
                        'train_rows': len(fold.train_rows),
                        'train_audit': fold.train_audit,
                        'validation_rows': len(fold.validation_rows), 'scores': scores,
                        'weather_non_regression': all(
                            scores['weather'][key] <= scores['historical'][key]
                            for key in ('off_block_mae', 'on_block_mae'))
                        and scores['weather']['time_order_rate'] == 1.0,
                        'weather_improves_plain_model': all(
                            scores['weather'][key] < scores['plain_gradient_boosting'][key]
                            for key in ('off_block_mae', 'on_block_mae')),
                        'residual_non_regression': all(
                            scores['weather_residual'][key] <= scores['historical'][key]
                            for key in ('off_block_mae', 'on_block_mae'))
                        and scores['weather_residual']['time_order_rate'] == 1.0,
                        'adverse_only_non_regression': all(
                            scores['adverse_only_residual'][key] <= scores['historical'][key]
                            for key in ('off_block_mae', 'on_block_mae'))
                        and scores['adverse_only_residual']['time_order_rate'] == 1.0})
    return {'dataset_manifest_hash': manifest['manifest_hash'],
            'evaluation_scope': 'train_split_expanding_dates_only',
            'folds': reports,
            'weather_passes_all_folds': all(item['weather_non_regression'] for item in reports),
            'weather_improves_plain_model_all_folds': all(
                item['weather_improves_plain_model'] for item in reports),
            'residual_passes_all_folds': all(item['residual_non_regression'] for item in reports),
            'adverse_only_passes_all_folds': all(
                item['adverse_only_non_regression'] for item in reports)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-train-dates', type=int, default=9)
    parser.add_argument('--validation-dates', type=int, default=3)
    parser.add_argument('--step-dates', type=int, default=3)
    args = parser.parse_args()
    result = evaluate_weather_rolling(args.dataset,
                                      initial_train_dates=args.initial_train_dates,
                                      validation_dates=args.validation_dates,
                                      step_dates=args.step_dates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'folds': len(result['folds']),
                      'weather_passes_all_folds': result['weather_passes_all_folds']}))


if __name__ == '__main__':
    main()
