"""A02 bounded candidate protocol and immutable evidence."""

import json
from pathlib import Path

import pytest


def test_all_candidates_share_splits_and_labels():
    from training.experiment_runner import candidate_specs, validate_prediction_ids

    specs = candidate_specs()
    assert set(specs) == {'historical_median', 'plain_gradient_boosting',
                          'weather_gradient_boosting'}
    assert len({spec['split_policy'] for spec in specs.values()}) == 1
    assert len({spec['label_policy'] for spec in specs.values()}) == 1
    with pytest.raises(ValueError, match='prediction IDs'):
        validate_prediction_ids([{'record_id': 'a'}], [{'record_id': 'b'}])


def test_weather_ablation_changes_only_weather_features():
    from training.experiment_runner import candidate_specs

    specs = candidate_specs()
    plain = specs['plain_gradient_boosting']
    weather = specs['weather_gradient_boosting']
    assert plain['flight_model'] == weather['flight_model']
    assert plain['flow_model'] == weather['flow_model']
    assert plain['parameters'] == weather['parameters']
    assert plain['flight_feature_pipeline'] != weather['flight_feature_pipeline']


def test_report_keeps_horizon_regressions():
    from training.experiment_runner import metric_deltas
    from training.flow_model import FLOW_LABELS

    baseline = {name: {'mae': 2.0} for name in FLOW_LABELS}
    candidate = {name: {'mae': 1.0} for name in FLOW_LABELS}
    candidate['takeoff_60m']['mae'] = 4.0
    deltas = metric_deltas(candidate, baseline)
    assert set(deltas) == set(FLOW_LABELS)
    assert deltas['takeoff_15m'] == -1.0
    assert deltas['takeoff_60m'] == 2.0


def test_existing_experiment_output_not_overwritten(tmp_path):
    from training.experiment_runner import run_experiment

    output = tmp_path / 'result.json'
    output.write_text('previous')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'experiment_id': 'test'}))
    with pytest.raises(ValueError, match='exists'):
        run_experiment(config, output)
    assert output.read_text() == 'previous'


def test_flow_train_labels_must_finish_before_validation_boundary():
    from types import SimpleNamespace
    from zoneinfo import ZoneInfo
    from training.experiment_runner import _flow_fold

    fold = SimpleNamespace(train_dates=('2025-05-09',),
                           validation_dates=('2025-05-10',))
    rows = [
        {'record_id': 'safe', 'features': {'prediction_cutoff_time': '2025-05-09T14:00:00Z'}},
        {'record_id': 'late', 'features': {'prediction_cutoff_time': '2025-05-09T15:30:00Z'}},
        {'record_id': 'valid', 'features': {'prediction_cutoff_time': '2025-05-09T16:00:00Z'}},
    ]
    train, validation, excluded = _flow_fold(rows, fold, ZoneInfo('Asia/Shanghai'))
    assert [row['record_id'] for row in train] == ['safe']
    assert [row['record_id'] for row in validation] == ['valid']
    assert excluded == 1


def test_checked_in_experiment_targets_match_flow_contract():
    from training.flow_model import FLOW_LABELS

    config = json.loads((Path(__file__).resolve().parents[2] /
                         'configs/experiments/zggg-accuracy-v1.json').read_text())
    assert set(FLOW_LABELS) <= set(config['targets'])
    assert config['candidates'] == ['historical_median', 'plain_gradient_boosting',
                                    'weather_gradient_boosting']
