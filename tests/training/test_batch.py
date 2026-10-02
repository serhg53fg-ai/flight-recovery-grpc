import json
import pytest
from training.dataset import build_dataset, DatasetConfig
from tests.training.test_dataset import row, source


def test_batch_export_keeps_failures_and_filters_features(tmp_path):
    from training.batch import export_predictions
    dataset = tmp_path / 'dataset'
    build_dataset(source(tmp_path, [row(day) for day in range(1, 13)]), dataset,
                  DatasetConfig(min_test_rows=1))
    calls = []
    def predict(features):
        calls.append(features)
        if len(calls) == 1:
            raise ValueError('private input must not be logged')
        return dict(off_block_delay_min=0, taxi_out_min=10, airborne_min=100, taxi_in_min=5)
    result = export_predictions(dataset, tmp_path / 'run', predict, {'kind': 'test'})
    assert result['success_count'] == len(calls) - 1
    assert result['failure_count'] == 1
    assert 'labels' not in calls[0]
    failure = json.loads((tmp_path / 'run/failures.jsonl').read_text())
    assert failure['error_type'] == 'ValueError'
    assert 'private' not in json.dumps(failure)
    assert result['dataset_manifest_hash']
    assert result['prompt_version'] == 'flight-duration-prompt-v1'
    with pytest.raises(ValueError):
        export_predictions(dataset, tmp_path / 'run', predict, {'kind': 'test'})


def test_zggg_batch_preserves_cutoff_weather_for_v2_prompt(tmp_path):
    from training.batch import export_predictions
    from training.prompt import PROMPT_VERSION
    from training.zggg_dataset import build_zggg_dataset

    rows = [row(day) for day in range(1, 13)]
    for day, item in enumerate(rows, start=1):
        issue_day = 30 if day == 1 else day - 1
        item['起飞站METAR'] = (
            f'METAR ZGGG {issue_day:02d}2350Z 18005KT 9999 CLR 20/10 Q1013='
        )
    dataset = tmp_path / 'zggg'
    manifest = build_zggg_dataset(source(tmp_path, rows), dataset,
                                  DatasetConfig(min_test_rows=1))
    received = []

    def predict(features):
        received.append(features)
        return dict(off_block_delay_min=0, taxi_out_min=10,
                    airborne_min=100, taxi_in_min=5)

    run = export_predictions(dataset, tmp_path / 'run', predict, {'kind': 'test'})

    assert run['dataset_manifest_hash'] == manifest.manifest_hash
    assert run['prompt_version'] == PROMPT_VERSION
    assert received[0]['airport_direction'] == 'DEPARTURE'
    assert received[0]['departure_weather']['metar']['missing'] is False
    assert 'labels' not in received[0]
