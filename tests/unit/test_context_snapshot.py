import hashlib
import json

import pandas as pd
import pytest


FLIGHT = {
    '航班号': 'CZ1234', '机尾号': 'B1234', '机型': 'A320',
    '计划起飞站四字码': 'ZGGG', '计划到达站四字码': 'ZSPD',
    '计划离港时间': '2025-05-02T00:30:00Z',
    '计划到港时间': '2025-05-02T02:20:00Z',
}


def replay_dataset(tmp_path):
    from apps.flight.context.replay import load_replay_dataset

    root = tmp_path / 'datasets'; directory = root / 'may-v1'; directory.mkdir(parents=True)
    rows = [
        {**FLIGHT, '实际起飞时间': '2025-05-02T00:42:00Z',
         '实际落地时间': '2025-05-02T02:10:00Z'},
        {'航班号': 'MU9999', '机尾号': 'B9999', '机型': 'A320',
         '计划起飞站四字码': 'ZSPD', '计划到达站四字码': 'ZGGG',
         '计划离港时间': '2025-05-01T22:00:00Z',
         '计划到港时间': '2025-05-02T00:45:00Z',
         '实际起飞时间': '2025-05-01T22:20:00Z',
         '实际落地时间': '2025-05-02T00:48:00Z'},
    ]
    source = directory / 'flights.csv'; pd.DataFrame(rows).to_csv(source, index=False)
    weather = directory / 'weather.csv'
    pd.DataFrame([
        {'kind': 'METAR', 'airport': 'ZGGG', 'issue_time': '2025-05-02T00:10:00Z',
         'received_time': '2025-05-02T00:12:00Z',
         'report_text': 'METAR ZGGG 020010Z 08005KT CAVOK 25/18 Q1012', 'source': 'test'},
        {'kind': 'METAR', 'airport': 'ZGGG', 'issue_time': '2025-05-02T00:20:00Z',
         'received_time': '2025-05-02T00:40:00Z',
         'report_text': 'METAR ZGGG 020020Z 10009KT CAVOK 25/18 Q1012', 'source': 'test'},
    ]).to_csv(weather, index=False)
    manifest = {
        'schema_version': 'zggg-replay-v1', 'airport': 'ZGGG', 'dataset_id': 'may-v1',
        'source_file': 'flights.csv', 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'weather_file': 'weather.csv', 'weather_sha256': hashlib.sha256(weather.read_bytes()).hexdigest(),
        'dataset_manifest_hash': 'dataset-hash-v1',
        'feature_contract_version': 'zggg-airport-flow-v1',
        'coverage_start': '2025-05-01T23:00:00Z',
        'coverage_end': '2025-05-02T02:00:00Z',
        'schedule_coverage': 'complete_zggg_schedule_declared',
        'schedule_as_of_known': False,
    }
    (directory / 'replay.json').write_text(json.dumps(manifest))
    return load_replay_dataset('may-v1', root), directory


def test_future_actuals_do_not_change_snapshot(tmp_path):
    from apps.flight.context.snapshot import prepare_replay_submission

    dataset, _ = replay_dataset(tmp_path)
    first = prepare_replay_submission([FLIGHT], 'Asia/Shanghai', dataset)
    changed = {**dataset, 'frame': dataset['frame'].copy()}
    changed['frame'].loc[0, '实际起飞时间'] = '2025-05-02T01:20:00Z'
    changed['frame'].loc[1, '实际落地时间'] = '2025-05-02T01:25:00Z'
    second = prepare_replay_submission([FLIGHT], 'Asia/Shanghai', changed)
    assert first['submission_context']['snapshot_hash'] == second['submission_context']['snapshot_hash']
    assert first['flights'] == second['flights']
    assert '实际起飞时间' not in first['flights'][0]


def test_completed_event_before_cutoff_changes_snapshot(tmp_path):
    from apps.flight.context.snapshot import prepare_replay_submission

    dataset, _ = replay_dataset(tmp_path)
    first = prepare_replay_submission([FLIGHT], 'Asia/Shanghai', dataset)
    changed = {**dataset, 'frame': dataset['frame'].copy()}
    changed['frame'].loc[1, '实际落地时间'] = '2025-05-02T00:25:00Z'
    second = prepare_replay_submission([FLIGHT], 'Asia/Shanghai', changed)
    assert first['submission_context']['snapshot_hash'] != second['submission_context']['snapshot_hash']
    assert second['flights'][0]['zggg_context']['flow_features'][0]['completed_landing'] == 1


def test_late_weather_excluded_and_received_basis_recorded(tmp_path):
    from apps.flight.context.snapshot import prepare_replay_submission

    dataset, _ = replay_dataset(tmp_path)
    result = prepare_replay_submission([FLIGHT], 'Asia/Shanghai', dataset)
    report = result['flights'][0]['zggg_context']['departure_weather']['metar']
    assert report['issue_time'] == '2025-05-02T00:10:00Z'
    assert result['submission_context']['snapshots'][0]['weather_availability_basis']['departure_metar'] == 'received_time'
    assert result['submission_context']['snapshots'][0]['observed_event_availability_basis'] == 'event_time_assumed'


def test_unknown_weather_received_time_is_labeled_assumption(tmp_path):
    from apps.flight.context.snapshot import prepare_replay_submission

    dataset, _ = replay_dataset(tmp_path)
    dataset = {**dataset, 'weather': {**dataset['weather']}}
    old_report = dataset['weather']['ZGGG'][0]
    dataset['weather']['ZGGG'] = [(old_report[0], None)]
    result = prepare_replay_submission([FLIGHT], 'Asia/Shanghai', dataset)
    assert result['submission_context']['snapshots'][0]['weather_availability_basis']['departure_metar'] == 'issue_time_assumed'


def test_incomplete_coverage_rejected_before_prediction(tmp_path):
    from apps.flight.context.snapshot import prepare_replay_submission
    from apps.flight.domain.prediction import parse_time

    dataset, _ = replay_dataset(tmp_path)
    dataset = {**dataset, 'coverage_end': parse_time('2025-05-02T00:45:00Z')}
    with pytest.raises(ValueError, match='覆盖'):
        prepare_replay_submission([FLIGHT], 'Asia/Shanghai', dataset)


def test_replay_source_hash_mismatch_rejected(tmp_path):
    from apps.flight.context.replay import load_replay_dataset

    _, directory = replay_dataset(tmp_path)
    with (directory / 'flights.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='hash'):
        load_replay_dataset('may-v1', directory.parent)


def test_weather_archive_issue_time_mismatch_rejected(tmp_path):
    from apps.flight.context.replay import load_replay_dataset

    _, directory = replay_dataset(tmp_path)
    weather = directory / 'weather.csv'
    frame = pd.read_csv(weather)
    frame.loc[0, 'issue_time'] = '2025-05-02T00:01:00Z'
    frame.to_csv(weather, index=False)
    manifest_path = directory / 'replay.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['weather_sha256'] = hashlib.sha256(weather.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='issue time'):
        load_replay_dataset('may-v1', directory.parent)
