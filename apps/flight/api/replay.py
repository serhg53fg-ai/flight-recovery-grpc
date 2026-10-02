"""Server-owned replay dataset selection for durable HTTP submissions."""

from pathlib import Path

from flask import Blueprint, current_app, jsonify

from apps.flight.context.replay import load_replay_dataset
from apps.flight.context.snapshot import prepare_replay_submission


replay = Blueprint('replay', __name__, url_prefix='/api/v1')


def prepare_submission(flights, timezone, requested_id=None):
    config = current_app.config
    configured_id = config.get('REPLAY_DATASET_ID')
    if not configured_id:
        if requested_id is not None:
            raise ValueError('历史回放数据集尚未配置')
        return flights, None
    dataset_id = requested_id or configured_id
    dataset = load_replay_dataset(dataset_id, Path(config['REPLAY_ROOT']))
    release = config['RELEASE_IDENTITY']
    if dataset['dataset_manifest_hash'] != release['dataset_manifest_hash']:
        raise ValueError('回放数据集与发布模型的数据版本不匹配')
    if dataset['feature_contract_version'] != release['feature_contract_version']:
        raise ValueError('回放数据集与发布模型的特征契约不匹配')
    if not isinstance(flights, list) or any(not isinstance(item, dict) or 'zggg_context' in item for item in flights):
        raise ValueError('公开回放输入只允许普通航班字段，不接受内部上下文')
    prepared = prepare_replay_submission(flights, timezone, dataset)
    prepared['submission_context']['release_identity'] = release
    return prepared['flights'], prepared['submission_context']


@replay.get('/replay-datasets')
def datasets():
    config = current_app.config
    dataset_id = config.get('REPLAY_DATASET_ID')
    if not dataset_id:
        return jsonify(datasets=[])
    root = Path(config['REPLAY_ROOT'])
    available = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / 'replay.json').is_file():
            continue
        try:
            dataset = load_replay_dataset(directory.name, root)
        except ValueError:
            continue
        if (dataset['dataset_manifest_hash'] != config['RELEASE_IDENTITY']['dataset_manifest_hash'] or
                dataset['feature_contract_version'] != config['RELEASE_IDENTITY']['feature_contract_version']):
            continue
        available.append({'dataset_id': directory.name, 'airport': 'ZGGG',
                          'coverage_start': dataset['coverage_start'].isoformat(),
                          'coverage_end': dataset['coverage_end'].isoformat(),
                          'schedule_as_of_known': dataset['schedule_as_of_known']})
    return jsonify(datasets=available)
