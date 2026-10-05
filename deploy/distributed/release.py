"""Verify an immutable ZGGG release and expose its serving identity."""

from __future__ import annotations

import json
from pathlib import Path

from apps.flight.domain.model_contract import validate_source
from scripts.zggg_release import sha256_path


def release_identity(release: dict) -> dict:
    if release.get('schema_version') != 'zggg-release-v2':
        raise ValueError('release v1 requires explicit migration')
    mode = release.get('flight_mode')
    if mode not in ('historical', 'qwen'):
        raise ValueError('invalid flight mode')
    source = 'BASELINE' if mode == 'historical' else 'LLM'
    if release.get('source') != source:
        raise ValueError('release source does not match flight mode')
    result = {name: release.get(name) for name in (
        'release_version', 'source', 'model_version', 'prompt_version',
        'flight_model_version', 'flow_model_version', 'feature_contract_version',
        'dataset_manifest_hash')}
    if any(not isinstance(value, str) or not value or len(value) > 128 for value in result.values()):
        raise ValueError('release identity is incomplete')
    validate_source(result['source'])
    expected = f"{result['flight_model_version']}+{result['flow_model_version']}"
    if result['model_version'] != expected or len(expected) > 128:
        raise ValueError('release model version does not match components')
    if 'fallback_identity' in release:
        fallback = release['fallback_identity']
        if (mode != 'qwen' or not isinstance(fallback, dict) or
                fallback != {'source': 'BASELINE', 'flight_model_version': 'historical-flight-v1',
                             'prompt_version': 'historical-duration-v1'}):
            raise ValueError('invalid fallback identity')
        result['fallback_identity'] = dict(fallback)
    if release.get('deployment_stage') == 'experimental':
        result['deployment_stage'] = 'experimental'
        result['flight_gate_passed'] = release.get('flight_gate_passed')
    return result


def validate_experimental_release(release: dict) -> None:
    """Experimental operation is explicit and keeps the actual accuracy decision."""
    release_identity(release)
    artifacts = release.get('artifacts', {})
    if (release.get('flight_mode') != 'qwen' or not release.get('fallback_identity') or
            not {'model', 'adapter', 'accuracy_report'} <= artifacts.keys() or
            release.get('flight_gate_passed') is not False or
            release.get('flow_gate_passed') is not True or 'independent_evidence' in release):
        raise ValueError('experimental Qwen requires Adapter, fallback and failed accuracy evidence')
    record = artifacts['accuracy_report']
    try:
        path = Path(record['path'])
        if sha256_path(path) != record['sha256']:
            raise ValueError('accuracy report hash mismatch')
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('passed') is not False or not report.get('failures'):
            raise ValueError('accuracy evidence must retain the failed decision')
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('invalid accuracy evidence') from error


def load_release(path: Path, *, verify_artifacts: bool = True) -> dict:
    try:
        release = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError('invalid release manifest') from error
    release_identity(release)
    stage = release.get('deployment_stage', 'validated')
    if stage == 'experimental':
        validate_experimental_release(release)
    elif stage != 'validated':
        raise ValueError('invalid deployment stage')
    if (release.get('airport') != 'ZGGG' or release.get('flow_gate_passed') is not True or
            (stage != 'experimental' and release.get('flight_gate_passed') is not True)):
        raise ValueError('release is not approved for ZGGG')
    artifacts = release.get('artifacts')
    required = {'historical_model', 'flow_model', 'feature_contract'}
    if release['flight_mode'] == 'qwen':
        required.add('model')
    if not isinstance(artifacts, dict) or not required <= artifacts.keys():
        raise ValueError('release artifacts are incomplete')
    for name, record in artifacts.items():
        if not isinstance(record, dict) or not record.get('sha256'):
            raise ValueError(f'artifact hash is missing: {name}')
        artifact_path = Path(record.get('path', '')).resolve()
        if not artifact_path.exists():
            raise ValueError(f'artifact is missing: {name}')
        if verify_artifacts and sha256_path(artifact_path) != record['sha256']:
            raise ValueError(f'artifact hash mismatch: {name}')
    return release
