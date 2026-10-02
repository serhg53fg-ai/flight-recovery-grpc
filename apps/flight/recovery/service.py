"""Bind a recovery plan to immutable terminal predictions and provenance."""
import hashlib

from apps.flight.storage.documents import encode, validate_job_id
from apps.flight.domain.model_contract import validate_source
from .domain import identity, instant
from .engine import schedule, ALGORITHM_VERSION
from .validation import validate_plan
from .summary import summarize_recovery
from apps.flight.situation.snapshot import build_situation_snapshot


def build_recovery(job, config, scenario_version):
    validate_job_id(job['job_id'])
    identity(scenario_version, '场景版本')
    if job.get('status') not in {'SUCCEEDED', 'PARTIAL', 'FAILED', 'EXPIRED', 'CANCELLED'}:
        raise ValueError('预测任务必须为终态')
    results = job.get('results')
    if not isinstance(results, list) or not 1 <= len(results) <= 1000:
        raise ValueError('预测条目必须为1～1000')
    records, provenance, excluded = [], [], []
    for row in results:
        if not isinstance(row, dict) or type(row.get('success')) is not bool:
            raise ValueError('无效预测结果')
        fid = identity(row.get('flight_id'), '预测flight_id')
        if not row['success']:
            excluded.append(dict(flight_id=fid, reason=row.get('error_code') or 'PREDICTION_FAILED'))
            continue
        try:
            validate_source(row['source'])
            meta = {k: identity(row[k], k) for k in ('model_version', 'worker_id', 'trace_id')}
            p, info = row['prediction_data'], row['flight_info']
            times = [instant(p[k]) for k in ('实际离港时间', '实际起飞时间', '实际落地时间', '实际到港时间')]
            if times != sorted(times):
                raise ValueError('无效预测时序')
            records.append(dict(flight_id=fid, tail=info['机尾号'], origin=info['计划起飞站四字码'],
                                destination=info['计划到达站四字码'], planned_departure=info['计划离港时间'],
                                planned_arrival=info['计划到港时间'], predicted_departure=p['实际离港时间'],
                                predicted_arrival=p['实际到港时间']))
            provenance.append(dict(flight_id=fid, source=row['source'], **meta))
        except (KeyError, TypeError):
            raise ValueError('成功预测缺少规范化输入或来源信息') from None
    if not records:
        raise ValueError('预测任务没有可用于调度的成功结果')
    situation_snapshot = build_situation_snapshot(job, config)
    snapshot = dict(source_job_id=job['job_id'], scenario_version=scenario_version, scenario=config,
                    algorithm_version=ALGORITHM_VERSION, inputs=records, provenance=provenance, excluded=excluded)
    snapshot['situation_snapshot'] = situation_snapshot
    if job.get('submission_context'):
        context = job['submission_context']
        snapshot['source_context'] = {name: context[name] for name in (
            'replay_dataset_id', 'dataset_manifest_hash', 'feature_contract_version',
            'snapshot_hash', 'release_identity') if name in context}
    # JSON roundtrip isolates all mutable caller data before computation/storage.
    import json
    snapshot = json.loads(encode(snapshot))
    digest = hashlib.sha256(encode(snapshot).encode()).hexdigest()
    plan = schedule(snapshot['inputs'], snapshot['scenario'])
    errors = validate_plan(snapshot['inputs'], snapshot['scenario'], plan)
    result = {**snapshot, **plan, 'input_digest': digest, 'validation_errors': errors,
              'status': 'INVALID' if errors else plan['status']}
    result['summary'] = summarize_recovery(snapshot['inputs'], result)
    return result
