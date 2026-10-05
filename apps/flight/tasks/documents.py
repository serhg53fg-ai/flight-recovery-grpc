"""Immutable submission snapshots and identities for durable execution."""
import hashlib
from uuid import uuid4

from google.protobuf.json_format import MessageToDict

from apps.flight.domain.prediction import normalize_flight, normalize_zggg_context, timezone_for
from apps.flight.domain.model_contract import validate_source
from apps.flight.services.prediction import serializable
from apps.flight.storage.documents import encode, new_job


class IdempotencyConflict(ValueError):
    pass


class AdmissionFull(ValueError):
    pass


def bounded_integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'无效{label}')
    return value


def label(value, name, maximum=128):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or '\x00' in value:
        raise ValueError(f'无效{name}')
    return value


def prepare(flights, timezone, key, model_version, source, prompt_version, ttl_seconds, max_attempts,
            *, submission_context: dict | None = None):
    if not isinstance(flights, list) or not 1 <= len(flights) <= 1000:
        raise ValueError('航班数组长度必须为1～1000')
    timezone_for(timezone)
    label(key, '幂等键')
    label(model_version, '模型版本')
    label(prompt_version, 'Prompt版本')
    validate_source(source)
    bounded_integer(ttl_seconds, 1, 3600, '任务期限')
    bounded_integer(max_attempts, 1, 5, '领取次数')
    raw = serializable(flights)
    frozen_context = None if submission_context is None else serializable(submission_context)
    identity = [raw, timezone, model_version, source, prompt_version, ttl_seconds, max_attempts]
    if frozen_context is not None:
        identity.append(frozen_context)
    digest = hashlib.sha256(encode(identity).encode()).hexdigest()
    job = new_job(len(flights), timezone)
    job.update(status='QUEUED', model_version=model_version, source=source,
               prompt_version=prompt_version, max_attempts=max_attempts, expired_count=0,
               cancelled_count=0, cancel_requested=False, mode='durable')
    if frozen_context is not None:
        job['submission_context'] = frozen_context
        identity = frozen_context.get('release_identity') or {}
        if identity.get('fallback_identity'):
            job.update(fallback_enabled=True,
                       fallback_reason='主模型运行失败时使用发布清单声明的历史模型；结果记录实际来源')
    records = []
    for index, data in enumerate(raw, 1):
        base = dict(index=index, job_id=job['job_id'], flight_id=str(uuid4()),
                    flight_info=data if isinstance(data, dict) else {}, success=False, degraded=False)
        normalized = None
        normalized_context = None
        try:
            flight = normalize_flight(data, timezone)
            flight.flight_id = base['flight_id']
            normalized = MessageToDict(flight, preserving_proto_field_name=True)
            context = normalize_zggg_context(data, flight, timezone)
            normalized_context = (MessageToDict(context, preserving_proto_field_name=True)
                                  if context is not None else None)
        except ValueError as exc:
            base.update(trace_id=str(uuid4()), error_code='INVALID_ARGUMENT', error=str(exc))
        records.append(dict(base_result=base, normalized_input=normalized,
                            normalized_context=normalized_context))
    return job, records, digest


def summary(job):
    return dict(total=job['total_count'], completed=job['completed_count'], success=job['success_count'],
                failed=job['failed_count'], cancelled=job.get('cancelled_count', 0), expired=job['expired_count'])
