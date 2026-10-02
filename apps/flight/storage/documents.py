"""Shared job document invariants for persistent repositories."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

from apps.flight.domain.prediction import timezone_for

FINAL_STATES = frozenset({'SUCCEEDED', 'PARTIAL', 'FAILED'})


def validate_job_id(job_id):
    try:
        if not isinstance(job_id, str) or str(UUID(job_id)) != job_id:
            raise ValueError()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError('无效 job_id') from exc
    return job_id


def validate_limit(limit):
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('任务列表上限必须为1—1000')
    return limit


def encode(document):
    return json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True)


def new_job(total, input_timezone):
    if type(total) is not int or not 1 <= total <= 10000:
        raise ValueError('任务航班数量必须为1—10000')
    timezone_for(input_timezone)
    now = datetime.now(timezone.utc).isoformat()
    return dict(job_id=str(uuid4()), created_at=now, updated_at=now,
                status='RUNNING', total_count=total, success_count=0,
                failed_count=0, completed_count=0, llm_success_count=0,
                input_timezone=input_timezone, results=[], fallback_enabled=False,
                fallback_reason='传统模型特征和目标单位尚未通过验证，自动回退关闭')


def _summary(results):
    if not isinstance(results, list) or any(not isinstance(r, dict) or
            not isinstance(r.get('success', False), bool) for r in results):
        raise ValueError('结果必须为包含布尔success的记录列表')
    success = sum(r.get('success', False) for r in results)
    status = 'SUCCEEDED' if success == len(results) else ('PARTIAL' if success else 'FAILED')
    return dict(status=status, success_count=success, failed_count=len(results)-success,
                completed_count=len(results),
                llm_success_count=sum(r.get('success', False) and r.get('source') == 'LLM' for r in results))


def completed_job(job, results):
    if not isinstance(results, list) or len(results) != job['total_count']:
        raise ValueError('最终结果数量必须等于输入数量')
    if job['status'] in FINAL_STATES:
        if encode(job['results']) != encode(results):
            raise ValueError('终态任务不能用不同结果覆盖')
        return job
    if job['status'] != 'RUNNING':
        raise ValueError('任务状态不允许完成')
    final = {**job, **_summary(results), 'results': results,
             'updated_at': datetime.now(timezone.utc).isoformat()}
    return validate_document(final)


def validate_document(document):
    if not isinstance(document, dict):
        raise ValueError('无效任务文档')
    try:
        validate_job_id(document['job_id'])
        total = document['total_count']
        if type(total) is not int or not 1 <= total <= 10000:
            raise ValueError('无效任务数量')
        timezone_for(document['input_timezone'])
        for key in ('created_at', 'updated_at'):
            if datetime.fromisoformat(document[key]).tzinfo is None:
                raise ValueError('任务时间必须包含时区')
        if document['status'] == 'RUNNING':
            expected = dict(success_count=0, failed_count=0, completed_count=0, llm_success_count=0)
            if document['results'] != []:
                raise ValueError('运行中任务不应导入未提交结果')
        elif document['status'] in FINAL_STATES:
            if len(document['results']) != total:
                raise ValueError('任务结果数量不一致')
            expected = _summary(document['results'])
        else:
            raise ValueError('无效任务状态')
        for key, value in expected.items():
            if document[key] != value or (key != 'status' and type(document[key]) is not int):
                raise ValueError('任务状态或统计不一致')
        return json.loads(encode(document))
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('任务文档字段不完整或非法') from exc
