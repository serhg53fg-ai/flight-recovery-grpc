"""Consistent read snapshots for HTTP and resumable event consumers."""
import json

from apps.flight.storage.documents import validate_job_id, validate_limit
from .documents import bounded_integer


def read_event_page(repository, job_id, after=0, limit=100):
    validate_job_id(job_id)
    bounded_integer(after, 0, 2**63-1, '事件游标')
    validate_limit(limit)
    with repository._session() as cursor:
        cursor.execute('SELECT document,event_seq FROM durable_jobs WHERE job_id=%s', (job_id,))
        row = cursor.fetchone()
        if not row:
            raise KeyError('没有找到持久任务')
        if after > row['event_seq']:
            raise ValueError('事件游标超过当前任务事件序号')
        cursor.execute('SELECT event_id,document FROM durable_events WHERE job_id=%s AND event_id>%s '
                       'ORDER BY event_id LIMIT %s', (job_id, after, limit))
        events = [dict(event_id=r['event_id'], **json.loads(r['document'])) for r in cursor.fetchall()]
        return dict(job=json.loads(row['document']), last_event_id=row['event_seq'], events=events)


def list_jobs(repository, limit=50):
    validate_limit(limit)
    with repository._session() as cursor:
        cursor.execute("SELECT document FROM durable_jobs ORDER BY JSON_EXTRACT(document,'$.created_at') DESC,job_id DESC LIMIT %s", (limit,))
        return [{k: v for k, v in json.loads(r['document']).items() if k != 'results'} for r in cursor.fetchall()]
