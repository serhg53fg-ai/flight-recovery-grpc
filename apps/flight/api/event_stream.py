"""Read-only SSE transport over consistent durable MySQL event pages."""
import json
import re
import time

from apps.flight.storage.repository import StorageError
from apps.flight.tasks.documents import summary
from apps.flight.tasks.repository import JOB_TERMINAL


def parse_cursor(headers, args):
    value = headers.get('Last-Event-ID')
    if value is None or value == '':
        value = args.get('after', '0')
    if not isinstance(value, str) or not re.fullmatch(r'0|[1-9][0-9]{0,18}', value) or int(value) > 2**63-1:
        raise ValueError('无效事件游标')
    return int(value)


def encode_event(kind, data, event_id=None):
    identity = f'id: {event_id}\n' if event_id is not None else ''
    return 'retry: 1000\n' + identity + f'event: {kind}\ndata: ' + json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n\n'


def stream_events(repository, job_id, cursor, poll_seconds=.5, heartbeat_seconds=15, max_seconds=30):
    started = heartbeat = time.monotonic()
    try:
        while time.monotonic() - started < max_seconds:
            page = repository.read_event_page(job_id, cursor)
            for event in page['events']:
                event_id = event['event_id']
                yield encode_event(event['type'], event, event_id)
                cursor = event_id
            # Metadata and sequence share the event page's database snapshot.
            # Terminal status alone cannot terminate an incomplete page.
            if page['job']['status'] in JOB_TERMINAL and cursor == page['last_event_id']:
                job = page['job']
                yield encode_event('snapshot', dict(type='snapshot', job_id=job_id, status=job['status'],
                                   summary=summary(job), sources=[job['source']] if job['success_count'] else []))
                return
            if page['events']:
                continue
            now = time.monotonic()
            if now - heartbeat >= heartbeat_seconds:
                yield ': heartbeat\n\n'
                heartbeat = now
            time.sleep(min(poll_seconds, max(0, max_seconds-(time.monotonic()-started))))
        yield ': reconnect\n\n'
    except StorageError:
        yield encode_event('dependency_error', dict(type='dependency_error', job_id=job_id,
                           error_code='STORAGE_UNAVAILABLE', error='任务存储暂不可用，请从最后游标重连'))
