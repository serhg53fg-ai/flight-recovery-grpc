"""MySQL is authoritative: locks, leases, fencing, results and outbox."""
from contextlib import contextmanager
import json
from pathlib import Path
from uuid import uuid4

from pymysql.cursors import DictCursor

from apps.flight.storage.documents import encode, validate_job_id, validate_limit
from apps.flight.storage.mysql_jobs import MySQLJobStore, _transaction
from apps.flight.storage.repository import StorageError
from .documents import bounded_integer, label, prepare, summary, IdempotencyConflict, AdmissionFull

TERMINAL = frozenset(('SUCCEEDED', 'FAILED', 'EXPIRED', 'CANCELLED'))
JOB_TERMINAL = TERMINAL | {'PARTIAL'}


def _validate(cursor, database):
    expected = {
        'durable_schema': ['singleton', 'version'], 'queue_admission': ['singleton', 'capacity'],
        'durable_jobs': ['job_id', 'submission_key', 'input_digest', 'deadline_at', 'event_seq', 'document'],
        'durable_flights': ['flight_id', 'job_id', 'ordinal', 'status', 'fence', 'owner', 'trace_id',
                            'lease_until', 'input_document', 'result_document'],
        'durable_attempts': ['flight_id', 'fence', 'trace_id', 'owner', 'status', 'started_at', 'finished_at'],
        'durable_events': ['job_id', 'event_id', 'document'],
        'prediction_outbox': ['message_id', 'job_id', 'flight_id', 'published_at', 'failures', 'next_publish_at'],
    }
    cursor.execute('SELECT version FROM durable_schema WHERE singleton=1')
    if cursor.fetchone() != {'version': 1}:
        raise StorageError('持久执行schema版本不兼容')
    cursor.execute('SELECT TABLE_NAME,ENGINE FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s', (database,))
    engines = {r['TABLE_NAME']: r['ENGINE'] for r in cursor.fetchall()}
    for table, columns in expected.items():
        if engines.get(table) != 'InnoDB':
            raise StorageError('持久执行表引擎不兼容')
        cursor.execute('SELECT COLUMN_NAME,COLUMN_TYPE,IS_NULLABLE,COLLATION_NAME FROM information_schema.COLUMNS '
                       'WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s', (database, table))
        actual = {r['COLUMN_NAME']: r for r in cursor.fetchall()}
        if not set(columns) <= set(actual):
            raise StorageError('持久执行表结构不兼容')
        types = dict(singleton='tinyint', version='int', capacity='int unsigned',
                     submission_key='varchar(128)', input_digest='char(64)', deadline_at='datetime(6)',
                     event_seq='bigint unsigned', event_id='bigint unsigned', ordinal='int unsigned',
                     status='varchar(16)', fence='int unsigned', owner='varchar(128)',
                     failures='int unsigned')
        for column in columns:
            kind = ('json' if column.endswith('document') else 'char(36)' if column.endswith('_id')
                    else 'datetime(6)' if column.endswith('_at') or column == 'lease_until' else types.get(column))
            # event_id is numeric, unlike UUID identities.
            if column == 'event_id':
                kind = 'bigint unsigned'
            nullable = column in {'lease_until', 'result_document', 'published_at', 'finished_at'} or (
                table == 'durable_flights' and column in {'owner', 'trace_id'})
            if actual[column]['COLUMN_TYPE'] != kind or actual[column]['IS_NULLABLE'] != ('YES' if nullable else 'NO'):
                raise StorageError('持久执行字段类型不兼容')
            if column in {'job_id', 'flight_id', 'message_id'} and actual[column]['COLLATION_NAME'] != 'ascii_bin':
                raise StorageError('持久执行ID排序规则不兼容')
            if column in {'submission_key', 'owner'} and actual[column]['COLLATION_NAME'] != 'utf8mb4_bin':
                raise StorageError('持久执行身份排序规则不兼容')
    cursor.execute('SELECT TABLE_NAME,INDEX_NAME,NON_UNIQUE,COLUMN_NAME,SUB_PART FROM information_schema.STATISTICS '
                   'WHERE TABLE_SCHEMA=%s ORDER BY TABLE_NAME,INDEX_NAME,SEQ_IN_INDEX', (database,))
    indexes = {}
    for r in cursor.fetchall():
        indexes.setdefault((r['TABLE_NAME'], r['INDEX_NAME'], r['NON_UNIQUE']), []).append((r['COLUMN_NAME'], r['SUB_PART']))
    required = {
        ('durable_schema', 'PRIMARY', 0): ['singleton'], ('queue_admission', 'PRIMARY', 0): ['singleton'],
        ('durable_jobs', 'PRIMARY', 0): ['job_id'], ('durable_jobs', 'submission_key', 0): ['submission_key'],
        ('durable_flights', 'PRIMARY', 0): ['flight_id'], ('durable_flights', 'flight_order', 0): ['job_id', 'ordinal'],
        ('durable_flights', 'flight_recovery', 1): ['status', 'lease_until'],
        ('durable_attempts', 'PRIMARY', 0): ['flight_id', 'fence'], ('durable_attempts', 'trace_id', 0): ['trace_id'],
        ('durable_events', 'PRIMARY', 0): ['job_id', 'event_id'],
        ('prediction_outbox', 'PRIMARY', 0): ['message_id'], ('prediction_outbox', 'flight_id', 0): ['flight_id'],
        ('prediction_outbox', 'outbox_due', 1): ['published_at', 'next_publish_at'],
    }
    if any(indexes.get(key) != [(c, None) for c in columns] for key, columns in required.items()):
        raise StorageError('持久执行索引或唯一约束不兼容')
    cursor.execute('SELECT k.TABLE_NAME,k.COLUMN_NAME,k.REFERENCED_TABLE_NAME,k.REFERENCED_COLUMN_NAME,r.DELETE_RULE '
                   'FROM information_schema.KEY_COLUMN_USAGE k JOIN information_schema.REFERENTIAL_CONSTRAINTS r '
                   'ON k.CONSTRAINT_SCHEMA=r.CONSTRAINT_SCHEMA AND k.TABLE_NAME=r.TABLE_NAME '
                   'AND k.CONSTRAINT_NAME=r.CONSTRAINT_NAME WHERE k.TABLE_SCHEMA=%s', (database,))
    foreign_keys = {(r['TABLE_NAME'], r['COLUMN_NAME'], r['REFERENCED_TABLE_NAME'], r['REFERENCED_COLUMN_NAME'], r['DELETE_RULE']) for r in cursor.fetchall()}
    for table, column, parent in [('durable_flights', 'job_id', 'durable_jobs'),
                                   ('durable_attempts', 'flight_id', 'durable_flights'),
                                   ('durable_events', 'job_id', 'durable_jobs'),
                                   ('prediction_outbox', 'job_id', 'durable_jobs'),
                                   ('prediction_outbox', 'flight_id', 'durable_flights')]:
        if (table, column, parent, column, 'CASCADE') not in foreign_keys:
            raise StorageError('持久执行外键约束不兼容')
    cursor.execute('SELECT capacity FROM queue_admission WHERE singleton=1')
    row = cursor.fetchone()
    if not row or not 1 <= row['capacity'] <= 100000:
        raise StorageError('持久执行积压配置不兼容')


def initialize_durable(config):
    MySQLJobStore(config)
    statements = Path(__file__).with_name('schema.sql').read_text().split(';')
    with _transaction(config) as conn, conn.cursor(DictCursor) as cursor:
        for statement in statements:
            if statement.strip():
                cursor.execute(statement)
        cursor.execute('INSERT INTO durable_schema VALUES (1,1) ON DUPLICATE KEY UPDATE singleton=singleton')
        cursor.execute('INSERT INTO queue_admission VALUES (1,1000) ON DUPLICATE KEY UPDATE singleton=singleton')
        _validate(cursor, config.database)


class DurableRepository:
    def __init__(self, config):
        self.config = config
        MySQLJobStore(config)
        with self._session() as cursor:
            _validate(cursor, config.database)

    @contextmanager
    def _session(self):
        with _transaction(self.config) as conn, conn.cursor(DictCursor) as cursor:
            yield cursor

    def ping(self):
        with self._session() as cursor:
            cursor.execute('SELECT version FROM durable_schema WHERE singleton=1')
            return cursor.fetchone() == {'version': 1}

    def record_stage(self, claim, stage, seconds, outcome):
        from apps.flight.observability import record_stage
        record_stage(stage, seconds, outcome)
        with self._session() as cursor:
            self._event(cursor, claim['job_id'], 'stage', flight_id=claim['flight_id'],
                        trace_id=claim['trace_id'], attempt=claim['fence'],
                        stage=stage, seconds=float(seconds), outcome=outcome)

    def stage_observations(self):
        with self._session() as cursor:
            cursor.execute("SELECT document FROM durable_events WHERE "
                           "JSON_UNQUOTE(JSON_EXTRACT(document,'$.type'))='stage'")
            return [json.loads(row['document']) for row in cursor.fetchall()]

    @staticmethod
    def _event(cursor, job_id, kind, **payload):
        cursor.execute('UPDATE durable_jobs SET event_seq=event_seq+1 WHERE job_id=%s', (job_id,))
        cursor.execute('SELECT event_seq FROM durable_jobs WHERE job_id=%s', (job_id,))
        sequence = cursor.fetchone()['event_seq']
        cursor.execute('INSERT INTO durable_events VALUES (%s,%s,%s)',
                       (job_id, sequence, encode(dict(type=kind, job_id=job_id, **payload))))

    def _complete_event(self, cursor, job):
        if job['status'] in JOB_TERMINAL:
            self._event(cursor, job['job_id'], 'complete', status=job['status'], summary=summary(job),
                        sources=job.get('sources', []))

    @staticmethod
    def _refresh(cursor, job_id):
        cursor.execute('SELECT document FROM durable_jobs WHERE job_id=%s', (job_id,))
        job = json.loads(cursor.fetchone()['document'])
        cursor.execute('SELECT status,fence,result_document FROM durable_flights WHERE job_id=%s', (job_id,))
        flights = cursor.fetchall()
        successful_results = [json.loads(f['result_document']) for f in flights
                              if f['status'] == 'SUCCEEDED' and f['result_document']]
        success = sum(f['status'] == 'SUCCEEDED' for f in flights)
        failed = sum(f['status'] == 'FAILED' for f in flights)
        expired = sum(f['status'] == 'EXPIRED' for f in flights)
        cancelled = sum(f['status'] == 'CANCELLED' for f in flights)
        complete = success + failed + expired + cancelled
        status = ('RUNNING' if any(f['fence'] for f in flights) else 'QUEUED') if complete < len(flights) else (
            'CANCELLED' if cancelled else 'SUCCEEDED' if success == len(flights) else 'PARTIAL' if success else
            'EXPIRED' if expired == len(flights) else 'FAILED')
        job.update(status=status, success_count=success, failed_count=failed, expired_count=expired,
                   cancelled_count=cancelled, completed_count=complete,
                   llm_success_count=sum(r.get('source') == 'LLM' for r in successful_results),
                   sources=sorted({r['source'] for r in successful_results}),
                   fallback_count=sum(bool(r.get('flight_degraded') or r.get('flow_degraded'))
                                      for r in successful_results))
        cursor.execute('SELECT UTC_TIMESTAMP(6) AS now')
        job['updated_at'] = cursor.fetchone()['now'].isoformat() + '+00:00'
        cursor.execute('UPDATE durable_jobs SET document=%s WHERE job_id=%s', (encode(job), job_id))
        return job

    def submit(self, flights, timezone, key, model_version, source, prompt_version, ttl_seconds=300, max_attempts=3,
               *, submission_context: dict | None = None):
        job, records, digest = prepare(flights, timezone, key, model_version, source, prompt_version,
                                       ttl_seconds, max_attempts, submission_context=submission_context)
        with self._session() as cursor:
            # First DB lock also serializes idempotent-key and capacity decisions.
            cursor.execute('SELECT capacity FROM queue_admission WHERE singleton=1 FOR UPDATE')
            capacity = cursor.fetchone()['capacity']
            cursor.execute('SELECT job_id,input_digest FROM durable_jobs WHERE submission_key=%s', (key,))
            old = cursor.fetchone()
            if old:
                if old['input_digest'] != digest:
                    raise IdempotencyConflict('幂等键已绑定不同输入或执行配置')
                job_id = old['job_id']
            else:
                pending = sum(r['normalized_input'] is not None for r in records)
                cursor.execute("SELECT COUNT(*) AS n FROM durable_flights WHERE status IN ('QUEUED','RUNNING')")
                if cursor.fetchone()['n'] + pending > capacity:
                    raise AdmissionFull('持久任务积压达到上限，请稍后提交')
                job_id = job['job_id']
                cursor.execute('INSERT INTO durable_jobs (job_id,submission_key,input_digest,deadline_at,document) '
                               'VALUES (%s,%s,%s,UTC_TIMESTAMP(6)+INTERVAL %s SECOND,%s)',
                               (job_id, key, digest, ttl_seconds, encode(job)))
                for index, record in enumerate(records, 1):
                    flight_id = record['base_result']['flight_id']
                    valid = record['normalized_input'] is not None
                    cursor.execute('INSERT INTO durable_flights (job_id,flight_id,ordinal,status,input_document,result_document) '
                                   'VALUES (%s,%s,%s,%s,%s,%s)',
                                   (job_id, flight_id, index, 'QUEUED' if valid else 'FAILED', encode(record),
                                    None if valid else encode(record['base_result'])))
                    if valid:
                        cursor.execute('INSERT INTO prediction_outbox (message_id,job_id,flight_id,next_publish_at) '
                                       'VALUES (%s,%s,%s,UTC_TIMESTAMP(6))', (str(uuid4()), job_id, flight_id))
                refreshed = self._refresh(cursor, job_id)
                self._event(cursor, job_id, 'submitted')
                for record in records:
                    if record['normalized_input'] is None:
                        self._event(cursor, job_id, 'result', result=record['base_result'], progress=summary(refreshed))
                self._complete_event(cursor, refreshed)
        return self.get(job_id)

    def get(self, job_id):
        validate_job_id(job_id)
        with self._session() as cursor:
            # Consistent read snapshot includes summary and results together.
            cursor.execute('SELECT document,deadline_at,event_seq FROM durable_jobs WHERE job_id=%s', (job_id,))
            row = cursor.fetchone()
            if not row:
                raise KeyError('没有找到持久任务')
            job = json.loads(row['document'])
            job.setdefault('cancelled_count', 0)
            job.setdefault('cancel_requested', False)
            job.update(mode='durable', last_event_id=row['event_seq'])
            job['deadline_at'] = row['deadline_at'].isoformat() + '+00:00'
            cursor.execute('SELECT flight_id,ordinal,status,fence,input_document,result_document '
                           'FROM durable_flights WHERE job_id=%s ORDER BY ordinal', (job_id,))
            job['flights'] = [dict(flight_id=r['flight_id'], index=r['ordinal'], status=r['status'],
                                   fence=r['fence'], **json.loads(r['input_document']),
                                   result=json.loads(r['result_document']) if r['result_document'] else None)
                              for r in cursor.fetchall()]
            job['results'] = [f['result'] for f in job['flights'] if f['result'] is not None]
            return job

    @staticmethod
    def _lock(cursor, job_id, flight_id):
        validate_job_id(job_id)
        validate_job_id(flight_id)
        cursor.execute('SELECT document,deadline_at FROM durable_jobs WHERE job_id=%s FOR UPDATE', (job_id,))
        job = cursor.fetchone()
        if not job:
            raise KeyError('没有找到持久任务')
        cursor.execute('SELECT * FROM durable_flights '
                       'WHERE job_id=%s AND flight_id=%s FOR UPDATE', (job_id, flight_id))
        flight = cursor.fetchone()
        if not flight:
            raise KeyError('没有找到持久航班')
        # MySQL NOW/UTC_TIMESTAMP is fixed at statement start, before a lock
        # wait. Evaluate time only after both row locks have been acquired.
        cursor.execute('SELECT UTC_TIMESTAMP(6) AS now')
        now = cursor.fetchone()['now']
        flight['leased'] = flight['lease_until'] is not None and flight['lease_until'] > now
        return json.loads(job['document']), (job['deadline_at'] - now).total_seconds(), flight

    def _fail(self, cursor, job_id, flight, code, status='FAILED'):
        base = json.loads(flight['input_document'])['base_result']
        result = {**base, 'trace_id': flight['trace_id'] or str(uuid4()), 'error_code': code,
                  'error': '任务期限已到' if status == 'EXPIRED' else '执行器失联且恢复次数已耗尽'}
        cursor.execute('UPDATE durable_flights SET status=%s,result_document=%s,lease_until=NULL WHERE flight_id=%s',
                       (status, encode(result), flight['flight_id']))
        cursor.execute("UPDATE durable_attempts SET status=%s,finished_at=UTC_TIMESTAMP(6) "
                       "WHERE flight_id=%s AND fence=%s AND status='RUNNING'",
                       (status, flight['flight_id'], flight['fence']))
        refreshed = self._refresh(cursor, job_id)
        self._event(cursor, job_id, 'result', flight_id=flight['flight_id'], error_code=code,
                    result=result, progress=summary(refreshed))
        self._complete_event(cursor, refreshed)

    def claim(self, job_id, flight_id, owner, lease_seconds=30):
        label(owner, '执行者')
        bounded_integer(lease_seconds, 1, 300, '租约')
        with self._session() as cursor:
            job, remaining, f = self._lock(cursor, job_id, flight_id)
            if f['status'] in TERMINAL:
                return None
            if remaining <= 0:
                self._fail(cursor, job_id, f, 'DEADLINE_EXCEEDED', 'EXPIRED')
                return None
            if f['status'] == 'RUNNING' and f['leased']:
                return None
            if f['fence'] >= job['max_attempts']:
                self._fail(cursor, job_id, f, 'EXECUTOR_LOST')
                return None
            cursor.execute("UPDATE durable_attempts SET status='LOST',finished_at=UTC_TIMESTAMP(6) "
                           "WHERE flight_id=%s AND fence=%s AND status='RUNNING'", (flight_id, f['fence']))
            fence, trace = f['fence'] + 1, str(uuid4())
            cursor.execute("UPDATE durable_flights SET status='RUNNING',fence=%s,owner=%s,trace_id=%s,"
                           'lease_until=UTC_TIMESTAMP(6)+INTERVAL %s SECOND WHERE flight_id=%s',
                           (fence, owner, trace, lease_seconds, flight_id))
            cursor.execute("INSERT INTO durable_attempts (flight_id,fence,trace_id,owner,status,started_at) "
                           "VALUES (%s,%s,%s,%s,'RUNNING',UTC_TIMESTAMP(6))", (flight_id, fence, trace, owner))
            self._refresh(cursor, job_id)
            self._event(cursor, job_id, 'claimed', flight_id=flight_id, fence=fence)
            prepared = json.loads(f['input_document'])
            prepared['base_result']['trace_id'] = trace
            return dict(job_id=job_id, flight_id=flight_id, owner=owner, fence=fence, trace_id=trace,
                        remaining_seconds=remaining, job=job, **prepared)

    @staticmethod
    def _matches(f, claim):
        return f['status'] == 'RUNNING' and bool(f['leased']) and (
            f['fence'], f['owner'], f['trace_id']) == (claim['fence'], claim['owner'], claim['trace_id'])

    def heartbeat(self, claim, lease_seconds=30):
        bounded_integer(lease_seconds, 1, 300, '租约')
        with self._session() as cursor:
            _, remaining, f = self._lock(cursor, claim['job_id'], claim['flight_id'])
            if remaining <= 0 or not self._matches(f, claim):
                return False
            cursor.execute('UPDATE durable_flights SET lease_until=UTC_TIMESTAMP(6)+INTERVAL %s SECOND WHERE flight_id=%s',
                           (lease_seconds, claim['flight_id']))
            return True

    def complete(self, claim, result):
        if not isinstance(result, dict) or type(result.get('success')) is not bool:
            raise ValueError('无效执行结果')
        if any(result.get(key) != claim['base_result'][key] for key in ('job_id', 'flight_id', 'trace_id', 'index')):
            raise ValueError('执行结果身份不匹配')
        with self._session() as cursor:
            job, remaining, f = self._lock(cursor, claim['job_id'], claim['flight_id'])
            if remaining <= 0 or not self._matches(f, claim):
                return False
            if result['success']:
                from apps.flight.services.prediction import validate_serving_identity
                from apps.flight.clients.inference import PredictionError
                try:
                    validate_serving_identity(result, job)
                except (PredictionError, KeyError, TypeError, ValueError):
                    result = {**claim['base_result'], 'success': False, 'error_code': 'DATA_LOSS', 'error': '模型版本或来源不匹配'}
            status = 'SUCCEEDED' if result['success'] else 'FAILED'
            cursor.execute('UPDATE durable_flights SET status=%s,result_document=%s,lease_until=NULL WHERE flight_id=%s',
                           (status, encode(result), claim['flight_id']))
            cursor.execute('UPDATE durable_attempts SET status=%s,finished_at=UTC_TIMESTAMP(6) WHERE flight_id=%s AND fence=%s',
                           (status, claim['flight_id'], claim['fence']))
            refreshed = self._refresh(cursor, claim['job_id'])
            self._event(cursor, claim['job_id'], 'result', flight_id=claim['flight_id'], success=result['success'],
                        result=result, progress=summary(refreshed))
            self._complete_event(cursor, refreshed)
            return True

    def cancel(self, job_id):
        validate_job_id(job_id)
        with self._session() as cursor:
            cursor.execute('SELECT document FROM durable_jobs WHERE job_id=%s FOR UPDATE', (job_id,))
            row = cursor.fetchone()
            if not row:
                raise KeyError('没有找到持久任务')
            job = json.loads(row['document'])
            if job['status'] not in JOB_TERMINAL:
                cursor.execute("SELECT * FROM durable_flights WHERE job_id=%s AND status IN ('QUEUED','RUNNING') "
                               'ORDER BY ordinal FOR UPDATE', (job_id,))
                flights = cursor.fetchall()
                job['cancel_requested'] = True
                cursor.execute('UPDATE durable_jobs SET document=%s WHERE job_id=%s', (encode(job), job_id))
                self._event(cursor, job_id, 'cancel_requested')
                results = []
                for flight in flights:
                    result = {**json.loads(flight['input_document'])['base_result'],
                              'trace_id': flight['trace_id'] or str(uuid4()),
                              'success': False, 'error_code': 'CANCELLED', 'error': '用户已取消任务'}
                    cursor.execute("UPDATE durable_flights SET status='CANCELLED',result_document=%s,lease_until=NULL "
                                   'WHERE flight_id=%s', (encode(result), flight['flight_id']))
                    cursor.execute("UPDATE durable_attempts SET status='CANCELLED',finished_at=UTC_TIMESTAMP(6) "
                                   "WHERE flight_id=%s AND fence=%s AND status='RUNNING'", (flight['flight_id'], flight['fence']))
                    results.append(result)
                refreshed = self._refresh(cursor, job_id)
                for result in results:
                    self._event(cursor, job_id, 'result', result=result, progress=summary(refreshed))
                self._complete_event(cursor, refreshed)
        return self.get(job_id)

    def read_event_page(self, job_id, after=0, limit=100):
        from .queries import read_event_page
        return read_event_page(self, job_id, after, limit)

    def list_jobs(self, limit=50):
        from .queries import list_jobs
        return list_jobs(self, limit)

    def events(self, job_id, after=0):
        validate_job_id(job_id)
        bounded_integer(after, 0, 2**63-1, '事件游标')
        with self._session() as cursor:
            cursor.execute('SELECT event_id,document FROM durable_events WHERE job_id=%s AND event_id>%s ORDER BY event_id LIMIT 1000', (job_id, after))
            return [dict(event_id=r['event_id'], **json.loads(r['document'])) for r in cursor.fetchall()]

    def pending_outbox(self, limit=100):
        validate_limit(limit)
        with self._session() as cursor:
            cursor.execute("SELECT o.message_id,o.job_id,o.flight_id FROM prediction_outbox o "
                           "JOIN durable_flights f ON o.flight_id=f.flight_id WHERE o.published_at IS NULL "
                           "AND o.next_publish_at<=UTC_TIMESTAMP(6) AND f.status IN ('QUEUED','RUNNING') "
                           'ORDER BY o.next_publish_at,o.message_id LIMIT %s', (limit,))
            return list(cursor.fetchall())

    def mark_published(self, message_id):
        validate_job_id(message_id)
        with self._session() as cursor:
            cursor.execute('UPDATE prediction_outbox SET published_at=UTC_TIMESTAMP(6),failures=0 WHERE message_id=%s', (message_id,))

    def publication_failed(self, message_id):
        validate_job_id(message_id)
        with self._session() as cursor:
            cursor.execute('UPDATE prediction_outbox SET failures=LEAST(failures+1,1000),'
                           'next_publish_at=UTC_TIMESTAMP(6)+INTERVAL LEAST(60,POW(2,LEAST(failures,5))) SECOND '
                           'WHERE message_id=%s AND published_at IS NULL', (message_id,))

    def is_terminal(self, job_id, flight_id):
        with self._session() as cursor:
            validate_job_id(job_id)
            validate_job_id(flight_id)
            cursor.execute('SELECT status FROM durable_flights WHERE job_id=%s AND flight_id=%s', (job_id, flight_id))
            f = cursor.fetchone()
            return bool(f and f['status'] in TERMINAL)

    def reconcile(self, limit=100, republish=False):
        validate_limit(limit)
        changed = 0
        # Maintenance reconstruction resets every unfinished notification.
        # LIMIT applies only to normal lease recovery, never to this reset.
        if republish:
            with self._session() as cursor:
                cursor.execute("UPDATE prediction_outbox o JOIN durable_flights f ON o.flight_id=f.flight_id "
                               "SET o.published_at=NULL,o.next_publish_at=UTC_TIMESTAMP(6) "
                               "WHERE f.status IN ('QUEUED','RUNNING')")
                changed = cursor.rowcount
        with self._session() as cursor:
            cursor.execute("SELECT f.job_id,f.flight_id FROM durable_flights f JOIN durable_jobs j ON f.job_id=j.job_id "
                           "WHERE f.status IN ('QUEUED','RUNNING') AND (j.deadline_at<=UTC_TIMESTAMP(6) OR "
                           "(f.status='RUNNING' AND f.lease_until<=UTC_TIMESTAMP(6))) "
                           'ORDER BY j.deadline_at,f.flight_id LIMIT %s', (limit,))
            candidates = cursor.fetchall()
        for candidate in candidates:
            with self._session() as cursor:
                job_id, flight_id = candidate['job_id'], candidate['flight_id']
                job, remaining, f = self._lock(cursor, job_id, flight_id)
                if f['status'] in TERMINAL:
                    continue
                if remaining <= 0:
                    self._fail(cursor, job_id, f, 'DEADLINE_EXCEEDED', 'EXPIRED')
                elif f['status'] == 'RUNNING' and not f['leased']:
                    if f['fence'] >= job['max_attempts']:
                        self._fail(cursor, job_id, f, 'EXECUTOR_LOST')
                    else:
                        cursor.execute("UPDATE durable_attempts SET status='LOST',finished_at=UTC_TIMESTAMP(6) "
                                       "WHERE flight_id=%s AND fence=%s AND status='RUNNING'", (flight_id, f['fence']))
                        cursor.execute("UPDATE durable_flights SET status='QUEUED',lease_until=NULL WHERE flight_id=%s", (flight_id,))
                        cursor.execute('UPDATE prediction_outbox SET published_at=NULL,next_publish_at=UTC_TIMESTAMP(6) WHERE flight_id=%s', (flight_id,))
                        self._event(cursor, job_id, 'requeued', flight_id=flight_id)
                else:
                    continue
                changed += 1
        return changed
