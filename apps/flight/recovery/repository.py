"""Immutable recovery snapshots in an explicitly initialized shared database."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from apps.flight.storage.documents import encode, validate_job_id
from apps.flight.storage.mysql_jobs import _transaction
from apps.flight.storage.repository import StorageError
from .domain import identity
from .validation import validate_plan
from .summary import summarize_recovery
from .schema_validation import validate_schema


class RecoveryConflict(ValueError):
    pass


def initialize_schema(config):
    with _transaction(config) as connection, connection.cursor() as cursor:
        for statement in Path(__file__).with_name('schema.sql').read_text().split(';'):
            if statement.strip():
                cursor.execute(statement)
        cursor.execute('SELECT version FROM recovery_schema WHERE singleton=1')
        row = cursor.fetchone()
        if row and row[0] != 1:
            raise StorageError('恢复任务schema版本不兼容')
        cursor.execute('INSERT IGNORE INTO recovery_schema(singleton,version) VALUES(1,1)')
    RecoveryRepository(config)


class RecoveryRepository:
    def __init__(self, config):
        self.config = config
        with _transaction(config) as connection, connection.cursor() as cursor:
            cursor.execute('SELECT version FROM recovery_schema WHERE singleton=1')
            if cursor.fetchone() != (1,):
                raise StorageError('恢复任务schema尚未初始化或版本不兼容')
            validate_schema(cursor)

    def save(self, plan, key):
        identity(key, 'Idempotency-Key')
        snapshot = {k: plan[k] for k in ('source_job_id', 'scenario_version', 'scenario', 'algorithm_version', 'inputs', 'provenance', 'excluded')}
        if 'source_context' in plan:
            snapshot['source_context'] = plan['source_context']
        validate_job_id(snapshot['source_job_id'])
        errors = validate_plan(snapshot['inputs'], snapshot['scenario'], plan)
        if errors and plan['status'] != 'INVALID':
            raise ValueError('恢复方案未通过独立可行性校验')
        if plan.get('summary') != summarize_recovery(snapshot['inputs'], {**plan, 'validation_errors': errors}):
            raise ValueError('recovery summary does not match plan')
        digest = hashlib.sha256(encode(snapshot).encode()).hexdigest()
        document = {**json.loads(encode(plan)), 'input_digest': digest,
                    'recovery_id': str(uuid4()), 'created_at': datetime.now(timezone.utc).isoformat(),
                    'validation_errors': errors}
        with _transaction(self.config) as connection, connection.cursor() as cursor:
            # Duplicate path obtains an exclusive lock directly: INSERT failure
            # shared locks followed by FOR UPDATE can deadlock concurrent retries.
            cursor.execute('INSERT INTO recovery_jobs(recovery_id,submission_key,request_digest,created_at,source_job_id,document) VALUES(%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE submission_key=submission_key',
                           (document['recovery_id'], key, digest, document['created_at'], snapshot['source_job_id'], encode(document)))
            cursor.execute('SELECT request_digest,document FROM recovery_jobs WHERE submission_key=%s FOR UPDATE', (key,))
            row = cursor.fetchone()
            if not row or row[0] != digest:
                raise RecoveryConflict('此提交键已经绑定不同预测任务或场景') from None
            return json.loads(row[1])

    def get(self, recovery_id):
        validate_job_id(recovery_id)
        with _transaction(self.config) as connection, connection.cursor() as cursor:
            cursor.execute('SELECT document FROM recovery_jobs WHERE recovery_id=%s', (recovery_id,))
            row = cursor.fetchone()
            if not row:
                raise KeyError(recovery_id)
            return json.loads(row[0])
