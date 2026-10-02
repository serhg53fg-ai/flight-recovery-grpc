"""Reject incompatible recovery tables before accepting any business request."""
from apps.flight.storage.repository import StorageError


def validate_schema(cursor):
    def reject():
        raise StorageError('恢复任务表结构、引擎或约束不兼容')
    expected = {
        'recovery_schema': {'singleton': 'tinyint', 'version': 'int'},
        'recovery_jobs': {'recovery_id': 'char(36)', 'submission_key': 'varchar(128)',
                          'request_digest': 'char(64)', 'created_at': 'varchar(40)',
                          'source_job_id': 'char(36)', 'document': 'json'},
    }
    cursor.execute("SELECT TABLE_NAME,ENGINE FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME IN ('recovery_schema','recovery_jobs')")
    if dict(cursor.fetchall()) != {table: 'InnoDB' for table in expected}:
        reject()
    for table, columns in expected.items():
        cursor.execute('SELECT COLUMN_NAME,COLUMN_TYPE,IS_NULLABLE,CHARACTER_SET_NAME,COLLATION_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s', (table,))
        actual = {name: (kind, nullable, charset, collation) for name, kind, nullable, charset, collation in cursor.fetchall()}
        if set(actual) != set(columns):
            reject()
        for name, kind in columns.items():
            if actual[name][:2] != (kind, 'NO'):
                reject()
            if name == 'submission_key' and actual[name][2:] != ('utf8mb4', 'utf8mb4_bin'):
                reject()
            if name in {'recovery_id', 'source_job_id', 'request_digest'} and actual[name][2:] != ('ascii', 'ascii_bin'):
                reject()
            if name == 'created_at' and actual[name][2] != 'ascii':
                reject()
    cursor.execute("SELECT TABLE_NAME,INDEX_NAME,NON_UNIQUE,COLUMN_NAME,SUB_PART FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME IN ('recovery_schema','recovery_jobs') ORDER BY TABLE_NAME,INDEX_NAME,SEQ_IN_INDEX")
    indexes = {}
    for table, name, non_unique, column, prefix in cursor.fetchall():
        indexes.setdefault((table, name, non_unique), []).append((column, prefix))
    required = {('recovery_schema', 'PRIMARY', 0): ['singleton'],
                ('recovery_jobs', 'PRIMARY', 0): ['recovery_id'],
                ('recovery_jobs', 'recovery_submission', 0): ['submission_key'],
                ('recovery_jobs', 'recovery_created', 1): ['created_at', 'recovery_id'],
                ('recovery_jobs', 'recovery_source', 1): ['source_job_id', 'created_at']}
    if any(indexes.get(key) != [(column, None) for column in columns] for key, columns in required.items()):
        reject()
    if any(key[2] == 0 and key not in required for key in indexes):
        reject()
