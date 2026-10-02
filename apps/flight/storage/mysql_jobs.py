"""Transactional shared job storage, with explicit schema initialization."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import pymysql

from .documents import completed_job, encode, new_job, validate_document, validate_job_id, validate_limit
from .repository import StorageError

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MySQLConfig:
    database: str
    user: str
    host: str = '127.0.0.1'
    port: int = 3306
    password: str = field(default='', repr=False)
    unix_socket: str | None = None
    timeout: int = 5

    def __post_init__(self):
        if not isinstance(self.database, str) or not re.fullmatch(r'[A-Za-z0-9_]{1,64}', self.database):
            raise ValueError('无效MySQL数据库名称')
        if not isinstance(self.user, str) or not self.user or '\x00' in self.user:
            raise ValueError('必须配置MySQL用户')
        if not isinstance(self.host, str) or not self.host or '\x00' in self.host:
            raise ValueError('无效MySQL主机')
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError('无效MySQL端口')
        if type(self.timeout) is not int or not 1 <= self.timeout <= 60:
            raise ValueError('MySQL超时必须为1—60秒')
        if not isinstance(self.password, str):
            raise ValueError('无效MySQL凭据配置')
        if self.unix_socket is not None and (not isinstance(self.unix_socket, str) or
                not Path(self.unix_socket).is_absolute() or '\x00' in self.unix_socket):
            raise ValueError('MySQL socket必须为绝对路径')


def _rollback(conn):
    if conn is not None:
        try:
            conn.rollback()
        except pymysql.MySQLError:
            pass


@contextmanager
def _transaction(config):
    conn = None
    try:
        conn = pymysql.connect(host=config.host, port=config.port, user=config.user,
                               password=config.password, database=config.database,
                               unix_socket=config.unix_socket, charset='utf8mb4', autocommit=False,
                               connect_timeout=config.timeout, read_timeout=config.timeout,
                               write_timeout=config.timeout)
        # Multi-query job/event views must retain one snapshot on every server.
        with conn.cursor() as cursor:
            cursor.execute('SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        yield conn
        conn.commit()
    except pymysql.MySQLError:
        _rollback(conn)
        raise StorageError('MySQL任务存储不可用或schema不兼容') from None
    except BaseException:
        _rollback(conn)
        raise
    finally:
        if conn is not None:
            conn.close()


def initialize_schema(config):
    """Run using an initialization account; normal app startup never executes DDL."""
    statements = [statement.strip() for statement in
                  Path(__file__).with_name('mysql_schema.sql').read_text().split(';') if statement.strip()]
    with _transaction(config) as conn, conn.cursor() as cursor:
        cursor.execute(statements[0])
        cursor.execute('SELECT version FROM storage_schema WHERE singleton=1')
        row = cursor.fetchone()
        if row is not None and row[0] != SCHEMA_VERSION:
            raise StorageError('MySQL schema版本不兼容')
        for statement in statements[1:]:
            cursor.execute(statement)
        cursor.execute('INSERT INTO storage_schema (singleton, version) VALUES (1,%s) '
                       'ON DUPLICATE KEY UPDATE singleton=singleton', (SCHEMA_VERSION,))
        _validate_schema(cursor, config.database)


def _validate_schema(cursor, database):
    def reject():
        raise StorageError('MySQL表结构、引擎或约束不兼容')
    expected = {
        'storage_schema': {'singleton': 'tinyint', 'version': 'int'},
        'prediction_jobs': {'job_id': 'char(36)', 'created_at': 'varchar(40)', 'document': 'json'},
        'flight_predictions': {'job_id': 'char(36)', 'ordinal': 'int unsigned',
                               'flight_id': 'varchar(128)', 'document': 'json'},
    }
    cursor.execute('SELECT TABLE_NAME,ENGINE FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s', (database,))
    engines = dict(cursor.fetchall())
    if any(engines.get(table) != 'InnoDB' for table in expected):
        reject()
    for table, columns in expected.items():
        cursor.execute('SELECT COLUMN_NAME,COLUMN_TYPE,IS_NULLABLE FROM information_schema.COLUMNS '
                       'WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s', (database, table))
        actual = {name: (kind, nullable) for name, kind, nullable in cursor.fetchall()}
        if any(actual.get(name) != (kind, 'YES' if name == 'flight_id' else 'NO')
               for name, kind in columns.items()):
            reject()
    cursor.execute('SELECT TABLE_NAME,INDEX_NAME,NON_UNIQUE,COLUMN_NAME FROM information_schema.STATISTICS '
                   'WHERE TABLE_SCHEMA=%s ORDER BY TABLE_NAME,INDEX_NAME,SEQ_IN_INDEX', (database,))
    indexes = {}
    for table, name, non_unique, column in cursor.fetchall():
        indexes.setdefault((table, name, non_unique), []).append(column)
    for table, name, unique, columns in [
        ('storage_schema', 'PRIMARY', 0, ['singleton']),
        ('prediction_jobs', 'PRIMARY', 0, ['job_id']),
        ('prediction_jobs', 'jobs_created', 1, ['created_at', 'job_id']),
        ('flight_predictions', 'PRIMARY', 0, ['job_id', 'ordinal']),
        ('flight_predictions', 'flight_identity', 0, ['job_id', 'flight_id']),
    ]:
        if indexes.get((table, name, unique)) != columns:
            reject()
    cursor.execute('SELECT k.COLUMN_NAME,k.REFERENCED_TABLE_NAME,k.REFERENCED_COLUMN_NAME,r.DELETE_RULE '
                   'FROM information_schema.KEY_COLUMN_USAGE k JOIN information_schema.REFERENTIAL_CONSTRAINTS r '
                   'ON k.CONSTRAINT_SCHEMA=r.CONSTRAINT_SCHEMA AND k.TABLE_NAME=r.TABLE_NAME '
                   'AND k.CONSTRAINT_NAME=r.CONSTRAINT_NAME WHERE k.TABLE_SCHEMA=%s '
                   'AND k.TABLE_NAME=%s', (database, 'flight_predictions'))
    if ('job_id', 'prediction_jobs', 'job_id', 'CASCADE') not in cursor.fetchall():
        reject()


def _decode(text):
    try:
        return validate_document(json.loads(text))
    except (ValueError, TypeError):
        raise StorageError('持久任务文档不兼容') from None


def _details(cursor, job):
    for ordinal, row in enumerate(job['results'], 1):
        flight_id = row.get('flight_id')
        if flight_id is not None and (not isinstance(flight_id, str) or not 1 <= len(flight_id) <= 128):
            raise ValueError('无效航班结果ID')
        cursor.execute('INSERT INTO flight_predictions (job_id,ordinal,flight_id,document) '
                       'VALUES (%s,%s,%s,%s)', (job['job_id'], ordinal, flight_id, encode(row)))


class MySQLJobStore:
    def __init__(self, config: MySQLConfig):
        self.config = config
        with _transaction(config) as conn, conn.cursor() as cursor:
            cursor.execute('SELECT version FROM storage_schema WHERE singleton=1')
            row = cursor.fetchone()
            if row is None or row[0] != SCHEMA_VERSION:
                raise StorageError('MySQL schema未初始化或版本不兼容')
            _validate_schema(cursor, config.database)

    def create(self, total, input_timezone):
        job = new_job(total, input_timezone)
        with _transaction(self.config) as conn, conn.cursor() as cursor:
            cursor.execute('INSERT INTO prediction_jobs (job_id,created_at,document) VALUES (%s,%s,%s)',
                           (job['job_id'], job['created_at'], encode(job)))
        return job

    def get(self, job_id):
        validate_job_id(job_id)
        with _transaction(self.config) as conn, conn.cursor() as cursor:
            cursor.execute('SELECT document FROM prediction_jobs WHERE job_id=%s', (job_id,))
            row = cursor.fetchone()
        if row is None:
            raise KeyError('没有找到该任务')
        return _decode(row[0])

    def finish(self, job_id, results):
        validate_job_id(job_id)
        with _transaction(self.config) as conn, conn.cursor() as cursor:
            cursor.execute('SELECT document FROM prediction_jobs WHERE job_id=%s FOR UPDATE', (job_id,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError('没有找到该任务')
            original = _decode(row[0])
            job = completed_job(original, results)
            if job == original:
                return original
            cursor.execute('UPDATE prediction_jobs SET document=%s WHERE job_id=%s', (encode(job), job_id))
            _details(cursor, job)
        return job

    def list_jobs(self, limit=50):
        validate_limit(limit)
        with _transaction(self.config) as conn, conn.cursor() as cursor:
            cursor.execute('SELECT document FROM prediction_jobs ORDER BY created_at DESC,job_id DESC LIMIT %s', (limit,))
            rows = cursor.fetchall()
        return [{k: v for k, v in _decode(row[0]).items() if k != 'results'} for row in rows]

    def import_job(self, document):
        job = validate_document(document)
        with _transaction(self.config) as conn, conn.cursor() as cursor:
            cursor.execute('INSERT INTO prediction_jobs (job_id,created_at,document) VALUES (%s,%s,%s) '
                           'ON DUPLICATE KEY UPDATE job_id=job_id', (job['job_id'], job['created_at'], encode(job)))
            inserted = cursor.rowcount == 1
            cursor.execute('SELECT document FROM prediction_jobs WHERE job_id=%s FOR UPDATE', (job['job_id'],))
            if encode(_decode(cursor.fetchone()[0])) != encode(job):
                raise ValueError(f"迁移冲突: {job['job_id']}")
            if inserted:
                _details(cursor, job)
        return inserted
