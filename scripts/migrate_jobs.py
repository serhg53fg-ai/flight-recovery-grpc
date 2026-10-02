"""Explicit MySQL schema setup and resumable read-only SQLite migration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import shutil
import tempfile

from apps.flight.storage.documents import validate_document
from apps.flight.storage.factory import mysql_config, storage_environment
from apps.flight.storage.repository import StorageError


def migrate_sqlite(source, target):
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise ValueError('源SQLite文件不存在')
    # SQLite mode=ro can still create WAL/shared-memory sidecars. Never open
    # the source with SQLite: copy the stopped database and committed WAL.
    files = [source, Path(str(source) + '-wal')]
    def signatures():
        return [(p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
                if p.exists() else None for p in files]
    before = signatures()
    with tempfile.TemporaryDirectory(prefix='flight-migration-') as directory:
        snapshot = Path(directory) / source.name
        for p, signature in zip(files, before):
            if signature is not None:
                shutil.copyfile(p, Path(directory) / p.name)
        if signatures() != before:
            raise ValueError('源数据库正在变化，请停止业务后重试')
        return _migrate_snapshot(snapshot, target)


def _migrate_snapshot(source, target):
    counts = {'migrated': 0, 'skipped': 0}
    try:
        conn = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)
        try:
            conn.execute('BEGIN')
            rows = conn.execute('SELECT job_id,created_at,document FROM jobs ORDER BY created_at,job_id')
            for job_id, created_at, text in rows:
                try:
                    job = validate_document(json.loads(text))
                except (ValueError, TypeError):
                    raise ValueError('源任务文档校验失败') from None
                if job['job_id'] != job_id or job['created_at'] != created_at:
                    raise ValueError('源任务索引与文档不一致')
                if job['status'] == 'RUNNING':
                    raise ValueError(f'源任务仍为RUNNING，必须先停止并完成任务: {job_id}')
                counts['migrated' if target.import_job(job) else 'skipped'] += 1
        finally:
            conn.close()
    except sqlite3.Error:
        raise ValueError('源SQLite格式不兼容或无法只读访问') from None
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description='Initialize MySQL or migrate terminal SQLite jobs')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--initialize', action='store_true')
    mode.add_argument('--source-sqlite', type=Path)
    args = parser.parse_args(argv)
    try:
        from apps.flight.storage.mysql_jobs import MySQLJobStore, initialize_schema
        config = mysql_config(storage_environment())
        if args.initialize:
            initialize_schema(config)
            result = {'schema_version': 1, 'initialized': True}
        else:
            result = migrate_sqlite(args.source_sqlite, MySQLJobStore(config))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (StorageError, ValueError, OSError):
        # Configuration/driver exceptions may mention credentials supplied by a caller.
        print('存储初始化或迁移失败；请检查配置、schema和源任务终态，不会覆盖目标记录')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
