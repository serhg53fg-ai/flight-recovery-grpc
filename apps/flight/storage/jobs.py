"""SQLite job repository for single-host development."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .documents import completed_job, encode, new_job, validate_document, validate_job_id, validate_limit
from .repository import StorageError


class JobStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'jobs.sqlite3'
        with self._connect() as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, '
                         'created_at TEXT NOT NULL, document TEXT NOT NULL)')

    @contextmanager
    def _connect(self):
        try:
            conn = sqlite3.connect(self.db, timeout=15)
            try:
                with conn:
                    yield conn
            finally:
                conn.close()
        except sqlite3.Error:
            raise StorageError('任务存储不可用') from None

    def create(self, total, input_timezone):
        job = new_job(total, input_timezone)
        with self._connect() as conn:
            conn.execute('INSERT INTO jobs VALUES (?,?,?)',
                         (job['job_id'], job['created_at'], encode(job)))
        return job

    def finish(self, job_id, results):
        validate_job_id(job_id)
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT document FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if row is None:
                raise KeyError('没有找到该任务')
            job = completed_job(json.loads(row[0]), results)
            conn.execute('UPDATE jobs SET document=? WHERE job_id=?', (encode(job), job_id))
        return job

    def get(self, job_id):
        validate_job_id(job_id)
        with self._connect() as conn:
            row = conn.execute('SELECT document FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if row is None:
            raise KeyError('没有找到该任务')
        return json.loads(row[0])

    def list_jobs(self, limit=50):
        validate_limit(limit)
        with self._connect() as conn:
            rows = conn.execute('SELECT document FROM jobs ORDER BY created_at DESC, job_id DESC LIMIT ?',
                                (limit,)).fetchall()
        return [{k: v for k, v in json.loads(row[0]).items() if k != 'results'} for row in rows]

    def import_job(self, document):
        job = validate_document(document)
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT document FROM jobs WHERE job_id=?', (job['job_id'],)).fetchone()
            if row is not None:
                if encode(json.loads(row[0])) != encode(job):
                    raise ValueError(f"迁移冲突: {job['job_id']}")
                return False
            conn.execute('INSERT INTO jobs VALUES (?,?,?)', (job['job_id'], job['created_at'], encode(job)))
        return True
