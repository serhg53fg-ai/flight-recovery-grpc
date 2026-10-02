"""Explicit storage selection shared by Web and migration CLI."""
from __future__ import annotations

import os
from pathlib import Path


def storage_environment():
    return dict(STORAGE_BACKEND=os.environ.get('FLIGHT_STORAGE_BACKEND', 'sqlite'),
                **{f'MYSQL_{key}': os.environ.get(f'FLIGHT_MYSQL_{key}', default)
                   for key, default in (
                       ('HOST', '127.0.0.1'), ('PORT', '3306'), ('DATABASE', None),
                       ('USER', None), ('PASSWORD', ''), ('UNIX_SOCKET', None), ('TIMEOUT', '5'))})


def _integer(value, name):
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    raise ValueError(f'无效MySQL {name}配置')


def mysql_config(config):
    from .mysql_jobs import MySQLConfig
    return MySQLConfig(database=config.get('MYSQL_DATABASE'), user=config.get('MYSQL_USER'),
                       password=config.get('MYSQL_PASSWORD', ''), host=config.get('MYSQL_HOST', '127.0.0.1'),
                       port=_integer(config.get('MYSQL_PORT', 3306), 'port'),
                       timeout=_integer(config.get('MYSQL_TIMEOUT', 5), 'timeout'),
                       unix_socket=config.get('MYSQL_UNIX_SOCKET') or None)


def create_job_store(config, runtime_root):
    backend = config.get('STORAGE_BACKEND', 'sqlite')
    if backend == 'sqlite':
        from .jobs import JobStore
        return JobStore(Path(runtime_root) / 'jobs')
    if backend == 'mysql':
        from .mysql_jobs import MySQLJobStore
        return MySQLJobStore(mysql_config(config))
    raise ValueError('STORAGE_BACKEND必须为sqlite或mysql')
