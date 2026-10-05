"""Opt-in owned MySQL daemon; never connects to the system database."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from uuid import uuid4

import pytest


@pytest.fixture(scope='session')
def mysql_server():
    if os.environ.get('FLIGHT_MYSQL_TESTS') != '1':
        pytest.skip('opt-in isolated MySQL acceptance')
    import pymysql
    executable = shutil.which('mysqld')
    if not executable:
        pytest.fail('MySQL acceptance requires mysqld')
    with tempfile.TemporaryDirectory(prefix='frmysql-') as directory:
        root = Path(directory)
        data = root / 'data'
        init = subprocess.run([executable, '--no-defaults', '--initialize-insecure',
                               f'--datadir={data}'], capture_output=True, text=True, timeout=60)
        assert init.returncode == 0, init.stderr[-2000:]
        socket_path = str(root / 'mysql.sock')
        process = subprocess.Popen([
            executable, '--no-defaults', f'--datadir={data}', f'--socket={socket_path}',
            f'--pid-file={root / "mysql.pid"}', f'--log-error={root / "mysql.log"}',
            '--skip-networking', '--mysqlx=OFF', '--skip-log-bin',
            '--innodb-buffer-pool-size=32M', '--performance-schema=OFF',
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                try:
                    connection = pymysql.connect(unix_socket=socket_path, user='root', connect_timeout=1)
                    connection.close()
                    break
                except pymysql.MySQLError:
                    if process.poll() is not None:
                        pytest.fail((root / 'mysql.log').read_text()[-3000:])
                    time.sleep(.1)
            else:
                pytest.fail('isolated MySQL startup timed out')
            yield socket_path
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@pytest.fixture
def mysql_settings(mysql_server):
    import pymysql
    database = 'flight_test_' + uuid4().hex
    conn = pymysql.connect(unix_socket=mysql_server, user='root', autocommit=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute(f'CREATE DATABASE `{database}` CHARACTER SET utf8mb4')
        yield dict(unix_socket=mysql_server, user='root', database=database)
    finally:
        with conn.cursor() as cursor:
            cursor.execute(f'DROP DATABASE `{database}`')
        conn.close()


@pytest.fixture
def mysql_config(mysql_settings):
    from apps.flight.storage.mysql_jobs import MySQLConfig, initialize_schema
    config = MySQLConfig(**mysql_settings)
    initialize_schema(config)
    return config
