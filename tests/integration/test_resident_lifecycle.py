"""Owned project backup and restore acceptance, opt-in isolated MySQL."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.integration.mysql_fixtures import mysql_server
from tests.unit.test_resident import settings


@pytest.fixture
def backup_source(mysql_server, tmp_path):
    import pymysql
    from deploy.distributed.resident import validate
    from scripts.resident_backup import backup_owned_runtime, verify_backup

    runtime = Path(mysql_server).parent
    config = validate(settings(runtime))
    (runtime / 'stack.json').write_text(json.dumps(config))
    (runtime / 'initialized.json').write_text(json.dumps({'initialized': True}))
    connection = pymysql.connect(unix_socket=mysql_server, user='root', autocommit=True)
    try:
        with connection.cursor() as cursor:
            cursor.execute('CREATE DATABASE flight_resident')
            cursor.execute('CREATE TABLE flight_resident.backup_evidence (id INT PRIMARY KEY, value VARCHAR(20)) ENGINE=InnoDB')
            cursor.execute("INSERT INTO flight_resident.backup_evidence VALUES (1, 'preserved')")
        from apps.flight.storage.mysql_jobs import MySQLConfig, initialize_schema as initialize_storage
        from apps.flight.tasks.repository import initialize_durable
        from apps.flight.recovery.repository import initialize_schema as initialize_recovery
        db = MySQLConfig(database='flight_resident', user='root', unix_socket=mysql_server)
        initialize_storage(db)
        initialize_durable(db)
        initialize_recovery(db)
        output = tmp_path / 'backup'
        yield config, output, connection
    finally:
        with connection.cursor() as cursor:
            cursor.execute('DROP DATABASE IF EXISTS flight_resident')
        connection.close()


def test_transactional_backup_preserves_running_mysql_and_hashes(backup_source):
    from scripts.resident_backup import backup_owned_runtime, verify_backup

    config, output, connection = backup_source
    manifest = backup_owned_runtime(config, output)
    assert manifest['schema_version'] == 'resident-backup-v1'
    assert manifest['database_schema_versions'] == {'storage': 1, 'durable': 1, 'recovery': 1}
    assert verify_backup(output) == manifest
    assert manifest['database_sha256'] == hashlib.sha256((output / 'database.sql').read_bytes()).hexdigest()
    assert b'preserved' in (output / 'database.sql').read_bytes()
    with connection.cursor() as cursor:
        cursor.execute('SELECT value FROM flight_resident.backup_evidence WHERE id=1')
        assert cursor.fetchone()[0] == 'preserved'


def test_restore_into_new_runtime_keeps_source_unchanged(backup_source, tmp_path):
    import pymysql
    from deploy.distributed.resident import commands, validate, wait_mysql
    from scripts.resident_backup import backup_owned_runtime, restore_owned_runtime

    source, backup, original = backup_source
    manifest = backup_owned_runtime(source, backup)
    target = tmp_path / 'restored'
    config = validate({**settings(target), 'mysqld': shutil.which('mysqld'),
                       'mysql_client': shutil.which('mysql')})
    result = restore_owned_runtime(backup, config)
    assert result['restored'] is True
    assert result['database_sha256'] == manifest['database_sha256']
    assert (target / 'initialized.json').is_file()
    with (target / 'logs/restore-probe.log').open('wb') as log:
        server = subprocess.Popen(commands(config)['mysql'], stdout=log, stderr=log)
    try:
        wait_mysql(config, server)
        connection = pymysql.connect(unix_socket=str(target / 'mysql.sock'), user='root',
                                     database='flight_resident')
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT value FROM backup_evidence WHERE id=1')
                assert cursor.fetchone()[0] == 'preserved'
        finally:
            connection.close()
    finally:
        server.terminate()
        server.wait(10)
    with original.cursor() as cursor:
        cursor.execute('SELECT value FROM flight_resident.backup_evidence WHERE id=1')
        assert cursor.fetchone()[0] == 'preserved'


def test_restore_rejects_incompatible_schema_and_existing_target(backup_source, tmp_path):
    from scripts.resident_backup import backup_owned_runtime, restore_owned_runtime

    source, backup, _ = backup_source
    backup_owned_runtime(source, backup)
    target = tmp_path / 'restored'
    config = {**settings(target), 'mysqld': shutil.which('mysqld'),
              'mysql_client': shutil.which('mysql')}
    manifest_path = backup / 'backup.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['database_schema_versions']['storage'] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='schema'):
        restore_owned_runtime(backup, config)
    assert not target.exists()
    manifest['database_schema_versions']['storage'] = 1
    manifest_path.write_text(json.dumps(manifest))
    target.mkdir()
    with pytest.raises(ValueError, match='new'):
        restore_owned_runtime(backup, config)
