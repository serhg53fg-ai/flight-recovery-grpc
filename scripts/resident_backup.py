"""Consistent database export and hash-pinned metadata for an owned Resident stack."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from deploy.distributed.resident import validate
from deploy.distributed.release import load_release


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_config(config: dict) -> dict:
    value = validate(config)
    runtime = Path(value['runtime'])
    if not (runtime / 'initialized.json').is_file():
        raise ValueError('owned runtime is not initialized')
    try:
        prepared = validate(json.loads((runtime / 'stack.json').read_text()))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('owned stack configuration is missing') from exc
    if prepared != value:
        raise ValueError('running stack configuration differs from requested backup')
    return value


def _check_transactional_tables(socket: Path) -> dict:
    import pymysql

    connection = pymysql.connect(unix_socket=str(socket), user='root', database='flight_resident',
                                 connect_timeout=2, read_timeout=2)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT TABLE_NAME, ENGINE FROM information_schema.TABLES "
                           "WHERE TABLE_SCHEMA='flight_resident' AND TABLE_TYPE='BASE TABLE'")
            rows = cursor.fetchall()
            if not rows or any(engine != 'InnoDB' for _, engine in rows):
                raise ValueError('backup requires an initialized InnoDB-only database')
            versions = {}
            for name, table in (('storage', 'storage_schema'),
                                ('durable', 'durable_schema'), ('recovery', 'recovery_schema')):
                cursor.execute('SELECT version FROM ' + table + ' WHERE singleton=1')
                row = cursor.fetchone()
                if row != (1,):
                    raise ValueError('incompatible database schema: ' + name)
                versions[name] = row[0]
            return versions
    finally:
        connection.close()


def backup_owned_runtime(config: dict, output: Path) -> dict:
    """Dump one prepared, running project database without stopping it."""
    config = _prepared_config(config)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('backup output must be a new directory')
    if not output.parent.is_dir():
        raise ValueError('backup parent directory is missing')
    runtime = Path(config['runtime'])
    versions = _check_transactional_tables(runtime / 'mysql.sock')
    executable = config.get('mysqldump') or shutil.which('mysqldump')
    if not executable or not Path(executable).is_file():
        raise ValueError('mysqldump is unavailable')
    temporary = Path(tempfile.mkdtemp(prefix='.resident-backup-', dir=output.parent))
    try:
        dump = temporary / 'database.sql'
        with dump.open('wb') as stream:
            result = subprocess.run([str(executable), '--no-defaults', '--protocol=SOCKET',
                                     '--socket=' + str(runtime / 'mysql.sock'), '--user=root',
                                     '--single-transaction', '--quick', '--skip-lock-tables',
                                     '--set-gtid-purged=OFF', '--routines', '--triggers',
                                     '--events', '--hex-blob', '--databases', 'flight_resident'],
                                    stdout=stream, stderr=subprocess.PIPE, timeout=120)
        if result.returncode != 0 or dump.stat().st_size == 0:
            raise RuntimeError('transactional database dump failed')
        stack = temporary / 'stack.json'
        shutil.copy2(runtime / 'stack.json', stack)
        release_hash = None
        artifacts = {}
        if config['backend'] == 'composite':
            release_path = Path(config['release_manifest_path'])
            release = load_release(release_path)
            target = temporary / 'release.json'
            shutil.copy2(release_path, target)
            release_hash = _sha(target)
            artifacts = release['artifacts']
        manifest = {'schema_version': 'resident-backup-v1',
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'source_runtime': str(runtime),
                    'database': 'flight_resident',
                    'database_schema_versions': versions,
                    'database_sha256': _sha(dump),
                    'stack_sha256': _sha(stack),
                    'release_sha256': release_hash,
                    'release_identity': config.get('release_identity'),
                    'artifacts': artifacts}
        (temporary / 'backup.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        temporary.rename(output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_backup(directory: Path) -> dict:
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('backup directory is missing or linked')
    try:
        manifest = json.loads((directory / 'backup.json').read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('invalid backup manifest') from exc
    if manifest.get('schema_version') != 'resident-backup-v1' or manifest.get('database') != 'flight_resident':
        raise ValueError('incompatible backup schema')
    if manifest.get('database_schema_versions') != {'storage': 1, 'durable': 1, 'recovery': 1}:
        raise ValueError('incompatible database schema versions')
    for name, field in (('database.sql', 'database_sha256'), ('stack.json', 'stack_sha256')):
        path = directory / name
        if path.is_symlink() or not path.is_file() or _sha(path) != manifest.get(field):
            raise ValueError('backup file hash mismatch: ' + name)
    if manifest.get('release_sha256'):
        path = directory / 'release.json'
        if path.is_symlink() or not path.is_file() or _sha(path) != manifest['release_sha256']:
            raise ValueError('backup release hash mismatch')
    return manifest


def restore_owned_runtime(backup: Path, target_config: dict) -> dict:
    """Import a checked backup only into a newly prepared private runtime."""
    from deploy.distributed.preflight import check_environment
    from deploy.distributed.resident import commands, initialize, wait_mysql, write_config

    manifest = verify_backup(backup)
    config = validate(target_config)
    target = Path(config['runtime'])
    if target.exists() or target.is_symlink():
        raise ValueError('restore target must be a new runtime directory')
    if target == Path(manifest['source_runtime']).resolve() or Path(backup).resolve() in target.parents:
        raise ValueError('restore target must differ from source and backup')
    prepared = json.loads((Path(backup) / 'stack.json').read_text())
    if config['backend'] != prepared.get('backend') or config.get('release_identity') != manifest.get('release_identity'):
        raise ValueError('restore target release is incompatible with backup')
    preflight = check_environment(config)
    if not preflight['ok']:
        raise ValueError('restore preflight failed: ' + '; '.join(preflight['errors']))
    client = config.get('mysql_client') or shutil.which('mysql')
    if not client or not Path(client).is_file():
        raise ValueError('mysql client is unavailable')
    write_config(config)
    initialize(config)
    log = (target / 'logs/mysql-restore.log').open('ab')
    try:
        server = subprocess.Popen(commands(config)['mysql'], stdout=log, stderr=log)
    finally:
        log.close()
    try:
        wait_mysql(config, server)
        with (Path(backup) / 'database.sql').open('rb') as stream:
            result = subprocess.run([str(client), '--no-defaults', '--protocol=SOCKET',
                                     '--socket=' + str(target / 'mysql.sock'), '--user=root'],
                                    stdin=stream, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=120)
        if result.returncode != 0:
            raise RuntimeError('database restore import failed')
        if _check_transactional_tables(target / 'mysql.sock') != manifest['database_schema_versions']:
            raise ValueError('restored database schema is incompatible')
        document = {'restored': True, 'backup': str(Path(backup).resolve()),
                    'database_sha256': manifest['database_sha256'],
                    'release_identity': manifest.get('release_identity')}
        (target / 'restore.json').write_text(json.dumps(document, indent=2) + '\n')
        return document
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(5)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Back up or restore an owned Resident runtime')
    sub = parser.add_subparsers(dest='action', required=True)
    backup = sub.add_parser('backup')
    backup.add_argument('--config', type=Path, required=True)
    backup.add_argument('--output', type=Path, required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--backup', type=Path, required=True)
    restore = sub.add_parser('restore')
    restore.add_argument('--backup', type=Path, required=True)
    restore.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == 'backup':
            config = json.loads(args.config.read_text())
            result = backup_owned_runtime(config, args.output)
            summary = {'ok': True, 'output': str(args.output.absolute()),
                       'database_sha256': result['database_sha256']}
        elif args.action == 'verify':
            result = verify_backup(args.backup)
            summary = {'ok': True, 'database_sha256': result['database_sha256'],
                       'release_identity': result['release_identity']}
        else:
            config = json.loads(args.config.read_text())
            summary = restore_owned_runtime(args.backup, config)
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError,
            subprocess.TimeoutExpired) as exc:
        print('resident backup operation failed: ' + str(exc), file=__import__('sys').stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
