"""Read-only admission checks for an owned Resident deployment."""
import os
from pathlib import Path


def check_environment(raw: dict) -> dict:
    from .resident import check_ports, validate

    checks, errors = {}, []
    try:
        config = validate(raw)
        checks['config'] = True
    except (KeyError, TypeError, ValueError, OSError) as exc:
        checks['config'] = False
        errors.append(f'config: {exc}')
        return {'ok': False, 'checks': checks, 'errors': errors}
    for name in ('mysqld', 'redis', 'nginx', 'gateway'):
        path = Path(config[name])
        valid = path.is_file() and os.access(path, os.X_OK)
        checks['binary:' + name] = valid
        if not valid:
            errors.append('missing or non-executable binary: ' + name)
    for name in ('mysqldump', 'mysql_client'):
        if config.get(name):
            path = Path(config[name])
            valid = path.is_file() and os.access(path, os.X_OK)
            checks['binary:' + name] = valid
            if not valid:
                errors.append('missing or non-executable binary: ' + name)
    python = Path(config['project']) / '.venv/bin/python'
    checks['python'] = python.is_file() and os.access(python, os.X_OK)
    if not checks['python']:
        errors.append('project Python environment is missing')
    checks['static'] = (Path(config['project']) / 'apps/flight/static').is_dir()
    if not checks['static']:
        errors.append('project static directory is missing')
    try:
        check_ports([config[name] for name in ('http_port','web_a_port','web_b_port','gateway_port','metrics_port')]
                    + [int(worker['address'].rsplit(':', 1)[1]) for worker in config['workers']
                       if worker['managed']])
        checks['ports'] = True
    except OSError as exc:
        checks['ports'] = False
        errors.append(f'ports: {exc}')
    return {'ok': all(checks.values()), 'checks': checks, 'errors': errors}
