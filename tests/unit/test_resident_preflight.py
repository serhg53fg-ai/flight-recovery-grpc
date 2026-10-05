"""Read-only checks must fail before touching an owned or foreign stack."""
from pathlib import Path
import socket

from tests.unit.test_resident import settings


def test_missing_binary_preflight_preserves_runtime(tmp_path, monkeypatch):
    from deploy.distributed import resident
    from deploy.distributed.preflight import check_environment

    runtime = tmp_path / 'runtime'
    config = {**settings(runtime), 'gateway': str(tmp_path / 'missing-gateway')}
    monkeypatch.setattr(resident, 'stop', lambda *_: (_ for _ in ()).throw(AssertionError('stop called')))
    result = check_environment(config)
    assert result['ok'] is False
    assert result['checks']['binary:gateway'] is False
    assert not runtime.exists()


def test_occupied_port_is_reported_without_stopping_listener(tmp_path):
    from deploy.distributed.preflight import check_environment

    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        config = {**settings(tmp_path / 'runtime'), 'http_port': listener.getsockname()[1]}
        result = check_environment(config)
        assert result['ok'] is False
        assert result['checks']['ports'] is False
        assert not (tmp_path / 'runtime').exists()
        assert listener.fileno() >= 0


def test_second_managed_worker_port_conflict_is_reported(tmp_path):
    from deploy.distributed.preflight import check_environment

    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        config = {**settings(tmp_path / 'runtime'), 'workers': [
            {'worker_id': 'a', 'address': '127.0.0.1:50101',
             'capacity': 1, 'release_version': 'test-v1', 'managed': True},
            {'worker_id': 'b', 'address': f'127.0.0.1:{listener.getsockname()[1]}',
             'capacity': 1, 'release_version': 'test-v1', 'managed': True},
        ]}
        result = check_environment(config)
        assert result['ok'] is False
        assert result['checks']['ports'] is False
        assert listener.fileno() >= 0


def test_healthy_preflight_does_not_create_runtime(tmp_path):
    from deploy.distributed.preflight import check_environment

    runtime = tmp_path / 'runtime'
    result = check_environment(settings(runtime))
    assert result['ok'] is True
    assert result['checks']['config'] is True
    assert result['checks']['ports'] is True
    assert not runtime.exists()


def test_prepare_rejects_missing_gateway_before_creating_runtime(tmp_path):
    import json
    from deploy.distributed.resident import main

    runtime = tmp_path / 'runtime'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({**settings(runtime), 'gateway': str(tmp_path / 'missing')}))
    assert main(['prepare', '--config', str(config)]) == 2
    assert not runtime.exists()


def test_preflight_cli_is_read_only(tmp_path, capsys):
    import json
    from deploy.distributed.resident import main

    runtime = tmp_path / 'runtime'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(settings(runtime)))
    assert main(['preflight', '--config', str(config)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['ok'] is True
    assert not runtime.exists()
