from pathlib import Path
import runpy

import pytest
from flask import Flask

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('missing', ['FLIGHT_STORAGE_BACKEND', 'FLIGHT_DURABLE_ENABLED', 'FLIGHT_RECOVERY_ENABLED'])
def test_production_factory_requires_shared_durable_services(monkeypatch, missing):
    from deploy.distributed.wsgi import create_production_app
    for key, value in {'FLIGHT_STORAGE_BACKEND': 'mysql', 'FLIGHT_DURABLE_ENABLED': '1', 'FLIGHT_RECOVERY_ENABLED': '1'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv(missing, raising=False)
    with pytest.raises(ValueError): create_production_app()


def test_production_factory_disables_local_file_routes(monkeypatch):
    from deploy.distributed import wsgi
    for key, value in {'FLIGHT_STORAGE_BACKEND': 'mysql', 'FLIGHT_DURABLE_ENABLED': '1', 'FLIGHT_RECOVERY_ENABLED': '1', 'FLIGHT_WEB_INSTANCE': 'web-a'}.items():
        monkeypatch.setenv(key, value)
    app = Flask(__name__)
    app.add_url_rule('/api/save-scenario', 'legacy.save_scenario', lambda: 'old', methods=['POST'])
    app.add_url_rule('/api/run-optimization', 'legacy.run_optimization', lambda: 'old', methods=['POST'])
    app.add_url_rule('/health', 'health', lambda: 'ok')
    monkeypatch.setattr(wsgi, 'create_app', lambda: app)
    production = wsgi.create_production_app()
    with production.test_client() as client:
        assert client.post('/api/save-scenario').status_code == 410
        assert client.post('/api/run-optimization').status_code == 410
        assert client.get('/health').headers['X-Flight-Instance'] == 'web-a'


@pytest.mark.parametrize('endpoint', ['predict_batch', 'predict_stream', 'upload_predict'])
def test_production_batch_cannot_bypass_durable_acceptance(monkeypatch, endpoint):
    from deploy.distributed import wsgi
    for key, value in {'FLIGHT_STORAGE_BACKEND': 'mysql', 'FLIGHT_DURABLE_ENABLED': '1', 'FLIGHT_RECOVERY_ENABLED': '1'}.items():
        monkeypatch.setenv(key, value)
    app = Flask(__name__)
    app.add_url_rule('/old-batch', 'predictions.' + endpoint, lambda: 'bypassed', methods=['POST'])
    monkeypatch.setattr(wsgi, 'create_app', lambda: app)
    with wsgi.create_production_app().test_client() as client:
        response = client.post('/old-batch')
        assert response.status_code == 410
        assert response.json['error_code'] == 'SYNC_BATCH_DISABLED'


@pytest.mark.parametrize('ports', [(80, 7101, 7102), (8080, 7101, 7101), (True, 7101, 7102), (8080, 65536, 7102)])
def test_renderer_rejects_invalid_or_colliding_ports(tmp_path, ports):
    from deploy.distributed.nginx import render_config
    with pytest.raises(ValueError): render_config(ROOT, tmp_path, *ports)


@pytest.mark.parametrize('path', ['/srv/flight-example/user/tzb/tiaozhan4/run', '/srv/flight-example/user/rpc/run', '/srv/flight-example/user/grpc/run', '/tmp/x\ny'])
def test_renderer_rejects_original_or_injected_runtime(path):
    from deploy.distributed.nginx import render_config
    with pytest.raises(ValueError): render_config(ROOT, Path(path))


def test_gunicorn_avoids_forking_active_grpc_channels():
    settings = runpy.run_path(str(ROOT / 'deploy/distributed/gunicorn.conf.py'))
    assert settings['preload_app'] is False
    assert settings['worker_class'] == 'gthread' and settings['threads'] >= 4


@pytest.mark.parametrize('name', ['body', 'proxy', 'nginx.pid', 'access.log', 'nginx.conf'])
def test_runtime_output_symlinks_cannot_write_originals(tmp_path, name):
    from deploy.distributed.nginx import render_config
    (tmp_path / name).symlink_to('/srv/flight-example/user/grpc')
    with pytest.raises(ValueError): render_config(ROOT, tmp_path)


def test_resolved_path_is_checked_for_configuration_injection(tmp_path):
    from deploy.distributed.nginx import render_config
    target = tmp_path / 'a"b'
    target.mkdir()
    link = tmp_path / 'safe'
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError): render_config(ROOT, link)


def test_distributed_supervisor_template_can_relocate_to_server(tmp_path, monkeypatch):
    from supervisor.options import ServerOptions
    monkeypatch.setenv('FLIGHT_PROJECT_ROOT', str(tmp_path))
    for name in ('distributed', 'nginx', 'web-a', 'web-b'):
        (tmp_path / 'runtime' / name).mkdir(parents=True)
    template = ROOT / 'deploy/distributed/supervisor.conf.example'
    options = ServerOptions()
    options.realize(['-c', str(template)])
    programs = [p for group in options.process_group_configs for p in group.process_configs]
    assert programs
    assert all(p.directory == str(tmp_path) for p in programs)
