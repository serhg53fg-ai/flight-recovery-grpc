"""Owned high-port production stack; never manages system services."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from deploy.distributed.nginx import render_config
from apps.flight.recovery.repository import initialize_schema
from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.redis_fixtures import redis_socket, queue
from tests.integration.test_durable_storage import repo
from tests.integration.test_prediction_flow import gateway_address

ROOT = Path(__file__).resolve().parents[2]


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def http(url, method='GET', data=None, headers=None):
    if isinstance(data, dict):
        data = json.dumps(data).encode()
        headers = {'Content-Type': 'application/json', **(headers or {})}
    request = Request(url, method=method, data=data, headers=headers or {})
    try:
        return urlopen(request, timeout=5)
    except HTTPError as response:
        return response


def wait_http(url, process):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('owned HTTP process exited during startup')
        try:
            with http(url) as response:
                if response.status == 200:
                    return response.headers.get('X-Flight-Instance')
        except (URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(.05)
    raise TimeoutError('owned HTTP startup timed out')


def stop_process(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(5)


@pytest.fixture(scope='session')
def nginx_binary():
    if os.environ.get('FLIGHT_HTTP_TESTS') != '1':
        pytest.skip('opt-in owned Nginx/Gunicorn acceptance')
    value = os.environ.get('FLIGHT_NGINX_BINARY') or shutil.which('nginx')
    fallback = ROOT / 'runtime/tools/nginx-root/usr/sbin/nginx'
    if not value and fallback.exists():
        value = str(fallback)
    if not value:
        pytest.fail('production acceptance requires Nginx or FLIGHT_NGINX_BINARY')
    return str(Path(value).resolve())


def start_nginx(binary, directory, ports):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('body', 'proxy', 'fastcgi', 'uwsgi', 'scgi'):
        (directory / name).mkdir(exist_ok=True)
    config = directory / 'nginx.conf'
    config.write_text(render_config(ROOT, directory, *ports))
    command = [binary, '-p', str(directory) + '/', '-c', str(config)]
    check = subprocess.run(command + ['-t'], capture_output=True, text=True, timeout=10)
    assert check.returncode == 0, check.stderr
    log = open(directory / 'process.log', 'wb')
    try:
        return subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
    finally:
        log.close()


class ProductionStack:
    def __init__(self, binary, directory, config, gateway, environment_overrides=None):
        self.binary, self.directory = binary, directory
        self.ports = (free_port(), free_port(), free_port())
        while len(set(self.ports)) != 3:
            self.ports = (free_port(), free_port(), free_port())
        self.url = 'http://127.0.0.1:' + str(self.ports[0])
        self.environment = {**os.environ, 'FLIGHT_STORAGE_BACKEND': 'mysql',
            'FLIGHT_MYSQL_DATABASE': config.database, 'FLIGHT_MYSQL_USER': config.user,
            'FLIGHT_MYSQL_PASSWORD': config.password, 'FLIGHT_MYSQL_UNIX_SOCKET': config.unix_socket,
            'FLIGHT_DURABLE_ENABLED': '1', 'FLIGHT_DURABLE_MODEL_VERSION': 'test-v1',
            'FLIGHT_DURABLE_SOURCE': 'TEST', 'FLIGHT_DURABLE_PROMPT_VERSION': 'test-v1',
            'FLIGHT_RECOVERY_ENABLED': '1', 'FLIGHT_GATEWAY_ADDRESS': gateway,
            'FLIGHT_RPC_TIMEOUT_SECONDS': '2', 'PYTHONPATH': str(ROOT) + ':' + str(ROOT / 'generated'),
            **(environment_overrides or {})}
        self.web = [None, None]
        self.proxy = None

    def start_web(self, index):
        env = {**self.environment, 'FLIGHT_WEB_INSTANCE': 'web-' + ('a' if index == 0 else 'b'),
               'FLIGHT_RUNTIME_ROOT': str(self.directory / ('web-' + str(index)))}
        log = open(self.directory / ('web-' + str(index) + '.log'), 'ab')
        try:
            process = subprocess.Popen([sys.executable, '-m', 'gunicorn', '-c',
                str(ROOT / 'deploy/distributed/gunicorn.conf.py'), '--bind', '127.0.0.1:' + str(self.ports[index+1]),
                'deploy.distributed.wsgi:create_production_app()'], cwd=ROOT, env=env,
                stdout=log, stderr=log, start_new_session=True)
        finally:
            log.close()
        self.web[index] = process
        wait_http('http://127.0.0.1:' + str(self.ports[index+1]) + '/health', process)

    def close(self):
        stop_process(self.proxy)
        for process in self.web:
            stop_process(process)


@pytest.fixture
def production_stack(nginx_binary, mysql_config, repo, tmp_path, gateway_address):
    initialize_schema(mysql_config)
    stack = ProductionStack(nginx_binary, tmp_path, mysql_config, gateway_address)
    try:
        stack.start_web(0)
        stack.start_web(1)
        stack.proxy = start_nginx(nginx_binary, tmp_path / 'nginx', stack.ports)
        wait_http(stack.url + '/health', stack.proxy)
        yield stack
    finally:
        stack.close()
