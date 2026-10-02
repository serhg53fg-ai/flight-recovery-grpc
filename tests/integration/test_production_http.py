import json
import time
from uuid import uuid4
import pytest

from tests.integration.production_fixtures import (
    nginx_binary, mysql_server, mysql_settings, mysql_config, redis_socket, queue,
    repo, gateway_address, production_stack, http, stop_process, free_port, start_nginx, wait_http)
from tests.integration.test_durable_queue import publisher
from tests.integration.test_durable_execution import executor
from tests.integration.test_prediction_flow import FLIGHT, excel_file
from apps.flight.clients.inference import InferenceClient


def submit(stack):
    with http(stack.url + '/api/v1/prediction-jobs', 'POST', {'flights': [FLIGHT]}, {'Idempotency-Key': str(uuid4())}) as response:
        assert response.status == 202
        return json.load(response)


def event(response):
    lines = []
    while True:
        line = response.readline()
        if not line or line == b'\n':
            break
        lines.append(line.decode())
    return ''.join(lines)


def run_prediction(repo, queue, address):
    publisher(repo, queue).run_once()
    client = InferenceClient(address, 2)
    try: executor(repo, queue, client).run_once()
    finally: client.close()


def test_proxy_distributes_instances_and_serves_static(production_stack):
    seen = set()
    for _ in range(6):
        with http(production_stack.url + '/health') as response:
            assert response.status == 200
            seen.add(response.headers['X-Flight-Instance'])
    assert seen == {'web-a', 'web-b'}
    with http(production_stack.url + '/static/flight_jobs.js') as response:
        assert response.status == 200 and 'javascript' in response.headers['Content-Type']
        assert 'X-Flight-Instance' not in response.headers
        assert b'FlightJobs' in response.read()
    with http(production_stack.url + '/api/save-scenario', 'POST', {'scenario_name': 'test'}) as response:
        assert response.status == 410


def test_proxy_excel_202_executes_and_downloads_shared_result(production_stack, repo, queue, gateway_address):
    buffer, name = excel_file([FLIGHT])
    boundary = 'flight-' + uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\r\n\r\n'.encode()
            + buffer.getvalue() + f'\r\n--{boundary}--\r\n'.encode())
    with http(production_stack.url + '/api/v1/prediction-jobs/upload', 'POST', body,
              {'Content-Type': 'multipart/form-data; boundary=' + boundary, 'Idempotency-Key': 'excel-proxy'}) as response:
        assert response.status == 202
        job = json.load(response)
    assert job['status'] == 'QUEUED'
    run_prediction(repo, queue, gateway_address)
    seen = set()
    for _ in range(4):
        with http(production_stack.url + job['status_url']) as response:
            seen.add(response.headers['X-Flight-Instance'])
            assert json.load(response)['status'] == 'SUCCEEDED'
    assert seen == {'web-a', 'web-b'}
    with http(production_stack.url + job['download_url']) as response:
        assert response.status == 200 and response.read().startswith(b'PK')


def test_proxy_upload_limit_is_json_413(production_stack):
    with http(production_stack.url + '/api/v1/prediction-jobs/upload', 'POST', b'x' * (16*1024*1024 + 1)) as response:
        assert response.status == 413
        assert json.load(response)['error_code'] == 'REQUEST_TOO_LARGE'


def test_proxy_chunked_upload_limit_is_json_413(production_stack):
    from http.client import HTTPConnection
    connection = HTTPConnection('127.0.0.1', production_stack.ports[0], timeout=5)
    try:
        chunks = iter([b'x' * (1024*1024)] * 16 + [b'x'])
        connection.request('POST', '/api/v1/prediction-jobs/upload', body=chunks, encode_chunked=True)
        response = connection.getresponse()
        assert response.status == 413
        assert json.load(response)['error_code'] == 'REQUEST_TOO_LARGE'
    finally:
        connection.close()


def test_proxy_sse_is_immediate_and_survives_instance_loss(production_stack, repo, queue, gateway_address):
    stack = production_stack
    job = submit(stack)
    started = time.monotonic()
    response = http(stack.url + job['events_url'])
    origin = response.headers['X-Flight-Instance']
    block = event(response)
    assert 'id: 1\n' in block and 'event: submitted' in block
    assert time.monotonic() - started < 3
    stopped = 0 if origin == 'web-a' else 1
    stop_process(stack.web[stopped])
    response.close()
    assert repo.get(job['job_id'])['status'] == 'QUEUED'
    run_prediction(repo, queue, gateway_address)
    with http(stack.url + job['status_url']) as status:
        assert status.status == 200 and json.load(status)['status'] == 'SUCCEEDED'
    with http(stack.url + job['events_url'], headers={'Last-Event-ID': '1'}) as resumed:
        assert resumed.headers['X-Flight-Instance'] != origin
        data = resumed.read().decode()
        assert 'id: 1\n' not in data and 'event: complete' in data
    stack.start_web(stopped)
    with http('http://127.0.0.1:' + str(stack.ports[stopped+1]) + job['status_url']) as restarted:
        assert json.load(restarted)['status'] == 'SUCCEEDED'


def test_proxy_limits_sse_connections(production_stack, repo):
    job = submit(production_stack)
    streams = []
    try:
        for _ in range(8):
            response = http(production_stack.url + job['events_url'])
            assert response.status == 200 and 'id: 1' in event(response)
            streams.append(response)
        with http(production_stack.url + job['events_url']) as overflow:
            assert overflow.status == 429
            assert json.load(overflow)['error_code'] == 'SSE_CONNECTION_LIMIT'
    finally:
        repo.cancel(job['job_id'])
        for response in streams: response.close()


@pytest.mark.parametrize('failure', ['503', 'disconnect'])
def test_proxy_never_replays_post_after_upstream_error(nginx_binary, tmp_path, failure):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    bodies = []
    servers = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        def do_POST(self):
            bodies.append(self.rfile.read(int(self.headers['Content-Length'])))
            if failure == '503':
                self.send_response(503); self.end_headers(); self.wfile.write(b'accepted before failure')
            else:
                self.close_connection = True
        def log_message(self, *args): pass
    process = None
    try:
        for _ in range(2):
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
        port = free_port()
        process = start_nginx(nginx_binary, tmp_path, (port, servers[0].server_port, servers[1].server_port))
        url = 'http://127.0.0.1:' + str(port)
        wait_http(url + '/health', process)
        with http(url + '/api/v1/prediction-jobs', 'POST', b'one body') as response:
            assert response.status == (503 if failure == '503' else 502)
        assert bodies == [b'one body']
    finally:
        stop_process(process)
        for server in servers: server.shutdown(); server.server_close()


def test_proxy_does_not_retry_post_even_before_connect(nginx_binary, tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    bodies, servers = [], []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('X-Flight-Instance', self.server.label)
            self.end_headers(); self.wfile.write(b'ok')
        def do_POST(self):
            bodies.append(self.rfile.read(int(self.headers['Content-Length'])))
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        def log_message(self, *args): pass
    process = None
    try:
        for label in ('a', 'b'):
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            server.label = label
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
        port = free_port()
        process = start_nginx(nginx_binary, tmp_path, (port, servers[0].server_port, servers[1].server_port))
        url = 'http://127.0.0.1:' + str(port)
        previous = wait_http(url + '/health', process)
        target = servers[1 if previous == 'a' else 0]
        target.shutdown(); target.server_close()
        with http(url + '/api/v1/prediction-jobs', 'POST', b'one body') as response:
            assert response.status == 502
        assert bodies == []
        with http(url + '/health') as response:
            assert response.status == 200
    finally:
        stop_process(process)
        for server in servers: server.shutdown(); server.server_close()
