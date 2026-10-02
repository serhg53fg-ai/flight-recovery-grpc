"""Owned opt-in Redis daemon, private socket and data directory."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import pytest


@pytest.fixture(scope='session')
def redis_socket():
    if os.environ.get('FLIGHT_REDIS_TESTS') != '1':
        pytest.skip('opt-in isolated Redis acceptance')
    import redis
    executable = shutil.which('redis-server')
    if not executable:
        pytest.fail('Redis acceptance requires redis-server')
    with tempfile.TemporaryDirectory(prefix='flight-redis-') as directory:
        socket = str(Path(directory) / 'redis.sock')
        process = subprocess.Popen([executable, '--port', '0', '--unixsocket', socket,
                                    '--unixsocketperm', '700', '--dir', directory, '--save', '',
                                    '--appendonly', 'yes', '--appendfsync', 'everysec',
                                    '--maxmemory', '32mb', '--maxmemory-policy', 'noeviction'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        client = redis.Redis(unix_socket_path=socket, socket_timeout=1)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    if client.ping():
                        break
                except redis.RedisError:
                    if process.poll() is not None:
                        pytest.fail('owned Redis failed to start')
                    time.sleep(.05)
            else:
                pytest.fail('owned Redis startup timed out')
            yield socket
        finally:
            client.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(5)


@pytest.fixture
def queue(redis_socket):
    from uuid import uuid4
    from apps.flight.tasks.queue import StreamQueue
    queue = StreamQueue(unix_socket=redis_socket, stream='prediction:' + uuid4().hex)
    yield queue
    queue.client.delete(queue.stream)
    queue.close()
