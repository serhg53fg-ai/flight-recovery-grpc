"""One process per Web instance; gRPC is initialized after worker fork."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
chdir = str(ROOT)
pythonpath = str(ROOT / 'generated')
bind = '127.0.0.1:7101'
workers = 1
worker_class = 'gthread'
threads = 8
preload_app = False
timeout = 90
graceful_timeout = 35
keepalive = 5
accesslog = None
errorlog = '-'
capture_output = True


def worker_exit(server, worker):
    app = getattr(worker, 'wsgi', None)
    client = getattr(app, 'extensions', {}).get('prediction_client')
    if client is not None:
        client.close()
