"""Low-cardinality stage records and dependency-aware production readiness."""
import json
import math
import os
from pathlib import Path


STAGES = frozenset(('queue', 'rpc', 'persist', 'total'))
OUTCOMES = frozenset(('success', 'failure'))


def record_stage(stage: str, seconds: float, outcome: str) -> None:
    if stage not in STAGES or outcome not in OUTCOMES:
        raise ValueError('invalid stage or outcome')
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= 3600:
        raise ValueError('invalid stage duration')
    print(json.dumps({'event': 'flight.stage', 'stage': stage,
                      'seconds': float(seconds), 'outcome': outcome}), flush=True)


def _database_ready(app):
    repository = app.extensions.get('durable_repository')
    return repository is not None and repository.ping()


def _queue_ready(app):
    import redis
    socket = app.config.get('REDIS_UNIX_SOCKET')
    if not socket:
        return False
    client = redis.Redis(unix_socket_path=socket, socket_connect_timeout=.5,
                         socket_timeout=.5)
    try:
        return bool(client.ping())
    finally:
        client.close()


def _release_ready(app):
    from deploy.distributed.release import release_identity
    # Only the explicitly configured synthetic TEST deployment has no model release.
    if (app.config.get('DURABLE_ENABLED') is True and
            app.config.get('DURABLE_SOURCE') == 'TEST' and
            app.config.get('DURABLE_MODEL_VERSION') == 'test-v1' and
            app.config.get('DURABLE_PROMPT_VERSION') == 'test-v1' and
            not app.config.get('RELEASE_MANIFEST_PATH') and
            not app.config.get('RELEASE_IDENTITY') and
            not app.config.get('REPLAY_DATASET_ID')):
        return True
    path = app.config.get('RELEASE_MANIFEST_PATH')
    expected = app.config.get('RELEASE_IDENTITY')
    if not path or not expected:
        return False
    release = json.loads(Path(path).read_text(encoding='utf-8'))
    experimental = (app.config.get('EXPERIMENTAL_MODEL') is True and
                    release.get('deployment_stage') == 'experimental' and
                    release.get('flight_gate_passed') is False)
    if (release_identity(release) != expected or release.get('airport') != 'ZGGG' or
            (release.get('flight_gate_passed') is not True and not experimental) or
            release.get('flow_gate_passed') is not True):
        return False
    artifacts = release.get('artifacts')
    return isinstance(artifacts, dict) and bool(artifacts) and all(
        isinstance(record, dict) and Path(record.get('path', '')).exists()
        for record in artifacts.values())


def _worker_pool_ready(app):
    from google.protobuf.empty_pb2 import Empty
    from flight.v1 import prediction_pb2_grpc

    client = app.extensions.get('prediction_client')
    if client is None:
        return False
    status = prediction_pb2_grpc.GatewayAdminStub(client.channel).GetClusterStatus(Empty(), timeout=.5)
    return bool(status.serving and any(node.healthy for node in status.nodes))


def readiness_checks(app):
    probes = app.extensions.get('readiness_probes') or {
        'database': lambda: _database_ready(app),
        'queue': lambda: _queue_ready(app),
        'release': lambda: _release_ready(app),
        'worker_pool': lambda: _worker_pool_ready(app),
    }
    checks = {}
    for name, probe in probes.items():
        try:
            checks[name] = bool(probe())
        except Exception:
            checks[name] = False
    return checks
