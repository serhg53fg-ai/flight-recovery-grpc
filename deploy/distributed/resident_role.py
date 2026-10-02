"""Check bounded dependencies, then replace this process with its owned role."""
import argparse
import json
import os
from pathlib import Path
import threading
import time
from urllib.request import urlopen

from deploy.distributed.resident import validate, environment, commands, roles

DEPENDENCIES = {'mysql':(), 'redis':(), 'worker':(), 'gateway':(),
    'publisher':('mysql','redis'), 'executor':('mysql','redis','gateway'),
    'reconcile':('mysql',), 'web-a':('mysql','gateway'), 'web-b':('mysql','gateway'),
    'nginx':('web-a','web-b'), 'metrics':('mysql','gateway')}


def probe_dependency(c, role, address=None):
    try:
        if role=='mysql':
            r=Path(c['runtime'])
            import pymysql
            connection=pymysql.connect(unix_socket=str(r/'mysql.sock'),user='root',
                database='flight_resident',connect_timeout=1,read_timeout=1,write_timeout=1)
            try:
                with connection.cursor() as cursor: cursor.execute('SELECT 1 FROM storage_schema LIMIT 1')
            finally: connection.close()
            return True
        if role=='redis':
            r=Path(c['runtime'])
            import redis
            client=redis.Redis(unix_socket_path=str(r/'redis.sock'),socket_timeout=1)
            try: return bool(client.ping())
            finally: client.close()
        if role in ('worker','gateway'):
            from deploy.autodl.health import wait_for_serving
            target = address or (c['workers'][0]['address'] if role == 'worker' and c.get('workers')
                                 else f"127.0.0.1:{c[role+'_port']}")
            return wait_for_serving(target,'',.5,threading.Event())
        port=(c['http_port'] if role in ('nginx', 'readiness') else c['web_a_port'] if role=='web-a'
              else c['web_b_port'] if role=='web-b' else c['metrics_port'])
        path='/metrics' if role=='metrics' else '/ready' if role=='readiness' else '/health'
        with urlopen(f'http://127.0.0.1:{port}{path}',timeout=1) as response:
            return response.status==200 and (role != 'readiness' or json.load(response).get('ready') is True)
    except Exception:
        return False


def wait_dependencies(c, role, timeout=180, probe=probe_dependency):
    deadline=time.monotonic()+timeout
    for dependency in DEPENDENCIES.get(role, ()):
        while not probe(c, dependency):
            remaining=deadline-time.monotonic()
            if remaining<=0: raise TimeoutError(f'{role} dependency {dependency} did not become ready')
            time.sleep(min(.2,remaining))


def wait_worker_pool(c, timeout=180):
    """Wait for every configured Worker and reject an identity mismatch before Gateway starts."""
    from deploy.distributed.worker_pool import probe_worker_identity

    deadline = time.monotonic() + timeout
    for worker in c['workers']:
        while not probe_dependency(c, 'worker', worker['address']):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"worker {worker['worker_id']} did not become ready")
            time.sleep(min(.2, remaining))
        if c['backend'] == 'composite':
            identity = c['release_identity']
            probe_worker_identity(worker['address'], worker['worker_id'], {
                'source': identity['source'], 'model_version': identity['model_version'],
                'flow_model_version': identity['flow_model_version'],
            }, timeout=max(.1, min(20 if identity.get('fallback_identity') else 5,
                                  deadline - time.monotonic())),
                fallback_identity=identity.get('fallback_identity'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--role',required=True)
    args=parser.parse_args()
    # Only the model-owning Worker needs to rehash large weights. The stack
    # admission check has already verified every artifact before role startup.
    c=validate(json.loads(Path(args.config).read_text()),
               verify_artifacts=args.role.startswith('worker'))
    if args.role not in roles(c):
        parser.error('role is not owned by this runtime')
    wait_dependencies(c,args.role)
    if args.role == 'gateway':
        wait_worker_pool(c)
    env={**os.environ,**environment(c)}
    if args.role in ('web-a','web-b'):
        env.update(FLIGHT_WEB_INSTANCE=args.role,FLIGHT_RUNTIME_ROOT=str(Path(c['runtime'])/args.role))
    os.chdir(c['project'])
    argv=commands(c)[args.role]
    os.execve(argv[0],argv,env)


if __name__=='__main__': main()
