"""Owned single-instance experimental business stack configuration."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import re

from deploy.distributed.release import load_release, release_identity

from deploy.distributed.nginx import safe_path, render_config

PORTS = {'http_port':8080, 'web_a_port':7101, 'web_b_port':7102,
         'worker_port':50052, 'gateway_port':50051, 'metrics_port':9097}
ROLES = ('mysql','redis','worker','gateway','publisher','executor','reconcile','web-a','web-b','nginx','metrics')


def roles(c):
    managed = sum(worker['managed'] for worker in c['workers'])
    workers = (('worker',) + tuple(f'worker-{index}' for index in range(2, managed + 1))
               if managed else ())
    return ROLES[:2] + workers + ROLES[3:]


def worker_roles(c):
    return [(('worker' if index == 1 else f'worker-{index}'), worker)
            for index, worker in enumerate((w for w in c['workers'] if w['managed']), 1)]


def checked_path(value, preserve_name=False):
    path = safe_path(value)
    if preserve_name:
        path = Path(value).absolute()
    if '%' in str(path) or "'" in str(path) or any(char.isspace() for char in str(path)):
        raise ValueError('unsupported deployment path')
    return str(path)


def validate(raw, *, verify_artifacts=True):
    config = dict(raw)
    for field in ('project','runtime','mysqld','redis','nginx','gateway'):
        config[field] = checked_path(config[field], field in ('mysqld','redis','nginx','gateway'))
    for field in ('mysqldump', 'mysql_client'):
        if config.get(field):
            config[field] = checked_path(config[field], True)
    if config.get('backend') not in ('test','qwen','composite'):
        raise ValueError('invalid backend')
    for name, default in PORTS.items():
        value = config.setdefault(name, default)
        if type(value) is not int or not 1024 <= value <= 65535:
            raise ValueError('invalid deployment port')
    if len({config[k] for k in PORTS}) != len(PORTS):
        raise ValueError('deployment ports collide')
    for socket_name in ('mysql.sock', 'redis.sock', 'supervisor.sock'):
        if len(str(Path(config['runtime']) / socket_name).encode()) > 100:
            raise ValueError('runtime socket path is too long')
    if config['backend'] == 'qwen':
        if config.get('experimental_model') is not True:
            raise ValueError('explicit experimental_model=true is required; accuracy gate is not passed')
        config['model_path'] = checked_path(config['model_path'])
        if config.get('adapter_path'):
            config['adapter_path'] = checked_path(config['adapter_path'])
    if config['backend'] == 'composite':
        release_path = checked_path(config['release_manifest_path'])
        release = load_release(Path(release_path), verify_artifacts=verify_artifacts)
        if release.get('deployment_stage') == 'experimental' and config.get('experimental_model') is not True:
            raise ValueError('explicit experimental_model=true is required for experimental release')
        artifacts = release['artifacts']
        config['release_manifest_path'] = release_path
        config['dataset_manifest_hash'] = release['dataset_manifest_hash']
        config['release_identity'] = release_identity(release)
        config['flight_mode'] = release['flight_mode']
        for name in artifacts:
            config[f'{name}_path'] = checked_path(artifacts[name]['path'])
        if config.get('replay_root') or config.get('replay_dataset_id'):
            if not config.get('replay_root') or not config.get('replay_dataset_id'):
                raise ValueError('replay dataset and release must be configured together')
            from apps.flight.context.replay import load_replay_dataset
            config['replay_root'] = checked_path(config['replay_root'])
            dataset = load_replay_dataset(config['replay_dataset_id'], Path(config['replay_root']))
            if (dataset['dataset_manifest_hash'] != config['release_identity']['dataset_manifest_hash'] or
                    dataset['feature_contract_version'] != config['release_identity']['feature_contract_version']):
                raise ValueError('replay dataset does not match release identity')
    version = config['release_identity']['release_version'] if config['backend'] == 'composite' else 'test-v1'
    if config['backend'] == 'qwen':
        version = 'experimental-qwen-v1'
    if 'workers' not in config:
        config['workers'] = [{'worker_id': 'resident-worker',
                              'address': f"127.0.0.1:{config['worker_port']}",
                              'capacity': 1, 'release_version': version, 'managed': True}]
    workers = config['workers']
    if not isinstance(workers, list) or not workers or len(workers) > 32:
        raise ValueError('workers must be a nonempty bounded list')
    config['workers'] = [{**worker, 'managed': worker.get('managed', False)}
                         if isinstance(worker, dict) else worker for worker in workers]
    workers = config['workers']
    seen_ids, seen_addresses = set(), set()
    reserved_ports = {config[key] for key in ('http_port', 'web_a_port', 'web_b_port', 'gateway_port', 'metrics_port')}
    for worker in workers:
        if not isinstance(worker, dict) or set(worker) != {'worker_id','address','capacity','release_version','managed'}:
            raise ValueError('invalid worker fields')
        name, address = worker['worker_id'], worker['address']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name) or name in seen_ids:
            raise ValueError('invalid or duplicate worker_id')
        match = re.fullmatch(r'127\.0\.0\.1:(\d{1,5})', address) if isinstance(address, str) else None
        if not match or not 1024 <= int(match.group(1)) <= 65535 or address in seen_addresses or int(match.group(1)) in reserved_ports:
            raise ValueError('invalid or duplicate worker address')
        if type(worker['capacity']) is not int or not 1 <= worker['capacity'] <= 256:
            raise ValueError('invalid worker capacity')
        if type(worker['managed']) is not bool:
            raise ValueError('invalid worker managed flag')
        if worker['release_version'] != version:
            raise ValueError('worker release_version does not match approved release')
        seen_ids.add(name); seen_addresses.add(address)
    return config


def environment(c):
    runtime = Path(c['runtime'])
    version = 'test-v1'
    source, prompt_version = 'TEST', 'test-v1'
    if c['backend'] == 'composite':
        identity = c['release_identity']
        version, source, prompt_version = (identity[key] for key in ('model_version', 'source', 'prompt_version'))
    elif c['backend'] == 'qwen':
        version = Path(c['model_path']).name + ':flight-duration-prompt-v1'
        if c.get('adapter_path'):
            metadata = json.loads((Path(c['adapter_path'])/'adapter_metadata.json').read_text())
            version = (Path(c['model_path']).name+'+'+metadata['adapter_version'])[:128]
        source, prompt_version = 'LLM', 'flight-duration-prompt-v1'
    env = {'PYTHONPATH': c['project']+':'+str(Path(c['project'])/'generated'),
            'FLIGHT_STORAGE_BACKEND':'mysql','FLIGHT_MYSQL_USER':'root','FLIGHT_MYSQL_PASSWORD':'',
            'FLIGHT_MYSQL_DATABASE':'flight_resident','FLIGHT_MYSQL_UNIX_SOCKET':str(runtime/'mysql.sock'),
            'FLIGHT_REDIS_UNIX_SOCKET':str(runtime/'redis.sock'),'FLIGHT_REDIS_STREAM':'flight:resident',
            'FLIGHT_DURABLE_ENABLED':'1','FLIGHT_RECOVERY_ENABLED':'1',
            'FLIGHT_EXPERIMENTAL_MODEL':'1' if c.get('experimental_model') is True else '0',
            'FLIGHT_DURABLE_MODEL_VERSION':version,'FLIGHT_DURABLE_SOURCE':source,
            'FLIGHT_DURABLE_PROMPT_VERSION':prompt_version,
            'FLIGHT_GATEWAY_ADDRESS':f"127.0.0.1:{c['gateway_port']}",
            'FLIGHT_RPC_TIMEOUT_SECONDS':'60' if c.get('release_identity', {}).get('deployment_stage') == 'experimental' else '20',
            'FLIGHT_RPC_TIMEOUT':'60' if c.get('release_identity', {}).get('deployment_stage') == 'experimental' else '20'}
    if c['backend'] == 'composite' and c.get('replay_dataset_id'):
        env.update(FLIGHT_REPLAY_ROOT=c['replay_root'],
                   FLIGHT_REPLAY_DATASET_ID=c['replay_dataset_id'],
                   FLIGHT_RELEASE_MANIFEST_PATH=c['release_manifest_path'])
    return env


def commands(c):
    r = Path(c['runtime']); python = str(Path(c['project'])/'.venv/bin/python')
    result = {'mysql':[c['mysqld'],'--no-defaults',f'--datadir={r / "mysql-data"}',
              f'--socket={r / "mysql.sock"}',f'--pid-file={r / "mysql.pid"}',
              f'--log-error={r / "mysql.log"}','--skip-networking','--mysqlx=OFF',
              '--skip-log-bin','--secure-file-priv=NULL','--innodb-buffer-pool-size=128M'],
              'redis':[c['redis'],str(r/'redis.conf')],
              'gateway':[c['gateway'],'--config',str(r/'gateway.json')],
              'metrics':[python,'-m','deploy.metrics.exporter','--listen-port',str(c['metrics_port']),
                         '--gateway-address',f"127.0.0.1:{c['gateway_port']}",
                         '--mysql-socket',str(r/'mysql.sock')],
              'nginx':[c['nginx'],'-p',str(r/'nginx')+'/', '-c',str(r/'nginx/nginx.conf')]}
    for role, node in worker_roles(c):
        worker = [python,'-m','deploy.autodl.worker','--listen',node['address'],
                  '--backend',c['backend'],'--worker-id',node['worker_id'],
                  '--capacity',str(node['capacity']),'--max-seconds',
                  '45' if c.get('release_identity', {}).get('deployment_stage') == 'experimental' else '20']
        if c['backend'] == 'qwen' or (c['backend'] == 'composite' and c['flight_mode'] == 'qwen'):
            worker += ['--model-path',c['model_path'],'--output-mode','duration_components']
            if c.get('adapter_path'): worker += ['--adapter-path',c['adapter_path']]
        if c['backend'] == 'composite':
            worker += ['--flow-model-path',c['flow_model_path'],
                       '--flow-manifest-hash',c['dataset_manifest_hash'],
                       '--historical-model-path',c['historical_model_path'],
                       '--composite-flight-mode',c['flight_mode'],
                       '--expected-model-version',c['release_identity']['model_version'],
                       '--expected-source',c['release_identity']['source']]
        result[role] = worker
    if os.geteuid()==0: result['mysql'] += ['--user=root']
    for role in ('publisher','reconcile'):
        result[role]=[python,'-m','scripts.durable_prediction',role]
    result['executor']=[python,'-m','scripts.durable_prediction','executor','--owner','resident-executor','--lease','30']
    for role, port in (('web-a',c['web_a_port']),('web-b',c['web_b_port'])):
        result[role]=[python,'-m','gunicorn','-c','deploy/distributed/gunicorn.conf.py',
                      '--bind',f'127.0.0.1:{port}','deploy.distributed.wsgi:create_production_app()']
    return result


def supervisor_config(c):
    r=Path(c['runtime']); python=str(Path(c['project'])/'.venv/bin/python')
    text=f'''[supervisord]
nodaemon=false
umask=0077
logfile={r}/supervisor.log
pidfile={r}/supervisor.pid
childlogdir={r}/logs
[unix_http_server]
file={r}/supervisor.sock
chmod=0600
[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix://{r}/supervisor.sock
'''
    for index,role in enumerate(roles(c)):
        argv=[python,'-m','deploy.distributed.resident_role','--config',str(r/'stack.json'),'--role',role]
        text+=f'''\n[program:{role}]
directory={c['project']}
command={shlex.join(argv)}
priority={10+index}
autostart=true
autorestart=unexpected
startsecs=3
startretries=10
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=40
stdout_logfile={r}/logs/{role}.log
stderr_logfile={r}/logs/{role}-error.log
stdout_logfile_maxbytes=10MB
stderr_logfile_maxbytes=10MB
stdout_logfile_backups=3
stderr_logfile_backups=3
'''
    return text


def write_config(c):
    r=Path(c['runtime'])
    if r.is_symlink(): raise ValueError('runtime cannot be a symlink')
    r.mkdir(parents=True,exist_ok=True); r.chmod(0o700)
    for name in ('logs','mysql-data','redis-data','nginx','web-a','web-b'):
        path=r/name
        if path.is_symlink(): raise ValueError('output cannot be a symlink')
        path.mkdir(exist_ok=True)
    for name in ('body','proxy','fastcgi','uwsgi','scgi'): (r/'nginx'/name).mkdir(exist_ok=True)
    output={'stack.json':json.dumps(c,indent=2), 'supervisor.conf':supervisor_config(c),
      'gateway.json':json.dumps({'listen':f"127.0.0.1:{c['gateway_port']}",
         'rpc_timeout_ms':60000 if c.get('release_identity', {}).get('deployment_stage') == 'experimental' else 20000,
         'health_interval_ms':1000,'health_timeout_ms':200,'failure_threshold':3,'open_cooldown_ms':2000,
         'minimum_retry_budget_ms':50,'workers':[{'id':node['worker_id'],'address':node['address'],
             'capacity':node['capacity'],'enabled':True} for node in c['workers']]}),
      'redis.conf':f'port 0\nunixsocket {r}/redis.sock\nunixsocketperm 700\ndir {r}/redis-data\nsave ""\nappendonly yes\nappendfsync everysec\nmaxmemory 128mb\nmaxmemory-policy noeviction\ndaemonize no\n',
      'nginx/nginx.conf':render_config(c['project'],r/'nginx',c['http_port'],c['web_a_port'],c['web_b_port'])}
    for name,text in output.items():
        path=r/name
        if path.is_symlink() or (path.exists() and path.stat().st_nlink>1):
            raise ValueError('output must be a private regular file')
        path.write_text(text); path.chmod(0o600)


def wait_mysql(c, process, timeout=30):
    import time
    import pymysql
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if process.poll() is not None: raise RuntimeError('owned MySQL exited; inspect mysql.log')
        try:
            connection=pymysql.connect(unix_socket=str(Path(c['runtime'])/'mysql.sock'),user='root',connect_timeout=1)
            connection.close(); return
        except pymysql.MySQLError: time.sleep(.2)
    raise TimeoutError('owned MySQL startup timed out')


def initialize(c):
    r=Path(c['runtime']); data=r/'mysql-data'
    if (r/'supervisor.sock').exists(): raise ValueError('stop the owned stack before prepare')
    if not (data/'mysql').is_dir():
        if any(data.iterdir()): raise ValueError('partial MySQL initialization; inspect data, do not overwrite')
        init=[c['mysqld'],'--no-defaults','--initialize-insecure',f'--datadir={data}','--secure-file-priv=NULL']
        if os.geteuid()==0: init+=['--user=root']
        with (r/'logs/mysql-initialize.log').open('a') as log:
            subprocess.run(init,stdout=log,stderr=log,check=True,timeout=120)
    with (r/'logs/mysql-prepare.log').open('a') as log:
        process=subprocess.Popen(commands(c)['mysql'],stdout=log,stderr=log)
        try:
            wait_mysql(c,process)
            import pymysql
            connection=pymysql.connect(unix_socket=str(r/'mysql.sock'),user='root',autocommit=True)
            try:
                with connection.cursor() as cursor:
                    cursor.execute('CREATE DATABASE IF NOT EXISTS flight_resident CHARACTER SET utf8mb4')
            finally: connection.close()
            sys.path.insert(0,str(Path(c['project'])/'generated'))
            from apps.flight.storage.mysql_jobs import MySQLConfig, initialize_schema
            from apps.flight.tasks.repository import initialize_durable
            from apps.flight.recovery.repository import initialize_schema as initialize_recovery
            config=MySQLConfig(database='flight_resident',user='root',unix_socket=str(r/'mysql.sock'))
            initialize_schema(config); initialize_durable(config); initialize_recovery(config)
            marker=r/'initialized.json'
            marker.write_text(json.dumps({'initialized':True,'experimental_model':c.get('experimental_model',False)}))
            marker.chmod(0o600)
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)


def control(c, *arguments):
    python=str(Path(c['project'])/'.venv/bin/python')
    return subprocess.run([python,'-m','supervisor.supervisorctl','-c',
        str(Path(c['runtime'])/'supervisor.conf'),*arguments],timeout=120)


def pid_alive(pid):
    try:
        os.kill(pid,0)
        return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except (FileNotFoundError,ProcessLookupError):
        return False


def owned_supervisor_pid(c, pid):
    """Check only the PID named by this runtime, without enumerating processes."""
    if pid <= 1 or not pid_alive(pid):
        return False
    try:
        arguments = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError, OSError):
        return False
    config_path = str(Path(c['runtime']) / 'supervisor.conf').encode()
    return (any(b'supervisord' in part for part in arguments) and
            any(config_path in part for part in arguments))


def stop(c, timeout=420):
    import time
    runtime=Path(c['runtime'])
    pid_file = runtime / 'supervisor.pid'
    if pid_file.is_symlink() or not pid_file.is_file():
        raise ValueError('owned Supervisor pid file is missing or linked')
    pid=int(pid_file.read_text().strip())
    if not owned_supervisor_pid(c, pid):
        raise ValueError('owned Supervisor pid does not match this runtime')
    result=control(c,'shutdown')
    if result.returncode: return result.returncode
    deadline=time.monotonic()+timeout
    while pid_alive(pid) or (runtime/'supervisor.sock').exists():
        if time.monotonic()>=deadline: raise TimeoutError('owned Supervisor shutdown timed out')
        time.sleep(.1)
    return 0


def process_states(c, timeout=2):
    import socket
    from supervisor.xmlrpc import SupervisorTransport, UnixStreamHTTPConnection
    from xmlrpc.client import ServerProxy
    class Connection(UnixStreamHTTPConnection):
        def connect(self):
            self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect(self.socketfile)
    transport=SupervisorTransport(None,None,'unix://'+str(Path(c['runtime'])/'supervisor.sock'))
    def get_connection():
        connection=Connection('localhost')
        connection.socketfile=str(Path(c['runtime'])/'supervisor.sock')
        return connection
    transport._get_connection=get_connection
    try:
        with ServerProxy('http://127.0.0.1',transport=transport) as proxy:
            return {row['name']:row['statename'] for row in proxy.supervisor.getAllProcessInfo()}
    except Exception:
        return {}
    finally:
        if transport.connection is not None:
            transport.connection.close()


def check_health(c):
    from deploy.distributed.resident_role import probe_dependency
    result={role:probe_dependency(c,role) for role in ('mysql','redis','gateway','web-a','web-b','nginx','metrics','readiness')}
    result.update({node['worker_id']: probe_dependency(c, 'worker', node['address']) for node in c['workers']})
    states=process_states(c)
    healthy=all(result.values()) and all(states.get(role)=='RUNNING' for role in roles(c))
    print(json.dumps({'healthy':healthy,'services':result,'processes':states,
        'experimental_model':c.get('experimental_model',False)},ensure_ascii=False))
    return 0 if healthy else 1


def check_ports(ports):
    import socket
    for port in ports:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind(('127.0.0.1',port))


def start(c):
    import time
    r=Path(c['runtime'])
    if not (r/'initialized.json').is_file(): raise ValueError('prepare must complete before start')
    if (r/'supervisor.sock').exists(): raise ValueError('owned Supervisor socket exists; inspect status first')
    from deploy.distributed.preflight import check_environment
    preflight = check_environment(c)
    if not preflight['ok']:
        raise ValueError('preflight failed: ' + '; '.join(preflight['errors']))
    python=str(Path(c['project'])/'.venv/bin/python')
    subprocess.run([python,'-m','supervisor.supervisord','-c',str(r/'supervisor.conf')],
        env={**os.environ,**environment(c)},check=True,timeout=15)
    from deploy.distributed.resident_role import probe_dependency
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        states=process_states(c)
        if any(state=='FATAL' for state in states.values()):
            raise RuntimeError('owned role failed to start; inspect status and role logs')
        if probe_dependency(c,'nginx') and all(states.get(role)=='RUNNING' for role in roles(c)):
            return check_health(c)
        time.sleep(.5)
    raise TimeoutError('owned stack did not become ready; inspect status and role logs')


def main(argv=None):
    parser=argparse.ArgumentParser(description='Owned single-instance resident business stack')
    parser.add_argument('action',choices=('preflight','prepare','start','status','health','restart','stop'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--role')
    args=parser.parse_args(argv)
    try:
        # Admission/prepare/start perform the full check in check_environment.
        # Control/status processes only need the already verified release identity.
        c=validate(json.loads(Path(args.config).read_text()), verify_artifacts=False)
        if args.action == 'preflight':
            from deploy.distributed.preflight import check_environment
            result = check_environment(c)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result['ok'] else 2
        if args.action=='prepare':
            if (Path(c['runtime'])/'supervisor.sock').exists():
                raise ValueError('stop the owned stack before prepare; active configuration is unchanged')
            from deploy.distributed.preflight import check_environment
            preflight = check_environment(c)
            if not preflight['ok']:
                raise ValueError('preflight failed: ' + '; '.join(preflight['errors']))
            write_config(c); initialize(c)
            print(json.dumps({'prepared':True,'runtime':c['runtime'],'experimental_model':c.get('experimental_model',False)}))
            return 0
        prepared=validate(json.loads((Path(c['runtime'])/'stack.json').read_text()),
                          verify_artifacts=False)
        if prepared != c:
            raise ValueError('configuration differs from prepared stack; use runtime/stack.json or stop and prepare')
        c=prepared
        if args.action=='start': return start(c)
        if args.action=='health': return check_health(c)
        if args.action=='restart':
            if args.role is None: parser.error('restart requires --role')
            if args.role not in roles(c): raise ValueError('role is not owned by this runtime')
            return control(c,'restart',args.role).returncode
        if args.action=='stop': return stop(c)
        return control(c,'status').returncode
    except (ValueError,OSError,RuntimeError,TimeoutError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        print(f'resident operation failed: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__': raise SystemExit(main())
