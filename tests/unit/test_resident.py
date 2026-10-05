from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def settings(tmp_path):
    return {'project': str(ROOT), 'runtime': str(tmp_path), 'backend': 'test',
            'mysqld': '/usr/bin/true', 'redis': '/usr/bin/true', 'nginx': '/usr/bin/true',
            'gateway': '/usr/bin/true', 'experimental_model': False}


def test_config_contains_complete_owned_stack(tmp_path):
    from deploy.distributed.resident import validate, supervisor_config, commands
    config = validate(settings(tmp_path))
    text = supervisor_config(config)
    roles = commands(config)
    assert {'mysql', 'redis', 'worker', 'gateway', 'publisher', 'executor', 'reconcile', 'web-a', 'web-b', 'nginx', 'metrics'} == set(roles)
    assert 'stopasgroup=true' in text and 'killasgroup=true' in text
    assert 'autorestart=unexpected' in text
    assert 'unix_http_server' in text and 'inet_http_server' not in text
    assert '--output-mode' not in roles['worker'] or 'duration_components' in roles['worker']


def test_legacy_worker_config_becomes_one_managed_pool_member(tmp_path):
    from deploy.distributed.resident import validate

    config = validate(settings(tmp_path))
    assert config['workers'] == [{
        'worker_id': 'resident-worker', 'address': '127.0.0.1:50052',
        'capacity': 1, 'release_version': 'test-v1', 'managed': True,
    }]


def test_two_managed_workers_render_distinct_roles_and_gateway_nodes(tmp_path):
    import json
    from deploy.distributed.resident import validate, commands, roles, supervisor_config, write_config

    config = validate({**settings(tmp_path), 'workers': [
        {'worker_id': 'worker-a', 'address': '127.0.0.1:50101',
         'capacity': 2, 'release_version': 'test-v1', 'managed': True},
        {'worker_id': 'worker-b', 'address': '127.0.0.1:50102',
         'capacity': 2, 'release_version': 'test-v1', 'managed': True},
    ]})
    command = commands(config)
    assert {'worker', 'worker-2'} <= command.keys()
    assert command['worker'][command['worker'].index('--worker-id') + 1] == 'worker-a'
    assert command['worker-2'][command['worker-2'].index('--worker-id') + 1] == 'worker-b'
    assert roles(config).index('worker-2') < roles(config).index('gateway')
    assert '[program:worker-2]' in supervisor_config(config)
    write_config(config)
    nodes = json.loads((tmp_path / 'gateway.json').read_text())['workers']
    assert [node['id'] for node in nodes] == ['worker-a', 'worker-b']


def test_second_worker_role_has_no_prerequisite(tmp_path):
    from deploy.distributed.resident import validate
    from deploy.distributed.resident_role import wait_dependencies

    config = validate({**settings(tmp_path), 'workers': [
        {'worker_id': 'a', 'address': '127.0.0.1:50101',
         'capacity': 1, 'release_version': 'test-v1', 'managed': True},
        {'worker_id': 'b', 'address': '127.0.0.1:50102',
         'capacity': 1, 'release_version': 'test-v1', 'managed': True},
    ]})
    wait_dependencies(config, 'worker-2', timeout=.01)


def test_mixed_release_pool_rejected_before_runtime_mutation(tmp_path):
    from deploy.distributed.resident import validate

    config = {**settings(tmp_path), 'workers': [
        {'worker_id': 'worker-a', 'address': '127.0.0.1:50101',
         'capacity': 1, 'release_version': 'test-v1', 'managed': True},
        {'worker_id': 'worker-b', 'address': '127.0.0.1:50102',
         'capacity': 1, 'release_version': 'other-v1', 'managed': True},
    ]}
    with pytest.raises(ValueError, match='release_version'):
        validate(config)
    assert not (tmp_path / 'stack.json').exists()


def test_four_field_external_worker_defaults_to_unmanaged(tmp_path):
    from deploy.distributed.resident import validate

    node = {'worker_id': 'remote-a', 'address': '127.0.0.1:50111',
            'capacity': 3, 'release_version': 'test-v1'}
    config = validate({**settings(tmp_path), 'workers': [node]})
    assert config['workers'][0]['managed'] is False
    assert 'managed' not in node
    from deploy.distributed.resident import roles, commands
    assert 'worker' not in roles(config)
    assert 'worker' not in commands(config)


def test_qwen_without_experiment_acknowledgement_is_rejected(tmp_path):
    from deploy.distributed.resident import validate
    with pytest.raises(ValueError, match='experimental'):
        validate({**settings(tmp_path), 'backend':'qwen','model_path':str(tmp_path)})


def test_composite_rejects_release_artifact_mismatch_before_stop(tmp_path, monkeypatch):
    import hashlib
    import json
    from deploy.distributed import resident

    artifact = tmp_path / 'model.bin'
    artifact.write_bytes(b'current')
    release = tmp_path / 'candidate.json'
    release.write_text(json.dumps({
        'schema_version': 'zggg-release-v2', 'airport': 'ZGGG',
        'flight_mode': 'historical', 'source': 'BASELINE',
        'release_version': 'history-v1', 'flight_model_version': 'historical-flight-v1',
        'flow_model_version': 'gradient_boosting', 'model_version': 'historical-flight-v1+gradient_boosting',
        'feature_contract_version': 'zggg-airport-flow-v1', 'prompt_version': 'historical-duration-v1',
        'dataset_manifest_hash': 'dataset-v1',
        'flight_gate_passed': True, 'flow_gate_passed': True,
        'artifacts': {
            name: {'path': str(artifact), 'sha256': hashlib.sha256(b'original').hexdigest()}
            for name in ('historical_model', 'flow_model', 'feature_contract')},
    }))
    calls = []
    monkeypatch.setattr(resident, 'stop', lambda *_: calls.append('stop'))
    with pytest.raises(ValueError, match='artifact hash'):
        resident.validate({**settings(tmp_path), 'backend': 'composite',
                           'release_manifest_path': str(release)})
    assert calls == []


def test_composite_uses_only_approved_zggg_artifacts(tmp_path):
    import json
    from deploy.distributed import resident
    from scripts.zggg_release import sha256_path

    model = tmp_path / 'model'; model.mkdir(); (model / 'config.json').write_text('{}')
    adapter = tmp_path / 'adapter'; adapter.mkdir()
    (adapter / 'adapter_metadata.json').write_text(json.dumps({
        'adapter_version': 'zggg-adapter-v2',
    }))
    flow = tmp_path / 'flow.pkl'; flow.write_bytes(b'flow')
    historical = tmp_path / 'historical.pkl'; historical.write_bytes(b'history')
    contract = tmp_path / 'metrics.json'; contract.write_text(json.dumps({
        'dataset_manifest_hash': 'dataset-v1', 'feature_contract': ['cutoff_hour'],
    }))
    paths = {'model': model, 'adapter': adapter, 'flow_model': flow,
             'historical_model': historical, 'feature_contract': contract}
    release = tmp_path / 'candidate.json'
    release.write_text(json.dumps({
        'schema_version': 'zggg-release-v2', 'airport': 'ZGGG',
        'flight_mode': 'qwen', 'source': 'LLM',
        'release_version': 'qwen-v1', 'flight_model_version': 'qwen-model-v1',
        'flow_model_version': 'gradient_boosting', 'model_version': 'qwen-model-v1+gradient_boosting',
        'feature_contract_version': 'zggg-airport-flow-v1', 'prompt_version': 'zggg-weather-duration-prompt-v2',
        'dataset_manifest_hash': 'dataset-v1', 'flight_gate_passed': True,
        'flow_gate_passed': True, 'artifacts': {
            name: {'path': str(path), 'sha256': sha256_path(path)}
            for name, path in paths.items()
        },
    }))

    config = resident.validate({**settings(tmp_path / 'runtime'), 'backend': 'composite',
                                'release_manifest_path': str(release)})
    worker = resident.commands(config)['worker']
    assert worker[worker.index('--model-path') + 1] == str(model)
    assert worker[worker.index('--flow-model-path') + 1] == str(flow)
    assert worker[worker.index('--historical-model-path') + 1] == str(historical)
    assert worker[worker.index('--flow-manifest-hash') + 1] == 'dataset-v1'


@pytest.mark.parametrize('change', [
    {'runtime':'/srv/flight-example/user/grpc/live'}, {'runtime':'/tmp/bad%path'},
    {'http_port':80}, {'http_port':7101}, {'http_port':True}, {'backend':'invalid'}])
def test_reject_unsafe_configuration(tmp_path, change):
    from deploy.distributed.resident import validate
    with pytest.raises(ValueError):
        validate({**settings(tmp_path), **change})


def test_supervisor_socket_path_is_checked_before_prepare(tmp_path):
    from deploy.distributed.resident import validate

    base = str(tmp_path)
    runtime = tmp_path / ('x' * (101 - len(base) - len('/supervisor.sock') - 1))
    assert len(str(runtime / 'mysql.sock').encode()) <= 100
    assert len(str(runtime / 'supervisor.sock').encode()) > 100
    with pytest.raises(ValueError, match='socket path'):
        validate(settings(runtime))


def test_unhealthy_dependency_prevents_exec(tmp_path):
    from deploy.distributed.resident_role import wait_dependencies
    with pytest.raises(TimeoutError):
        wait_dependencies(settings(tmp_path), 'executor', timeout=.001,
                          probe=lambda config, dependency: False)


def test_dependencies_are_checked_for_executor_before_start(tmp_path):
    from deploy.distributed.resident_role import wait_dependencies
    seen = []
    wait_dependencies(settings(tmp_path), 'executor', timeout=1,
                      probe=lambda config, dependency: seen.append(dependency) or True)
    assert seen == ['mysql','redis','gateway']


def test_mysql_initialization_failure_cleans_up_process(tmp_path, monkeypatch):
    from deploy.distributed import resident
    c=resident.validate(settings(tmp_path)); resident.write_config(c)
    (tmp_path/'mysql-data/mysql').mkdir()
    events=[]
    class Process:
        def poll(self): return None
        def terminate(self): events.append('terminate')
        def wait(self, timeout): events.append('wait')
    monkeypatch.setattr(resident.subprocess,'Popen',lambda *a, **k:Process())
    monkeypatch.setattr(resident,'wait_mysql',lambda config, process: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(TimeoutError): resident.initialize(c)
    assert events == ['terminate','wait']


def test_binary_symlink_name_is_preserved_for_multicall_redis(tmp_path):
    from deploy.distributed.resident import validate, commands
    binary=tmp_path/'redis-server'
    binary.symlink_to('/usr/bin/true')
    c=validate({**settings(tmp_path),'redis':str(binary)})
    assert commands(c)['redis'][0] == str(binary)


def test_prepare_refuses_active_stack_before_writing_config(tmp_path, monkeypatch):
    import json
    from deploy.distributed import resident
    config=settings(tmp_path); (tmp_path/'supervisor.sock').write_text('active')
    source=tmp_path/'input.json'; source.write_text(json.dumps(config))
    writes=[]
    monkeypatch.setattr(resident,'write_config',lambda c:writes.append('write'))
    monkeypatch.setattr(resident,'initialize',lambda c:None)
    assert resident.main(['prepare','--config',str(source)]) == 2
    assert writes == []


def test_health_fails_when_executor_is_not_running(tmp_path, monkeypatch, capsys):
    import json
    from deploy.distributed import resident, resident_role
    monkeypatch.setattr(resident_role,'probe_dependency',lambda c, role, address=None:True)
    monkeypatch.setattr(resident,'process_states',lambda c:{role:'STOPPED' if role=='executor' else 'RUNNING' for role in resident.ROLES},raising=False)
    assert resident.check_health(resident.validate(settings(tmp_path))) == 1
    assert json.loads(capsys.readouterr().out)['healthy'] is False


def test_operations_reject_changed_external_configuration(tmp_path, monkeypatch):
    import json
    from deploy.distributed import resident
    config=settings(tmp_path); prepared=resident.validate(config); resident.write_config(prepared)
    source=tmp_path/'input.json'; source.write_text(json.dumps({**config,'http_port':8081}))
    calls=[]
    monkeypatch.setattr(resident,'check_health',lambda c:calls.append(c) or 0)
    assert resident.main(['health','--config',str(source)]) == 2
    assert calls == []


def test_paths_with_spaces_are_rejected(tmp_path):
    from deploy.distributed.resident import validate
    with pytest.raises(ValueError): validate({**settings(tmp_path),'runtime':str(tmp_path/'has space')})


def test_supervisor_health_has_bounded_timeout():
    import socket, tempfile, threading, time
    from deploy.distributed.resident import process_states
    with tempfile.TemporaryDirectory(prefix='frpc-') as root:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
            listener.bind(str(Path(root)/'supervisor.sock')); listener.listen(1)
            def hold_response():
                connection,_=listener.accept()
                with connection: time.sleep(.2)
            thread=threading.Thread(target=hold_response,daemon=True); thread.start()
            started=time.monotonic()
            try:
                assert process_states({'runtime':root},timeout=.02)=={}
                assert time.monotonic()-started < .15
            finally: thread.join(.5)


def test_port_check_allows_closed_connections_in_time_wait():
    import socket
    from deploy.distributed import resident
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',0)); port=listener.getsockname()[1]; listener.listen(1)
        with socket.create_connection(('127.0.0.1',port)) as client:
            accepted,_=listener.accept()
            accepted.close()
            assert client.recv(1)==b''
    resident.check_ports([port])


def test_port_check_rejects_active_listener():
    import socket
    from deploy.distributed import resident
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',0)); listener.listen(1)
        with pytest.raises(OSError): resident.check_ports([listener.getsockname()[1]])


def test_stop_waits_for_supervisor_exit_and_keeps_data(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from deploy.distributed import resident
    (tmp_path/'supervisor.pid').write_text('123456')
    data=tmp_path/'mysql-data'; data.mkdir(); (data/'keep').write_text('task')
    checks=[]; states=iter([True,True,False])
    monkeypatch.setattr(resident,'pid_alive',lambda pid:checks.append(pid) or next(states),raising=False)
    monkeypatch.setattr(resident,'control',lambda c,*a:SimpleNamespace(returncode=0))
    monkeypatch.setattr(resident,'owned_supervisor_pid',lambda c,pid:True)
    assert resident.stop({'runtime':str(tmp_path)},timeout=1)==0
    assert checks==[123456]*3
    assert (data/'keep').read_text()=='task'


def test_missing_process_is_reported_as_stopped():
    from deploy.distributed.resident import pid_alive
    assert pid_alive(2147483647) is False


def test_foreign_pid_never_stopped(tmp_path, monkeypatch):
    import os
    from deploy.distributed import resident

    (tmp_path / 'supervisor.pid').write_text(str(os.getpid()))
    monkeypatch.setattr(resident, 'control',
                        lambda *_: (_ for _ in ()).throw(AssertionError('foreign process signalled')))
    with pytest.raises(ValueError, match='owned Supervisor'):
        resident.stop({'runtime': str(tmp_path)})


def test_running_processes_do_not_make_unready_stack_healthy(tmp_path, monkeypatch, capsys):
    import json
    from deploy.distributed import resident, resident_role
    config = resident.validate(settings(tmp_path))
    monkeypatch.setattr(resident, 'process_states',
                        lambda c: {role: 'RUNNING' for role in resident.roles(c)})
    monkeypatch.setattr(resident_role, 'probe_dependency',
                        lambda c, role, address=None: role != 'readiness')
    assert resident.check_health(config) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['healthy'] is False
    assert report['services']['readiness'] is False
