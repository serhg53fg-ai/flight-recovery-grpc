import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import Request,urlopen
import grpc
from google.protobuf.empty_pb2 import Empty
from flight.v1 import prediction_pb2_grpc
from tests.integration.test_prediction_flow import FLIGHT,free_address

ROOT=Path(__file__).resolve().parents[2]

def test_launcher_runs_multi_worker_prediction_chain_and_stops(tmp_path):
    ports=[free_address().split(':')[1] for _ in range(3)]
    with (tmp_path/'launcher.log').open('w') as log:
        p=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_local.py'),
            '--backend','test','--web-port',ports[0],'--gateway-port',ports[1],'--worker-port',ports[2],
            '--worker-count','2',
            '--runtime-root',str(tmp_path/'runtime')],cwd=ROOT,stdout=log,stderr=log)
        try:
            base='http://127.0.0.1:'+ports[0]
            deadline=time.monotonic()+20
            while True:
                assert p.poll() is None,(tmp_path/'launcher.log').read_text()
                try:
                    with urlopen(base+'/health',timeout=0.5) as r: assert r.status==200
                    break
                except OSError:
                    if time.monotonic()>deadline: raise
                    time.sleep(0.1)
            selected=set()
            for _ in range(6):
                req=Request(base+'/predict_flight',data=json.dumps(FLIGHT).encode(),headers={'Content-Type':'application/json'})
                with urlopen(req,timeout=5) as r: data=json.load(r)
                assert data['success'] and data['source']=='TEST'
                selected.add(data['worker_id'])
            assert selected=={'local-worker-0','local-worker-1'}
            assert data['prediction_data']['实际离港时间']=='2025-05-02T09:40:00+08:00'
            channel=grpc.insecure_channel('127.0.0.1:'+ports[1])
            status=prediction_pb2_grpc.GatewayAdminStub(channel).GetClusterStatus(Empty(),timeout=2)
            channel.close()
            assert len(status.nodes)==2 and all(node.selected_count>0 for node in status.nodes)
        finally:
            p.send_signal(signal.SIGTERM)
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill();p.wait();raise
    assert p.returncode==0
    for name in ('worker-0','worker-1'):
        assert (tmp_path/'runtime/logs'/f'{name}.log').is_file()
    assert (tmp_path/'runtime'/'gateway.generated.json').is_file()
    gateway_log=(tmp_path/'runtime/logs'/'gateway.log').read_text()
    assert data['trace_id'] in gateway_log
    assert '"attempt"' in gateway_log and '"worker_id"' in gateway_log
