"""Temporary isolated GPU + business stack acceptance; never publishes model."""
import io, json, os, subprocess, time
from pathlib import Path
from uuid import uuid4
import grpc, pandas as pd, pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc
from tests.integration.production_fixtures import (
    nginx_binary, mysql_server, mysql_settings, mysql_config, redis_socket, queue, repo,
    ProductionStack, start_nginx, wait_http, http)
from tests.integration.test_prediction_flow import free_address, FLIGHT
from services.inference.adapters.qwen import QwenBackend
from services.inference.server import serve
from apps.flight.recovery.repository import initialize_schema
from apps.flight.clients.inference import InferenceClient
from apps.flight.tasks.publisher import OutboxPublisher
from apps.flight.tasks.executor import DurableExecutor
ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.environ.get('FLIGHT_GPU_TESTS') != '1', reason='opt-in real GPU business acceptance')
OUT = ROOT / 'runtime/server-gpu-production-acceptance'

@pytest.fixture(scope='module')
def gpu_gateway():
    OUT.mkdir(exist_ok=True)
    model_path = os.environ.get('FLIGHT_GPU_MODEL_PATH')
    assert model_path, 'FLIGHT_GPU_MODEL_PATH is required'
    backend = QwenBackend(model_path, adapter_path=os.environ.get('FLIGHT_GPU_ADAPTER_PATH'),
        output_mode='duration_components')
    original = backend.predict
    observed = []
    def predict(flight, cancelled, budget):
        assert flight.flight_nature == 'J'
        observed.append(flight.flight_nature)
        return original(flight, cancelled, budget)
    backend.predict = predict
    wa, ga = free_address(), free_address()
    server, executor = serve(wa, backend, 'gpu-business-worker', 1)
    process = channel = log = None
    try:
        config = OUT/'gateway.json'
        config.write_text(json.dumps({'listen':ga,'rpc_timeout_ms':20000,'health_interval_ms':100,
            'health_timeout_ms':50,'failure_threshold':3,'open_cooldown_ms':1000,
            'minimum_retry_budget_ms':50,'workers':[{'id':'gpu-business-worker','address':wa,'capacity':1,'enabled':True}]}))
        log = (OUT/'gateway.log').open('w')
        process = subprocess.Popen([os.environ.get('FLIGHT_GPU_GATEWAY_BINARY', str(ROOT/'build/phase2/flight_gateway')),'--config',str(config)],stdout=log,stderr=log)
        channel = grpc.insecure_channel(ga)
        grpc.channel_ready_future(channel).result(10)
        h = health_pb2_grpc.HealthStub(channel)
        deadline = time.monotonic()+10
        while h.Check(health_pb2.HealthCheckRequest(),timeout=1).status != health_pb2.HealthCheckResponse.SERVING:
            assert time.monotonic()<deadline
            time.sleep(.1)
        yield ga, backend.model_version, observed
    finally:
        if channel is not None:
            channel.close()
        if process is not None:
            process.terminate()
            try: process.wait(5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        if log is not None:
            log.close()
        server.stop(0).wait()
        executor.shutdown(wait=True)


def test_real_gpu_nginx_mysql_redis_closed_loop(nginx_binary,mysql_config,repo,queue,tmp_path,gpu_gateway):
    ga, version, observed = gpu_gateway
    initialize_schema(mysql_config)
    stack = ProductionStack(nginx_binary,tmp_path,mysql_config,ga)
    stack.environment.update(FLIGHT_RPC_TIMEOUT_SECONDS='20',FLIGHT_DURABLE_SOURCE='LLM',
        FLIGHT_DURABLE_MODEL_VERSION=version,FLIGHT_DURABLE_PROMPT_VERSION='flight-duration-prompt-v1')
    client = InferenceClient(ga,20)
    try:
        stack.start_web(0); stack.start_web(1)
        stack.proxy=start_nginx(nginx_binary,tmp_path/'nginx',stack.ports)
        wait_http(stack.url+'/health',stack.proxy)
        sample={**FLIGHT,'性质':'J','计划地面航程_Mile':650,'计划航段时间\n（24年同航季平均值）':135}
        data={'flights':[sample]}
        with http(stack.url+'/api/v1/prediction-jobs','POST',data,{'Idempotency-Key':str(uuid4())}) as reply:
            assert reply.status==202; job=json.load(reply)
        assert OutboxPublisher(repo,queue).run_once()==1
        assert DurableExecutor(repo,queue,client,'gpu-acceptance',30).run_once()
        seen=set()
        for _ in range(4):
            with http(stack.url+job['status_url']) as reply:
                seen.add(reply.headers['X-Flight-Instance']); final=json.load(reply)
                assert final['status']=='SUCCEEDED'
        assert seen=={'web-a','web-b'}
        result=final['results'][0]
        assert result['source']=='LLM' and result['model_version']==version
        assert result['normalized_input']['flight_nature']=='J'
        scenario={'airport':'ZGGG','horizon_start':'2025-05-02T08:00:00+08:00',
            'horizon_end':'2025-05-02T16:00:00+08:00','slot_minutes':15,
            'departure_capacity':1,'arrival_capacity':1,'mtt_minutes':30,'closures':[]}
        with http(stack.url+'/api/v1/recovery-jobs','POST',
                  {'source_job_id':job['job_id'],'scenario':scenario,'scenario_version':'gpu-v1'},
                  {'Idempotency-Key':str(uuid4())}) as reply:
            assert reply.status==201; recovery=json.load(reply)
        assert recovery['status']=='SUCCEEDED' and recovery['validation_errors']==[]
        assert recovery['provenance'][0]['source']=='LLM'
        assert recovery['situation_snapshot']['prediction']['successful']==1
        assert recovery['situation_snapshot']['prediction']['sources']=={'LLM':1}
        assert recovery['summary']['scheduled_count']==1
        with http(stack.url+recovery['download_url']) as reply:
            recovery_content=reply.read(); assert reply.status==200
        with http(stack.url+job['download_url']) as reply:
            content=reply.read(); assert reply.status==200
        frame=pd.read_excel(io.BytesIO(content)); assert frame.iloc[0]['性质']=='J'
        (OUT/'download.xlsx').write_bytes(content)
        (OUT/'recovery.json').write_bytes(recovery_content)
        (OUT/'result.json').write_text(json.dumps({'status':'PASS','production_release':False,
            'web_instances':sorted(seen),'observed_natures':observed,'result':result,
            'recovery_id':recovery['recovery_id'],
            'situation_snapshot_sha256':recovery['situation_snapshot']['snapshot_digest']},
            ensure_ascii=False,indent=2))
    finally:
        client.close(); stack.close()
