from __future__ import annotations
import io
import json
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
import pandas as pd
import pytest
import threading
from flight.v1 import prediction_pb2 as pb
from apps.flight.app import create_app
from services.inference.server import serve
from services.inference.adapters.test_backend import TestBackend

ROOT = Path(__file__).resolve().parents[2]
FLIGHT = {'航班号':'CZ9933','机尾号':'B20E8','机型':'A320',
          '计划起飞站四字码':'ZGGG','计划到达站四字码':'ZSPD',
          '计划离港时间':'2025-05-02 09:30:00','计划到港时间':'2025-05-02 11:45:00',
          '起飞站METAR':'ZGGG 020100Z 08005MPS CAVOK 25/18 Q1012','计划总流量':0}

def free_address():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        return f'127.0.0.1:{s.getsockname()[1]}'

@pytest.fixture(scope='module')
def gateway_address(tmp_path_factory):
    wa, ga = free_address(), free_address()
    server, executor = serve(wa, TestBackend(), 'integration-worker',8)
    config = tmp_path_factory.mktemp('prediction-flow') / 'gateway.json'
    config.write_text(json.dumps({'listen': ga, 'rpc_timeout_ms': 2000,
        # This fixture tests business results, not sub-50ms health latency.
        'health_interval_ms': 1000, 'health_timeout_ms': 500, 'failure_threshold': 3,
        'open_cooldown_ms': 1000, 'minimum_retry_budget_ms': 50,
        'workers': [{'id': 'integration-worker', 'address': wa, 'capacity': 8, 'enabled': True}]}))
    process = subprocess.Popen([str(ROOT/'build/phase2/flight_gateway'), '--config', str(config)],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    channel = grpc.insecure_channel(ga)
    try:
        grpc.channel_ready_future(channel).result(5)
        health = health_pb2_grpc.HealthStub(channel)
        deadline = time.monotonic() + 5
        while health.Check(health_pb2.HealthCheckRequest(service=''), timeout=1).status != health_pb2.HealthCheckResponse.SERVING:
            if time.monotonic() >= deadline:
                raise TimeoutError('gateway health did not become serving')
            time.sleep(.05)
        yield ga
    finally:
        channel.close()
        process.terminate()
        try: process.wait(3)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        server.stop(0).wait()
        executor.shutdown(wait=True)

@pytest.fixture
def app(tmp_path,gateway_address):
    app = create_app({'TESTING':True,'RUNTIME_ROOT':str(tmp_path),'GATEWAY_ADDRESS':gateway_address,
                      'RPC_TIMEOUT_SECONDS':1,'BATCH_WORKERS':2})
    yield app
    app.extensions['prediction_client'].close()

def excel_file(rows):
    f=io.BytesIO()
    pd.DataFrame(rows).to_excel(f,index=False)
    f.seek(0)
    return f,'flights.xlsx'

def test_single_and_excel_use_same_real_gateway_and_preserve_source(app):
    with app.test_client() as c:
        single=c.post('/predict_flight',json=FLIGHT)
        assert single.status_code==200
        one=single.get_json()
        assert one['success'] and one['source']=='TEST'
        assert json.loads(one['prediction'])['实际离港时间']=='2025-05-02T09:40:00+08:00'
        assert one['trace_id'] and one['worker_id']=='integration-worker'
        for endpoint in ('/upload','/api/excel-batch-predict'):
            upload=c.post(endpoint,data={'file':excel_file([FLIGHT])})
            assert upload.status_code==200
            job=upload.get_json()
            assert job['results'][0]['prediction_data']==one['prediction_data']
            assert job['llm_success_count']==0 and job['status']=='SUCCEEDED'
            saved=c.get('/api/jobs/'+job['job_id']).get_json()
            assert saved['results'][0]['trace_id']==job['results'][0]['trace_id']
            normalized=saved['results'][0]['normalized_input']
            assert normalized['departure_metar']==FLIGHT['起飞站METAR']
            assert normalized['planned_total_flow']==0
            assert normalized['planned_off_block']=='2025-05-02T01:30:00Z'
            downloaded=c.get('/api/jobs/'+job['job_id']+'/download')
            assert downloaded.status_code==200
            frame=pd.read_excel(io.BytesIO(downloaded.data))
            assert frame.loc[0,'预测来源']=='TEST'
            assert {'机场流量预测', '天气过期', '航班降级', '流量降级',
                    '数据版本', 'Prompt版本'} <= set(frame.columns)

def test_partial_batch_and_sse_have_consistent_failure_counts(app):
    bad={**FLIGHT,'计划到港时间':'bad'}
    with app.test_client() as c:
        batch=c.post('/predict_flights_batch',json={'flights':[FLIGHT,bad]}).get_json()
        assert (batch['status'],batch['success_count'],batch['failed_count'])==('PARTIAL',1,1), batch
        assert batch['results'][1]['error_code']=='INVALID_ARGUMENT'
        stream=c.post('/predict_flights_stream',json={'flights':[FLIGHT,bad]})
        events=[json.loads(x[6:]) for x in stream.data.decode().split('\n\n') if x.startswith('data: ')]
        final=events[-1]
        assert final['type']=='complete' and final['summary']=={'total':2,'success':1,'failed':1}
        assert final['sources']==['TEST']
        job=c.get('/api/jobs/'+final['job_id']).get_json()
        assert job['status']=='PARTIAL' and len(job['results'])==2

def test_jobs_and_timeline_are_isolated_with_concurrent_requests(app):
    def submit(number):
        with app.test_client() as c: return c.post('/predict_flight',json={**FLIGHT,'航班号':number}).get_json()
    with ThreadPoolExecutor(2) as executor: a,b=list(executor.map(submit,['CZ1111','CZ2222']))
    assert a['job_id']!=b['job_id']
    with app.test_client() as c:
        for result,expected in [(a,'CZ1111'),(b,'CZ2222')]:
            response=c.get('/api/jobs/'+result['job_id']+'/timeline')
            assert response.status_code==200
            entries=[x for items in response.get_json()['departure_flights'].values() for x in items]
            assert [x['航班号'] for x in entries]==[expected]
        assert c.get('/api/timeline-data').status_code==400
        assert c.get('/api/latest-results').get_json().get('predictions') is None
        assert c.get('/api/jobs/not-a-uuid').status_code==400
        assert c.get('/api/jobs/00000000-0000-0000-0000-000000000000').status_code==404

def test_unavailable_gateway_never_returns_fake_prediction(tmp_path):
    app=create_app({'TESTING':True,'RUNTIME_ROOT':str(tmp_path),'GATEWAY_ADDRESS':free_address(),'RPC_TIMEOUT_SECONDS':0.2})
    try:
        with app.test_client() as c:
            response=c.post('/predict_flight',json=FLIGHT)
            body=response.get_json()
            assert response.status_code==503
            assert not body['success'] and 'prediction' not in body
            assert body['error_code']=='UNAVAILABLE'
            assert c.get('/api/jobs/'+body['job_id']).get_json()['status']=='FAILED'
    finally: app.extensions['prediction_client'].close()

@pytest.mark.parametrize('payload',[None,{},[],{'flights':'bad'}])
def test_invalid_batch_envelope_returns_400(app,payload):
    with app.test_client() as c: assert c.post('/predict_flights_batch',json=payload).status_code==400

def test_preserves_pages_and_does_not_load_model_dependencies(app):
    with app.test_client() as c:
        for path in ('/','/flight_input','/timeline','/scenario','/overview'):
            assert c.get(path).status_code==200
        assert c.get('/health').get_json()['fallback_enabled'] is False
    assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules

def test_cancelled_stream_records_unfinished_rows_as_failures(app):
    with app.test_client() as c:
        response=c.post('/predict_flights_stream',json={'flights':[FLIGHT]*5},buffered=False)
        first=next(response.response)
        row=json.loads(first.decode().strip()[6:])
        response.close()
        saved=c.get('/api/jobs/'+row['job_id']).get_json()
        assert saved['status'] in ('PARTIAL','SUCCEEDED')
        assert saved['completed_count']==5


def test_json_and_excel_keep_flight_nature_in_saved_and_exported_results(app):
    sample = {**FLIGHT, '性质': 'J'}
    with app.test_client() as client:
        single = client.post('/predict_flight', json=sample).get_json()
        assert single['flight_info']['性质'] == 'J'
        job = client.post('/api/excel-batch-predict', data={'file': excel_file([sample])}).get_json()
        assert job['results'][0]['normalized_input']['flight_nature'] == 'J'
        exported = pd.read_excel(io.BytesIO(client.get(job['download_url']).data))
        assert exported.iloc[0]['性质'] == 'J'


def test_manual_input_has_one_optional_nature_field(app):
    from html.parser import HTMLParser
    class Inputs(HTMLParser):
        nature = []
        def handle_starttag(self, tag, attributes):
            attrs = dict(attributes)
            if tag == 'input' and attrs.get('name') in ('nature', 'flight_nature'):
                self.nature.append(attrs)
    with app.test_client() as client:
        response = client.get('/flight_input')
        assert response.status_code == 200
        page = response.get_data(as_text=True)
    parser = Inputs(); parser.feed(page)
    assert len(parser.nature) == 1
    assert 'required' not in parser.nature[0]
    assert page.count("flightData['性质'] =") == 1


def test_service_propagates_normalized_context_and_flow_provenance():
    from apps.flight.services.prediction import PredictionService
    captured = {}
    class Client:
        def predict(self, request, _cancel):
            captured['request'] = request
            response = pb.PredictResponse(trace_id=request.trace_id, source=pb.TEST,
                model_version='flight-v2', worker_id='worker-v2', data_version='snapshot-v1',
                prompt_version='zggg-weather-duration-prompt-v2', weather_stale=True)
            for field, value in (("off_block", "2025-05-02T01:40:00Z"),
                                 ("takeoff", "2025-05-02T01:50:00Z"),
                                 ("landing", "2025-05-02T03:45:00Z"),
                                 ("on_block", "2025-05-02T03:55:00Z")):
                getattr(response.prediction, field).FromJsonString(value)
            response.airport_flow.model_version = 'flow-v1'
            response.airport_flow.windows.add(horizon_minutes=15, takeoff=2,
                                               landing=3, total=5)
            return response
    payload = {**FLIGHT, 'zggg_context': {
        'prediction_cutoff_time': '2025-05-02 09:30:00', 'data_version': 'snapshot-v1',
        'departure_weather': {'metar': {'kind': 'METAR', 'airport': 'ZGGG',
            'issue_time': '2025-05-02T01:20:00Z', 'features': {'visibility_m': 3000}}},
        'flow_features': [{'horizon_minutes': 15, 'planned_takeoff': 2,
                           'planned_landing': 3}],
    }}
    result = PredictionService(Client(), None)._one(
        0, payload, {'job_id': 'job', 'input_timezone': 'Asia/Shanghai'}, threading.Event())
    assert result['success'] and result['airport_flow']['windows'][0]['total'] == 5
    assert result['weather_stale'] is True and result['data_version'] == 'snapshot-v1'
    assert result['normalized_context']['data_version'] == 'snapshot-v1'
    assert captured['request'].SerializeToString()


@pytest.mark.parametrize('mismatch', ['model', 'flow', 'source'])
def test_mismatched_worker_release_never_persists_success(mismatch):
    from apps.flight.services.prediction import PredictionService
    import threading

    class WrongReleaseClient:
        def predict(self, request, cancel):
            result = pb.PredictResponse(trace_id=request.trace_id,
                source=pb.TEST if mismatch == 'source' else pb.BASELINE,
                model_version=('other-history+schedule' if mismatch == 'model'
                               else 'historical-flight-v1+schedule'), worker_id='wrong-worker')
            for field, value in (('off_block', '2025-05-02T01:40:00Z'),
                                 ('takeoff', '2025-05-02T01:50:00Z'),
                                 ('landing', '2025-05-02T03:45:00Z'),
                                 ('on_block', '2025-05-02T03:55:00Z')):
                getattr(result.prediction, field).FromJsonString(value)
            result.airport_flow.model_version = 'other-flow' if mismatch == 'flow' else 'schedule'
            return result

    job = {'job_id': 'job', 'input_timezone': 'Asia/Shanghai',
           'submission_context': {'release_identity': {
               'source': 'BASELINE', 'model_version': 'historical-flight-v1+schedule',
               'flight_model_version': 'historical-flight-v1',
               'flow_model_version': 'schedule',
           }}}
    result = PredictionService(WrongReleaseClient(), None)._one(0, FLIGHT, job, threading.Event())
    assert result['success'] is False
    assert result['error_code'] == 'DATA_LOSS'
