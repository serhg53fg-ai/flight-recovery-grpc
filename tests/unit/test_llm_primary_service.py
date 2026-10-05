import threading
from types import SimpleNamespace

import pytest
from flight.v1 import prediction_pb2 as pb
from apps.flight.services.prediction import PredictionService
from apps.flight.api.predictions import rows_for_job
from tests.unit.test_composite_worker import prediction
from tests.unit.test_context_snapshot import FLIGHT


def run_result(*, fallback=True, flow_fallback=False, tamper=None):
    identity = {'source': 'LLM', 'model_version': 'qwen+adapter+gradient_boosting',
                'flight_model_version': 'qwen+adapter', 'flow_model_version': 'gradient_boosting',
                'prompt_version': 'zggg-weather-duration-prompt-v2',
                'fallback_identity': {'source': 'BASELINE', 'flight_model_version': 'historical-flight-v1',
                                      'prompt_version': 'historical-duration-v1'},
                'deployment_stage': 'experimental', 'flight_gate_passed': False}
    flow = 'planned-flow-fallback-v1' if flow_fallback else 'gradient_boosting'
    selected = identity['fallback_identity'] if fallback else identity
    response = pb.PredictResponse(prediction=prediction(), source=pb.BASELINE if fallback else pb.LLM,
        model_version=selected['flight_model_version'] + '+' + flow,
        prompt_version=selected['prompt_version'], worker_id='worker',
        flight_degraded=fallback, flow_degraded=flow_fallback,
        flight_fallback_reason='PRIMARY_TIMEOUT' if fallback else '',
        flow_fallback_reason='FLOW_MODEL_UNAVAILABLE' if flow_fallback else '')
    response.airport_flow.model_version = flow
    if tamper:
        tamper(response)
    def predict(request, cancel):
        response.trace_id = request.trace_id
        return response
    job = {'job_id': 'job', 'input_timezone': 'Asia/Shanghai', 'source': 'LLM',
           'model_version': identity['model_version'],
           'submission_context': {'release_identity': identity}}
    row = PredictionService(SimpleNamespace(predict=predict), None)._one(0, FLIGHT, job, threading.Event())
    return row, job


@pytest.mark.parametrize('fallback,flow_fallback', [(False, False), (True, False), (False, True), (True, True)])
def test_primary_and_declared_fallback_results_are_accepted(fallback, flow_fallback):
    row, job = run_result(fallback=fallback, flow_fallback=flow_fallback)
    assert row['success'], row
    assert row['degraded'] == (fallback or flow_fallback)
    assert row['deployment_stage'] == 'experimental'
    assert row['flight_gate_passed'] is False
    assert row['flight_fallback_reason'] == ('PRIMARY_TIMEOUT' if fallback else '')
    exported = rows_for_job({**job, 'results': [row]})[0]
    assert exported['航班兜底原因'] == row['flight_fallback_reason']
    assert exported['部署阶段'] == 'experimental'


@pytest.mark.parametrize('tamper', [
    lambda r: setattr(r, 'source', pb.LLM),
    lambda r: setattr(r, 'model_version', 'unknown+gradient_boosting'),
    lambda r: setattr(r, 'prompt_version', 'unknown'),
    lambda r: setattr(r, 'flight_fallback_reason', ''),
    lambda r: setattr(r.airport_flow, 'model_version', 'unknown-flow'),
])
def test_degraded_flag_does_not_bypass_release_identity(tamper):
    row, _ = run_result(tamper=tamper)
    assert row['success'] is False
    assert row['error_code'] == 'DATA_LOSS'
