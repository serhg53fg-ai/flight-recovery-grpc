"""One prediction path for JSON, spreadsheet and streaming inputs."""
from __future__ import annotations

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from uuid import uuid4

from google.protobuf.json_format import MessageToDict, ParseDict
from flight.v1 import prediction_pb2 as pb
from apps.flight.clients.inference import PredictionError
from apps.flight.domain.prediction import (flow_prediction_to_dict, normalize_flight,
                                           normalize_zggg_context, prediction_to_dict,
                                           timezone_for)


def serializable(value):
    if isinstance(value, dict): return {str(k): serializable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [serializable(v) for v in value]
    if isinstance(value, datetime): return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    if value is None or isinstance(value, (str,int,float,bool)): return value
    return str(value)


def validate_serving_identity(result, job):
    identity = (job.get('submission_context') or {}).get('release_identity')
    if not isinstance(result, dict):
        result = {'source': pb.PredictionSource.Name(result.source),
                  'model_version': result.model_version, 'prompt_version': result.prompt_version,
                  'flight_degraded': result.flight_degraded, 'flow_degraded': result.flow_degraded,
                  'flight_fallback_reason': result.flight_fallback_reason,
                  'flow_fallback_reason': result.flow_fallback_reason,
                  'airport_flow': {'model_version': result.airport_flow.model_version}
                                 if result.HasField('airport_flow') else None}
    source = result.get('source')
    if not identity:
        if job.get('model_version') and (result.get('model_version') != job['model_version'] or
                                         source != job.get('source')):
            raise PredictionError('DATA_LOSS', 'Worker 模型版本或来源与任务不匹配')
        return {}
    if job.get('model_version') and (job['model_version'] != identity['model_version'] or
                                     job.get('source') != identity['source']):
        raise PredictionError('DATA_LOSS', '任务与冻结发布身份不匹配')
    selected = identity
    if result.get('flight_degraded'):
        selected = identity.get('fallback_identity')
        if not selected or result.get('flight_fallback_reason') not in (
                'PRIMARY_TIMEOUT', 'PRIMARY_INVALID_OUTPUT', 'PRIMARY_UNAVAILABLE'):
            raise PredictionError('DATA_LOSS', '未声明的航班兜底')
    elif result.get('flight_fallback_reason'):
        raise PredictionError('DATA_LOSS', '航班兜底标记不一致')
    flow = identity['flow_model_version']
    if result.get('flow_degraded'):
        flow = 'planned-flow-fallback-v1'
        if result.get('flow_fallback_reason') != 'FLOW_MODEL_UNAVAILABLE':
            raise PredictionError('DATA_LOSS', '流量兜底原因不一致')
    elif result.get('flow_fallback_reason'):
        raise PredictionError('DATA_LOSS', '流量兜底标记不一致')
    if not isinstance(result.get('airport_flow'), dict) or result['airport_flow'].get('model_version') != flow:
        raise PredictionError('DATA_LOSS', 'Worker 流量模型发布身份不匹配')
    if (source != selected['source'] or
            result.get('model_version') != selected['flight_model_version'] + '+' + flow or
            ('prompt_version' in selected and result.get('prompt_version') != selected['prompt_version'])):
        raise PredictionError('DATA_LOSS', 'Worker 航班模型发布身份不匹配')
    return identity


class PredictionService:
    def __init__(self, client, store, max_workers=1, max_batch=1000):
        if not 1 <= max_workers <= 32 or not 1 <= max_batch <= 10000:
            raise ValueError('无效批量参数')
        self.client, self.store = client, store
        self.max_workers, self.max_batch = max_workers, max_batch

    def start(self, flights, input_timezone):
        if not isinstance(flights, list) or not 1 <= len(flights) <= self.max_batch:
            raise ValueError(f'航班数组长度必须为 1—{self.max_batch}')
        timezone_for(input_timezone)
        return self.store.create(len(flights), input_timezone)

    def _one(self, index, data, job, cancel, prepared=None):
        row = dict(prepared['base_result']) if prepared else dict(index=index+1, flight_id=str(uuid4()), trace_id=str(uuid4()),
                   job_id=job['job_id'], flight_info=serializable(data) if isinstance(data,dict) else {},
                   success=False, degraded=False)
        try:
            if cancel.is_set(): raise PredictionError('CANCELLED', '任务已取消')
            try:
                f = ParseDict(prepared['normalized_input'], pb.FlightFeatures()) if prepared else normalize_flight(data, job['input_timezone'])
                context = (ParseDict(prepared['normalized_context'], pb.ZgggPredictionContext())
                           if prepared and prepared.get('normalized_context') else
                           normalize_zggg_context(data, f, job['input_timezone']))
            except ValueError as exc:
                raise PredictionError('INVALID_ARGUMENT', str(exc)) from exc
            f.flight_id = row['flight_id']
            row['normalized_input'] = MessageToDict(f, preserving_proto_field_name=True)
            row['normalized_context'] = (MessageToDict(context, preserving_proto_field_name=True)
                                         if context is not None else None)
            tz = timezone_for(job['input_timezone'])
            row['flight_info'] = {
                '航班号':f.flight_number, '机尾号':f.tail_number, '机型':f.aircraft_type,
                '性质':f.flight_nature,
                '计划起飞站四字码':f.departure_airport, '计划到达站四字码':f.arrival_airport,
                '计划离港时间':f.planned_off_block.ToDatetime(tzinfo=tz).isoformat(),
                '计划到港时间':f.planned_on_block.ToDatetime(tzinfo=tz).isoformat(),
            }
            request = pb.PredictRequest(trace_id=row['trace_id'], flight=f)
            if context is not None:
                request.zggg_context.CopyFrom(context)
            result = self.client.predict(request, cancel)
            identity = validate_serving_identity(result, job)
            times = prediction_to_dict(result.prediction, job['input_timezone'])
            row.update(success=True, prediction=json.dumps(times, ensure_ascii=False),
                       prediction_data=times, source=pb.PredictionSource.Name(result.source),
                       worker_id=result.worker_id, model_version=result.model_version,
                       inference_ms=result.inference_ms, gateway_ms=result.gateway_ms,
                       airport_flow=flow_prediction_to_dict(
                           result, (context.prediction_cutoff if context is not None
                                    else f.planned_off_block).ToDatetime(tzinfo=tz).isoformat()),
                       weather_stale=result.weather_stale,
                       flight_degraded=result.flight_degraded,
                       flow_degraded=result.flow_degraded,
                       degraded=result.flight_degraded or result.flow_degraded,
                       flight_fallback_reason=result.flight_fallback_reason,
                       flow_fallback_reason=result.flow_fallback_reason,
                       deployment_stage=identity.get('deployment_stage', 'validated' if identity else 'unspecified'),
                       flight_gate_passed=identity.get('flight_gate_passed', True if identity else None),
                       data_version=result.data_version,
                       prompt_version=result.prompt_version)
        except PredictionError as exc:
            row.update(error_code=exc.code, error=str(exc))
        except Exception:
            row.update(error_code='INTERNAL', error='预测处理异常')
        return row

    def iter_results(self, flights, job):
        cancel = threading.Event()
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        futures = {executor.submit(self._one, i, f, job, cancel):i for i,f in enumerate(flights)}
        results = [None]*len(flights)
        try:
            for future in as_completed(futures):
                result = future.result()
                results[futures[future]] = result
                yield result
        finally:
            cancel.set()
            for future,index in futures.items():
                if results[index] is None:
                    if future.done() and not future.cancelled():
                        results[index] = future.result()
                    else:
                        future.cancel()
                        results[index] = dict(index=index+1,flight_id=str(uuid4()),trace_id=str(uuid4()),
                            job_id=job['job_id'],flight_info=serializable(flights[index]),success=False,
                            error_code='CANCELLED',error='流式连接已关闭或任务已取消',degraded=False)
            executor.shutdown(wait=False, cancel_futures=True)
            self.store.finish(job['job_id'], results)

    def predict_many(self, flights, input_timezone='Asia/Shanghai'):
        job = self.start(flights, input_timezone)
        for _ in self.iter_results(flights, job): pass
        return self.store.get(job['job_id'])
