"""Bounded gRPC inference worker; model failures are never simulated successes."""
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from flight.v1 import prediction_pb2 as pb, prediction_pb2_grpc as rpc
from apps.flight.domain.prediction import validate_request, validate_response
from services.inference.errors import InvalidInferenceRequest


class InferenceServicer(rpc.InferenceWorkerServicer):
    def __init__(self, backend, worker_id, max_inflight=1):
        if not worker_id.strip() or len(worker_id) > 128 or not 1 <= max_inflight <= 256:
            raise ValueError('无效 worker_id/max_inflight')
        self.backend, self.worker_id = backend, worker_id
        self.permits = threading.BoundedSemaphore(max_inflight)

    def Predict(self, request, context):
        try:
            validate_request(request)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        if not context.is_active():
            context.abort(grpc.StatusCode.CANCELLED, 'request cancelled')
        if not self.permits.acquire(blocking=False):
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, 'worker at capacity')
        start = time.monotonic()
        status = 'OK'
        try:
            remaining = context.time_remaining()
            budget = min(300, remaining) if remaining is not None else 20
            if budget <= 0:
                raise TimeoutError('request expired')
            cancelled = lambda: not context.is_active()
            if hasattr(self.backend, 'predict_response'):
                response = self.backend.predict_response(request, cancelled, budget)
                response.trace_id = request.trace_id
                response.worker_id = self.worker_id
                response.inference_ms = int((time.monotonic()-start)*1000)
            else:
                prediction = self.backend.predict(request.flight, cancelled, budget)
                response = pb.PredictResponse(trace_id=request.trace_id, prediction=prediction,
                    source=self.backend.source, model_version=self.backend.model_version,
                    worker_id=self.worker_id, inference_ms=int((time.monotonic()-start)*1000))
            if not context.is_active():
                raise TimeoutError('request no longer active')
            validate_response(response, request.trace_id)
            return response
        except TimeoutError as exc:
            status = 'DEADLINE_EXCEEDED' if context.is_active() else 'CANCELLED'
            context.abort(getattr(grpc.StatusCode, status), str(exc))
        except InvalidInferenceRequest as exc:
            status = 'INVALID_ARGUMENT'
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except ValueError as exc:
            status = 'DATA_LOSS'
            context.abort(grpc.StatusCode.DATA_LOSS, str(exc))
        except Exception:
            status = 'UNAVAILABLE'
            context.abort(grpc.StatusCode.UNAVAILABLE, 'model backend unavailable')
        finally:
            self.permits.release()
            print(json.dumps(dict(event='worker.predict', trace_id=request.trace_id,
                                  worker_id=self.worker_id, status=status,
                                  inference_ms=int((time.monotonic()-start)*1000))), flush=True)


def serve(listen, backend, worker_id='worker-1', max_inflight=1):
    executor = ThreadPoolExecutor(max_workers=max_inflight+4)
    server = grpc.server(executor, options=[('grpc.max_receive_message_length', 1024*1024),
                                           ('grpc.max_send_message_length', 1024*1024)])
    rpc.add_InferenceWorkerServicer_to_server(InferenceServicer(backend, worker_id, max_inflight), server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    if not server.add_insecure_port(listen):
        executor.shutdown(wait=False)
        raise RuntimeError(f'无法监听 {listen}')
    server.start()
    health_servicer.set('', health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set('flight.v1.InferenceWorker', health_pb2.HealthCheckResponse.SERVING)
    server._flight_health_servicer = health_servicer
    return server, executor


def mark_not_serving(server):
    health_servicer = server._flight_health_servicer
    health_servicer.set('', health_pb2.HealthCheckResponse.NOT_SERVING)
    health_servicer.set('flight.v1.InferenceWorker', health_pb2.HealthCheckResponse.NOT_SERVING)


def build_composite_backend(args):
    import pickle
    from training.flow_model import load_flow_artifact
    from services.inference.adapters.composite import CompositeBackend, HistoricalFlightBackend
    if not args.flow_model_path or not args.flow_manifest_hash or not args.historical_model_path:
        raise ValueError('composite artifacts are required')
    flow = load_flow_artifact(args.flow_model_path, args.flow_manifest_hash)
    with open(args.historical_model_path, 'rb') as stream:
        historical_model = pickle.load(stream)
    historical = HistoricalFlightBackend(historical_model)
    if args.composite_flight_mode == 'historical':
        backend = CompositeBackend(historical, flow)
    else:
        from services.inference.adapters.qwen import QwenBackend
        qwen = QwenBackend(args.model_path, args.max_new_tokens, args.max_seconds,
                           args.adapter_path, output_mode=args.output_mode)
        backend = CompositeBackend(qwen, flow, historical)
    expected_source = getattr(args, 'expected_source', None)
    expected_version = getattr(args, 'expected_model_version', None)
    if (expected_source and pb.PredictionSource.Name(backend.source) != expected_source or
            expected_version and backend.model_version != expected_version):
        raise ValueError('release identity does not match loaded worker models')
    return backend


def main():
    parser = argparse.ArgumentParser(description='Flight inference gRPC worker')
    parser.add_argument('--listen', default='127.0.0.1:50052')
    parser.add_argument('--backend', choices=['test', 'qwen', 'composite'], default='qwen')
    parser.add_argument('--model-path')
    parser.add_argument('--adapter-path')
    parser.add_argument('--flow-model-path')
    parser.add_argument('--flow-manifest-hash')
    parser.add_argument('--historical-model-path')
    parser.add_argument('--composite-flight-mode', choices=['qwen', 'historical'], default='qwen')
    parser.add_argument('--expected-model-version')
    parser.add_argument('--expected-source', choices=['BASELINE', 'LLM'])
    parser.add_argument('--output-mode', choices=['absolute_times', 'duration_components'])
    parser.add_argument('--worker-id', default='worker-1')
    parser.add_argument('--max-inflight', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=192)
    parser.add_argument('--max-seconds', type=float, default=20)
    args = parser.parse_args()
    if not 1 <= args.max_inflight <= 256:
        parser.error('max-inflight must be 1..256')
    try:
        if args.backend == 'test':
            from services.inference.adapters.test_backend import TestBackend
            backend = TestBackend()
        elif args.backend == 'qwen':
            from services.inference.adapters.qwen import QwenBackend
            backend = QwenBackend(args.model_path, args.max_new_tokens, args.max_seconds, args.adapter_path,
                                  output_mode=args.output_mode)
        else:
            backend = build_composite_backend(args)
        server, executor = serve(args.listen, backend, args.worker_id, args.max_inflight)
    except Exception as exc:
        parser.exit(2, f'Worker startup failed: {exc}\n')
    stopped = threading.Event()
    def stop(*_): stopped.set()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(json.dumps(dict(event='worker.ready', listen=args.listen,
                          source=pb.PredictionSource.Name(backend.source), worker_id=args.worker_id)), flush=True)
    try:
        stopped.wait()
    finally:
        mark_not_serving(server)
        server.stop(1).wait()
        executor.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    main()
