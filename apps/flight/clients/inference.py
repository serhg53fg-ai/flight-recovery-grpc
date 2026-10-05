"""Reusable gRPC client with explicit error and cancellation semantics."""
import threading
import math

import grpc
from flight.v1 import prediction_pb2_grpc as rpc
from apps.flight.domain.prediction import validate_response


class PredictionError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class InferenceClient:
    def __init__(self, address, timeout=20):
        if not address or not 0 < float(timeout) <= 300:
            raise ValueError('无效网关地址或 RPC 超时')
        self.timeout = float(timeout)
        self.channel = grpc.insecure_channel(address, options=[
            ('grpc.enable_retries', 0), ('grpc.max_receive_message_length', 1024*1024),
            ('grpc.max_send_message_length', 1024*1024)])
        self.stub = rpc.PredictionGatewayStub(self.channel)

    def predict(self, request, cancel=None, timeout=None):
        budget = self.timeout if timeout is None else float(timeout)
        if not math.isfinite(budget) or budget <= 0:
            raise PredictionError('DEADLINE_EXCEEDED', '任务剩余预算已耗尽')
        budget = min(self.timeout, budget)
        call = self.stub.Predict.future(request, timeout=budget)
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    call.cancel()
                    raise PredictionError('CANCELLED', '任务已取消')
                try:
                    result = call.result(timeout=0.05)
                    break
                except grpc.FutureTimeoutError:
                    continue
            validate_response(result, request.trace_id)
            return result
        except grpc.RpcError as exc:
            raise PredictionError(exc.code().name, exc.details() or '远程推理失败') from exc
        except grpc.FutureCancelledError as exc:
            raise PredictionError('CANCELLED', '任务已取消') from exc
        except ValueError as exc:
            raise PredictionError('DATA_LOSS', str(exc)) from exc

    def close(self):
        self.channel.close()
