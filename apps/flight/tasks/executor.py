"""One leased flight at a time through the existing gRPC prediction path."""
import threading
import time
from datetime import datetime, timezone
import json

from apps.flight.clients.inference import PredictionError
from apps.flight.services.prediction import PredictionService
from apps.flight.storage.repository import StorageError
from .documents import bounded_integer, label


class BudgetClient:
    def __init__(self, client, deadline):
        self.client, self.deadline = client, deadline
        self.last_rpc_seconds = 0.0

    def predict(self, request, cancel):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise PredictionError('DEADLINE_EXCEEDED', '任务剩余预算已耗尽')
        started = time.monotonic()
        try:
            return self.client.predict(request, cancel, timeout=min(self.client.timeout, remaining))
        finally:
            self.last_rpc_seconds = time.monotonic() - started


class DurableExecutor:
    def __init__(self, repository, queue, client, owner, lease_seconds=30):
        label(owner, '执行者')
        bounded_integer(lease_seconds, 1, 300, '租约')
        self.repository, self.queue, self.client = repository, queue, client
        self.owner, self.lease_seconds = owner, lease_seconds

    def run_once(self):
        delivery = self.queue.receive(self.owner, idle_ms=self.lease_seconds * 1000)
        if delivery is None:
            return False
        message_id, payload = delivery
        started = time.monotonic()
        claim = self.repository.claim(payload['job_id'], payload['flight_id'], self.owner, self.lease_seconds)
        if claim is None:
            if self.repository.is_terminal(payload['job_id'], payload['flight_id']):
                self.queue.ack(message_id)
            return True
        deadline = started + claim['remaining_seconds']
        cancel, stopped, lost = threading.Event(), threading.Event(), threading.Event()
        def renew():
            while not stopped.wait(self.lease_seconds / 3):
                try:
                    valid = self.repository.heartbeat(claim, self.lease_seconds)
                except StorageError:
                    valid = False
                if not valid:
                    lost.set()
                    cancel.set()
                    return
        heartbeat = threading.Thread(target=renew, name='prediction-lease', daemon=True)
        heartbeat.start()
        try:
            def observe(stage, seconds, outcome):
                try:
                    self.repository.record_stage(claim, stage, seconds, outcome)
                except StorageError:
                    pass  # A metric write cannot turn a valid prediction into failure.

            submitted = datetime.fromisoformat(claim['job']['created_at'])
            queue_seconds = max(0.0, (datetime.now(timezone.utc) - submitted).total_seconds())
            observe('queue', queue_seconds, 'success')
            budget_client = BudgetClient(self.client, deadline)
            service = PredictionService(budget_client, self.repository)
            result = service._one(claim['base_result']['index'] - 1, None, claim['job'], cancel, prepared=claim)
            observe('rpc', budget_client.last_rpc_seconds,
                    'success' if result['success'] else 'failure')
            if not lost.is_set():
                persist_started = time.monotonic()
                committed = self.repository.complete(claim, result)
                observe('persist', time.monotonic() - persist_started,
                        'success' if committed else 'failure')
                observe('total', max(0.0, (datetime.now(timezone.utc) - submitted).total_seconds()),
                        'success' if committed and result['success'] else 'failure')
                print(json.dumps({'event': 'flight.attempt', 'job_id': claim['job_id'],
                                  'flight_id': claim['flight_id'], 'attempt': claim['fence'],
                                  'trace_id': claim['trace_id'],
                                  'outcome': 'success' if committed and result['success'] else 'failure',
                                  'error_code': result.get('error_code')}), flush=True)
            if self.repository.is_terminal(payload['job_id'], payload['flight_id']):
                self.queue.ack(message_id)
            return True
        finally:
            stopped.set()
            heartbeat.join()
