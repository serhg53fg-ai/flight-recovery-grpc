from google.protobuf.empty_pb2 import Empty
from flight.v1 import prediction_pb2 as pb


def test_renders_gateway_snapshot_as_prometheus_metrics():
    from deploy.metrics.exporter import render_metrics

    snapshot = pb.ClusterStatus(serving=True)
    node = snapshot.nodes.add(id='worker-"a', healthy=True, circuit_state=pb.CLOSED,
                              inflight=2, capacity=4, selected_count=11,
                              success_count=9, failover_count=1, total_latency_ms=137)
    node.failure_counts['UNAVAILABLE'] = 2
    output = render_metrics(snapshot)
    assert 'flight_gateway_serving 1\n' in output
    assert 'flight_gateway_worker_healthy{worker="worker-\\"a"} 1\n' in output
    assert 'flight_gateway_worker_inflight{worker="worker-\\"a"} 2\n' in output
    assert 'flight_gateway_worker_success_total{worker="worker-\\"a"} 9\n' in output
    assert 'flight_gateway_worker_failure_total{worker="worker-\\"a",code="UNAVAILABLE"} 2\n' in output
    assert 'flight_gateway_worker_latency_milliseconds_total{worker="worker-\\"a"} 137\n' in output


def test_scrape_queries_gateway_each_time_and_fails_closed_on_rpc_error():
    import grpc
    import pytest
    from deploy.metrics.exporter import scrape_metrics

    class Stub:
        def __init__(self): self.calls = 0
        def GetClusterStatus(self, request, timeout):
            assert isinstance(request, Empty) and timeout == 0.5
            self.calls += 1
            if self.calls == 2: raise grpc.RpcError('unavailable')
            return pb.ClusterStatus(serving=True)

    stub = Stub()
    assert 'flight_gateway_serving 1' in scrape_metrics(stub, timeout=0.5)
    with pytest.raises(grpc.RpcError): scrape_metrics(stub, timeout=0.5)
    assert stub.calls == 2


def test_scrape_includes_persisted_stage_histograms():
    from deploy.metrics.exporter import scrape_metrics

    class Stub:
        def GetClusterStatus(self, request, timeout):
            return pb.ClusterStatus(serving=True)

    output = scrape_metrics(Stub(), stage_loader=lambda: [
        {'stage': 'rpc', 'outcome': 'success', 'seconds': .12,
         'job_id': 'must-not-be-a-label'},
    ])
    assert 'flight_gateway_serving 1' in output
    assert 'flight_stage_duration_seconds_count{stage="rpc",outcome="success"} 1' in output
    assert 'must-not-be-a-label' not in output
