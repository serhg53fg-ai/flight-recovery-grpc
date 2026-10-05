"""Send real predictions through a live AutoDL Gateway and verify release identity."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import grpc
from google.protobuf.empty_pb2 import Empty

from deploy.distributed.release import load_release, release_identity
from deploy.distributed.worker_pool import probe_request
from flight.v1 import prediction_pb2 as pb, prediction_pb2_grpc as rpc


@dataclass(frozen=True)
class Observation:
    worker_id: str
    source: str
    model_version: str
    flow_model_version: str
    flight_degraded: bool
    flow_degraded: bool


def evaluate(responses, *, expected_workers: set[str], expected_source: str,
             expected_model_version: str, expected_flow_model_version: str) -> dict:
    failures = []
    seen = {item.worker_id for item in responses}
    missing = sorted(expected_workers - seen)
    if missing:
        failures.append("workers not selected: " + ", ".join(missing))
    checks = (
        ("source", expected_source),
        ("model_version", expected_model_version),
        ("flow_model_version", expected_flow_model_version),
        ("flight_degraded", False),
        ("flow_degraded", False),
    )
    for index, item in enumerate(responses):
        if item.worker_id not in expected_workers:
            failures.append(f"response {index} unexpected worker_id: {item.worker_id}")
        for name, expected in checks:
            actual = getattr(item, name)
            if actual != expected:
                failures.append(f"response {index} {name}: {actual} != {expected}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "request_count": len(responses),
        "worker_ids_seen": sorted(seen),
        "failures": failures,
    }


def _observe(stub, timeout: float) -> Observation:
    result = stub.Predict(probe_request(), timeout=timeout)
    return Observation(
        worker_id=result.worker_id,
        source=pb.PredictionSource.Name(result.source),
        model_version=result.model_version,
        flow_model_version=(result.airport_flow.model_version
                            if result.HasField("airport_flow") else ""),
        flight_degraded=result.flight_degraded,
        flow_degraded=result.flow_degraded,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify live multi-Worker routing and one immutable release identity")
    parser.add_argument("--gateway-address", required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-worker", action="append", required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.requests <= 1000 or not 1 <= args.parallelism <= 64:
        parser.error("requests and parallelism are outside the accepted range")
    expected_workers = set(args.expected_worker)
    if len(expected_workers) != len(args.expected_worker):
        parser.error("expected workers must be unique")
    try:
        identity = release_identity(load_release(args.release_manifest))
        channel = grpc.insecure_channel(
            args.gateway_address, options=[("grpc.enable_retries", 0)])
        try:
            grpc.channel_ready_future(channel).result(timeout=args.timeout_seconds)
            prediction = rpc.PredictionGatewayStub(channel)
            with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
                futures = [executor.submit(_observe, prediction, args.timeout_seconds)
                           for _ in range(args.requests)]
                observations = [future.result() for future in futures]
            status = rpc.GatewayAdminStub(channel).GetClusterStatus(
                Empty(), timeout=args.timeout_seconds)
        finally:
            channel.close()
        report = evaluate(
            observations, expected_workers=expected_workers,
            expected_source=identity["source"],
            expected_model_version=identity["model_version"],
            expected_flow_model_version=identity["flow_model_version"])
        unhealthy = sorted(node.id for node in status.nodes
                           if node.id in expected_workers and not node.healthy)
        if not status.serving or unhealthy:
            report["failures"].append(
                "gateway/admin unhealthy: " + ", ".join(unhealthy))
            report["status"] = "FAIL"
        report.update(
            created_at=datetime.now(timezone.utc).isoformat(),
            gateway_address=args.gateway_address,
            release_identity=identity,
            observations=[asdict(item) for item in observations],
            cluster_nodes=[{"id": node.id, "healthy": node.healthy,
                            "circuit_state": pb.CircuitState.Name(node.circuit_state),
                            "selected_count": node.selected_count,
                            "success_count": node.success_count,
                            "failover_count": node.failover_count}
                           for node in status.nodes],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["status"] == "PASS" else 2
    except (OSError, ValueError, grpc.RpcError, grpc.FutureTimeoutError) as error:
        print(f"AutoDL cluster acceptance failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
