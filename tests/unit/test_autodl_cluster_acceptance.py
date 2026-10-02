from types import SimpleNamespace

import pytest

from scripts.autodl_cluster_acceptance import evaluate


def response(worker_id, *, source="LLM", model="flight-v1+flow-v1",
             flow="flow-v1", flight_degraded=False, flow_degraded=False):
    return SimpleNamespace(
        worker_id=worker_id, source=source, model_version=model,
        flow_model_version=flow, flight_degraded=flight_degraded,
        flow_degraded=flow_degraded)


def test_evaluate_accepts_all_expected_workers_with_one_release_identity():
    report = evaluate(
        [response("worker-a"), response("worker-b")],
        expected_workers={"worker-a", "worker-b"}, expected_source="LLM",
        expected_model_version="flight-v1+flow-v1",
        expected_flow_model_version="flow-v1")
    assert report == {
        "status": "PASS", "request_count": 2,
        "worker_ids_seen": ["worker-a", "worker-b"], "failures": []}


@pytest.mark.parametrize("item,reason", [
    (response("worker-a", model="wrong"), "model_version"),
    (response("worker-a", flow="wrong"), "flow_model_version"),
    (response("worker-a", source="BASELINE"), "source"),
    (response("worker-a", flight_degraded=True), "flight_degraded"),
    (response("worker-a", flow_degraded=True), "flow_degraded"),
])
def test_evaluate_rejects_identity_or_degradation(item, reason):
    report = evaluate(
        [item], expected_workers={"worker-a"}, expected_source="LLM",
        expected_model_version="flight-v1+flow-v1",
        expected_flow_model_version="flow-v1")
    assert report["status"] == "FAIL"
    assert reason in report["failures"][0]


def test_evaluate_rejects_missing_worker_distribution():
    report = evaluate(
        [response("worker-a"), response("worker-a")],
        expected_workers={"worker-a", "worker-b"}, expected_source="LLM",
        expected_model_version="flight-v1+flow-v1",
        expected_flow_model_version="flow-v1")
    assert report["status"] == "FAIL"
    assert report["failures"] == ["workers not selected: worker-b"]
