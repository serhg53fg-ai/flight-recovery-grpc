import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_prometheus_rules_cover_availability_capacity_failure_and_queue():
    document = json.loads((ROOT / 'deploy/monitoring/prometheus-rules.yml').read_text())
    rules = document['groups'][0]['rules']
    by_name = {rule['alert']: rule for rule in rules}

    assert len(by_name) == len(rules)
    assert by_name['FlightMetricsTargetDown']['labels']['severity'] == 'critical'
    assert 'flight_gateway_serving' in by_name['FlightGatewayNotServing']['expr']
    assert 'flight_gateway_worker_healthy' in by_name['FlightWorkerUnhealthy']['expr']
    assert 'flight_gateway_worker_capacity' in by_name['FlightWorkerSaturated']['expr']
    assert 'increase(' in by_name['FlightRpcFailureIncrease']['expr']
    assert 'histogram_quantile(0.95' in by_name['FlightQueueP95High']['expr']
    assert '> 10' in by_name['FlightQueueP95High']['expr']


def test_prometheus_example_is_loopback_and_loads_rules():
    document = json.loads((ROOT / 'deploy/monitoring/prometheus.example.yml').read_text())
    assert document['rule_files'] == ['prometheus-rules.yml']
    assert document['scrape_configs'][0]['job_name'] == 'flight-resident'
    assert document['scrape_configs'][0]['static_configs'][0]['targets'] == ['127.0.0.1:52105']
