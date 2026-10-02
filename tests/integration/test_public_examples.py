"""Public synthetic inputs must exercise the actual gRPC business contract."""
import json
from pathlib import Path
from apps.flight.recovery.service import build_recovery
from tests.integration.test_prediction_flow import app, gateway_address

ROOT = Path(__file__).resolve().parents[2]


def test_public_batch_expected_times_and_recovery(app):
    flights = json.loads((ROOT / 'examples/synthetic-batch.json').read_text())
    expected = json.loads((ROOT / 'examples/expected-test-times.json').read_text())
    scenario = json.loads((ROOT / 'examples/recovery-scenario.json').read_text())
    with app.test_client() as client:
        response = client.post('/predict_flights_batch', json=flights)
        assert response.status_code == 200
        job = response.get_json()
        assert job['status'] == 'SUCCEEDED'
        assert job['success_count'] == len(flights) == 4
        assert [row['source'] for row in job['results']] == ['TEST'] * 4
        for row, reference in zip(job['results'], expected):
            assert row['flight_info']['航班号'] == reference['flight_number']
            assert row['prediction_data'] == reference['prediction_data']
        plan = build_recovery(job, scenario, 'synthetic-demo-v1')
        assert plan['validation_errors'] == []
        assert plan['status'] == 'SUCCEEDED'
        assert len(plan['assignments']) == 4

        assert plan['summary']['predicted_baseline_feasible'] is False
        assert plan['summary']['affected_flight_count'] == 2
        assert plan['summary']['added_delay_minutes'] == 6

        snapshot = plan['situation_snapshot']
        assert snapshot['risk_level'] == 'HIGH'
        slots = [c for c in snapshot['conflicts'] if c['type'].endswith('_SLOT_CAPACITY')]
        assert {c['type'] for c in slots} == {'DEPARTURE_SLOT_CAPACITY', 'ARRIVAL_SLOT_CAPACITY'}
        assert [c['excess'] for c in slots] == [1, 1]
