import pytest

from apps.flight.tasks.config import validate_durable_config
from tests.unit.test_recovery_engine import scenario
from tests.unit.test_recovery_service import prediction_job


def test_baseline_async_config_accepted():
    config = dict(DURABLE_ENABLED=True, STORAGE_BACKEND='mysql',
                  DURABLE_MODEL_VERSION='historical-flight-v1+gradient_boosting',
                  DURABLE_SOURCE='BASELINE', DURABLE_PROMPT_VERSION='historical-duration-v1',
                  DURABLE_TTL_SECONDS=300, DURABLE_MAX_ATTEMPTS=3,
                  SSE_POLL_SECONDS=.5, SSE_HEARTBEAT_SECONDS=15, SSE_MAX_SECONDS=30)
    validate_durable_config(config)


def test_baseline_recovery_preserves_provenance():
    from apps.flight.recovery.service import build_recovery

    job = prediction_job()
    job['results'][0].update(source='BASELINE', model_version='historical-flight-v1+gradient_boosting')
    plan = build_recovery(job, scenario(), 'weather-v1')
    assert plan['status'] == 'SUCCEEDED'
    assert plan['validation_errors'] == []
    assert plan['provenance'][0]['source'] == 'BASELINE'
    assert plan['provenance'][0]['model_version'] == 'historical-flight-v1+gradient_boosting'


@pytest.mark.parametrize('source', [None, '', 'baseline', 'UNKNOWN', 1])
def test_unknown_source_rejected(source):
    from apps.flight.domain.model_contract import validate_source

    with pytest.raises(ValueError):
        validate_source(source)


def test_cli_accepts_baseline_source_before_environment_validation(monkeypatch):
    from scripts.durable_prediction import main

    monkeypatch.delenv('FLIGHT_MYSQL_DATABASE', raising=False)
    assert main(['submit', '--input', '/missing', '--key', 'key', '--model-version', 'history-v1',
                 '--source', 'BASELINE', '--prompt-version', 'history-v1']) == 2
