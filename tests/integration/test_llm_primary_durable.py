import pytest

from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.test_durable_storage import repo
from tests.unit.test_llm_primary_service import run_result
from tests.unit.test_context_snapshot import FLIGHT


@pytest.mark.parametrize('tamper', [False, True])
def test_durable_fallback_preserves_real_source_and_counts(repo, tamper):
    result, prototype = run_result()
    job = repo.submit([FLIGHT], 'Asia/Shanghai', 'fallback-case', prototype['model_version'],
                      'LLM', 'zggg-weather-duration-prompt-v2',
                      submission_context=prototype['submission_context'])
    claim = repo.claim(job['job_id'], job['flights'][0]['flight_id'], 'executor', 30)
    result.update({key: claim['base_result'][key] for key in ('job_id', 'flight_id', 'trace_id', 'index')})
    if tamper:
        result['model_version'] = 'unknown+gradient_boosting'
    assert repo.complete(claim, result)
    final = repo.get(job['job_id'])
    assert final['llm_success_count'] == 0
    assert final['status'] == ('FAILED' if tamper else 'SUCCEEDED')
    if not tamper:
        assert final['results'][0]['source'] == 'BASELINE'
        assert final['results'][0]['flight_fallback_reason'] == 'PRIMARY_TIMEOUT'
        assert final['sources'] == ['BASELINE']
