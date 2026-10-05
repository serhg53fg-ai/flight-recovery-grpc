from concurrent.futures import ThreadPoolExecutor

import pytest

from apps.flight.storage.jobs import JobStore


def test_job_identity_and_result_isolation_survive_reopen(tmp_path):
    store = JobStore(tmp_path)
    a = store.create(1, 'Asia/Shanghai')
    b = store.create(1, 'UTC')
    store.finish(a['job_id'], [{'success': True, 'flight_id': 'A', 'source': 'TEST'}])
    store.finish(b['job_id'], [{'success': False, 'flight_id': 'B', 'error_code': 'UNAVAILABLE'}])
    reopened = JobStore(tmp_path)
    assert reopened.get(a['job_id'])['status'] == 'SUCCEEDED'
    assert reopened.get(b['job_id'])['status'] == 'FAILED'
    assert reopened.get(a['job_id'])['results'][0]['flight_id'] == 'A'
    assert reopened.get(a['job_id'])['llm_success_count'] == 0
    assert reopened.get(b['job_id'])['input_timezone'] == 'UTC'


def test_partial_results_and_real_model_counts(tmp_path):
    store = JobStore(tmp_path)
    j = store.create(3, 'UTC')
    store.finish(j['job_id'], [
        {'success': True, 'source': 'LLM'}, {'success': True, 'source': 'TEST'},
        {'success': False, 'error_code': 'INVALID_ARGUMENT'},
    ])
    result = store.get(j['job_id'])
    assert (result['status'], result['success_count'], result['failed_count']) == ('PARTIAL', 2, 1)
    assert result['llm_success_count'] == 1


def test_path_traversal_and_unknown_job_are_rejected(tmp_path):
    store = JobStore(tmp_path)
    with pytest.raises(ValueError): store.get('../../other')
    with pytest.raises(KeyError): store.get('00000000-0000-0000-0000-000000000000')


def test_parallel_creates_do_not_collide(tmp_path):
    store = JobStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = list(pool.map(lambda _: store.create(1, 'UTC'), range(12)))
    assert len({j['job_id'] for j in jobs}) == 12
    assert len(store.list_jobs()) == 12


def test_finalization_is_idempotent_and_rejects_conflicting_results(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(1, 'UTC')
    result = store.finish(job['job_id'], [{'success': True, 'source': 'TEST'}])
    assert store.finish(job['job_id'], result['results']) == result
    with pytest.raises(ValueError, match='终态'):
        store.finish(job['job_id'], [{'success': False}])
    assert store.get(job['job_id']) == result


@pytest.mark.parametrize('total', [0, -1, True, 1.5])
def test_rejects_invalid_total(tmp_path, total):
    with pytest.raises(ValueError):
        JobStore(tmp_path).create(total, 'UTC')


@pytest.mark.parametrize('limit', [0, -1, True, 1001, '50'])
def test_rejects_invalid_list_limit(tmp_path, limit):
    with pytest.raises(ValueError):
        JobStore(tmp_path).list_jobs(limit)


def test_import_is_idempotent_and_checks_counts(tmp_path):
    source = JobStore(tmp_path / 'source')
    job = source.create(1, 'UTC')
    final = source.finish(job['job_id'], [{'success': False, 'error_code': 'UNAVAILABLE'}])
    target = JobStore(tmp_path / 'target')
    assert target.import_job(final) is True
    assert target.import_job(final) is False
    assert target.get(final['job_id']) == final
    with pytest.raises(ValueError):
        target.import_job({**final, 'status': 'SUCCEEDED'})
    with pytest.raises(ValueError):
        target.import_job({**final, 'success_count': 100})
