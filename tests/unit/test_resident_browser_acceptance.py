import json

from tests.unit.test_resident_ops import FakeClient


def test_browser_acceptance_records_real_business_and_ui_evidence(tmp_path):
    from scripts.resident_browser_acceptance import run_browser_acceptance

    def runner(base_url, job_id, screenshot):
        assert base_url == 'http://127.0.0.1:52100'
        assert job_id == 'job-a'
        screenshot.write_bytes(b'png')
        return {'title': '情景设定 - 航班预测系统', 'airport': 'ZGGG',
                'task_status': 'SUCCEEDED', 'recovery_status': 'SUCCEEDED',
                'situation': '运行态势：NORMAL'}

    output = tmp_path / 'browser'
    report = run_browser_acceptance(FakeClient(), 'http://127.0.0.1:52100', output,
                                    browser_runner=runner, poll_seconds=.01,
                                    timeout_seconds=1)

    assert report['status'] == 'PASS'
    assert report['demo']['prediction']['source'] == 'LLM'
    assert report['browser']['airport'] == 'ZGGG'
    assert json.loads((output / 'summary.json').read_text()) == report
    assert (output / 'scenario.png').read_bytes() == b'png'


def test_browser_acceptance_rejects_incomplete_ui_result(tmp_path):
    import pytest
    from scripts.resident_browser_acceptance import run_browser_acceptance

    def runner(base_url, job_id, screenshot):
        return {'airport': 'ZGGG', 'task_status': 'SUCCEEDED',
                'recovery_status': 'FAILED', 'situation': ''}

    with pytest.raises(RuntimeError, match='browser acceptance'):
        run_browser_acceptance(FakeClient(), 'http://127.0.0.1:52100', tmp_path / 'bad',
                               browser_runner=runner, poll_seconds=.01, timeout_seconds=1)


def test_browser_acceptance_rejects_existing_output(tmp_path):
    import pytest
    from scripts.resident_browser_acceptance import run_browser_acceptance

    output = tmp_path / 'existing'; output.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        run_browser_acceptance(FakeClient(), 'http://127.0.0.1:52100', output)
