import threading

import pytest
from werkzeug.serving import make_server

from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.test_recovery_jobs import recovery_repo, recovery_app
from tests.integration.test_prediction_flow import gateway_address, FLIGHT
from tests.browser.test_async_pages import pytestmark


@pytest.fixture
def recovery_browser(recovery_app):
    from playwright.sync_api import sync_playwright
    server = make_server('127.0.0.1', 0, recovery_app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
            context = browser.new_context()
            try: yield context, f'http://127.0.0.1:{server.server_port}'
            finally: context.close(); browser.close()
    finally:
        server.shutdown(); thread.join(5); server.server_close()


def test_scenario_page_binds_prediction_and_restores_recovery(recovery_browser, recovery_app):
    from playwright.sync_api import expect
    job = recovery_app.test_client().post('/predict_flights_batch', json={'flights': [FLIGHT]}).json
    context, base = recovery_browser
    page = context.new_page()
    page.goto(base + '/scenario?job_id=' + job['job_id'])
    page.wait_for_function('window.FlightRecovery', timeout=5000)
    expect(page.locator('#recoveryAirport')).to_have_value('ZGGG')
    page.fill('#recoveryStart', '2025-05-02T08:00:00+08:00')
    page.fill('#recoveryEnd', '2025-05-02T16:00:00+08:00')
    page.locator('#recoverySubmit').click()
    expect(page.locator('#recoveryStatus')).to_contain_text('SUCCEEDED')
    expect(page.locator('#recoveryResults tbody tr')).to_have_count(1)
    expect(page.locator('#recoverySource')).to_contain_text(job['job_id'])
    expect(page.locator('#recoverySource')).to_contain_text('TEST')
    expect(page.locator('#recoverySummary')).to_contain_text('已安排 1')
    expect(page.locator('#recoverySummary')).to_contain_text('方案校验：通过')
    expect(page.locator('#recoveryMetrics')).to_contain_text('最大延误')
    expect(page.locator('#recoveryMetrics')).to_contain_text('完成率 100.0%')
    expect(page.locator('#situationSummary')).to_contain_text('调度前预测态势：NORMAL')
    expect(page.locator('#situationSummary')).to_contain_text('预测成功 1/1')
    expect(page.locator('#situationRisks')).to_contain_text('调度前未识别高风险冲突')
    expect(page.locator('#recoveryResults thead')).to_contain_text('预测离港')
    expect(page.locator('#recoveryResults thead')).to_contain_text('调整后离港')
    rid = page.evaluate("sessionStorage.getItem('flight.selectedRecoveryId')")
    assert rid
    page.reload()
    expect(page.locator('#recoveryStatus')).to_contain_text(rid)
    page.locator('#recoveryDownload').click()


def test_recovery_errors_are_safe_and_do_not_fall_back(recovery_browser):
    from playwright.sync_api import expect
    context, base = recovery_browser
    page = context.new_page()
    page.goto(base + '/scenario')
    page.wait_for_function('window.FlightRecovery', timeout=5000)
    page.locator('#recoverySubmit').click()
    expect(page.locator('#recoveryStatus')).to_contain_text('请选择预测任务')
    page.evaluate("FlightRecovery.render({recovery_id:'demo', source_job_id:'<img src=x onerror=window.__recoveryXss=1>', status:'INVALID', assignments:[], unassigned:[], excluded:[], provenance:[], validation_errors:['<script>bad</script>']})")
    assert page.evaluate('window.__recoveryXss') is None
    expect(page.locator('#recoverySource')).to_contain_text('<img')
