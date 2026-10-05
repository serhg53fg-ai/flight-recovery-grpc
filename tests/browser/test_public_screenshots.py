"""Opt-in screenshots of real pages using only public synthetic TEST inputs."""
import json
import os
import threading
from pathlib import Path
import pytest
from werkzeug.serving import make_server
from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.test_recovery_jobs import recovery_app, recovery_repo
from tests.integration.test_prediction_flow import gateway_address

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get('FLIGHT_PUBLIC_SCREENSHOTS') != '1', reason='opt-in synthetic screenshot generation')


def test_capture_synthetic_business_pages(recovery_app):
    from playwright.sync_api import sync_playwright, expect
    flights = json.loads((ROOT / 'examples/synthetic-batch.json').read_text())
    server = make_server('127.0.0.1', 0, recovery_app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = ROOT / 'images'
    output.mkdir(exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
            try:
                page = browser.new_page(viewport={'width': 1440, 'height': 1000}, timezone_id='Asia/Shanghai')
                base = f'http://127.0.0.1:{server.server_port}'
                def capture(name):
                    page.evaluate("""() => {
                        const label = document.createElement('div');
                        label.textContent = 'PUBLIC DEMO · 合成航班 / TEST · 非真实预测效果';
                        label.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:99999;padding:9px;text-align:center;background:#fff4c2;color:#392c00;font:700 16px sans-serif;';
                        label.id='public-demo-watermark'; document.body.appendChild(label);
                    }""")
                    page.screenshot(path=str(output / name))
                    page.locator('#public-demo-watermark').evaluate('(node) => node.remove()')
                page.goto(base + '/flight_input')
                page.evaluate("switchMode('json')")
                page.locator('#jsonInput').fill(json.dumps(flights, ensure_ascii=False, indent=2))
                capture('synthetic-input.png')
                page.locator('button[onclick="processJsonData()"]').click()
                page.wait_for_function('FlightJobs.selected()', timeout=15000)
                job_id = page.evaluate('FlightJobs.selected()')
                def complete():
                    job = page.request.get(base + '/api/jobs/' + job_id).json()
                    return job['status'] == 'SUCCEEDED'
                import time
                deadline = time.monotonic() + 15
                while not complete():
                    if time.monotonic() >= deadline: raise TimeoutError('synthetic prediction timed out')
                    time.sleep(.1)
                expect(page.locator('#prediction-source')).to_contain_text('TEST')
                expect(page.locator('#model-notice')).to_contain_text('TEST')
                page.evaluate("FlightJobs.record({status: 'QUEUED'})")
                expect(page.locator('#model-notice')).to_contain_text('等待')
                page.evaluate('(job) => FlightJobs.record(job)', page.request.get(base + '/api/jobs/' + job_id).json())
                expect(page.locator('#model-notice')).to_contain_text('TEST')
                page.locator('.prediction-table').first.evaluate('(node) => window.scrollTo(0, node.getBoundingClientRect().top + window.scrollY - 90)')
                capture('synthetic-prediction.png')
                page.goto(base + '/timeline?job_id=' + job_id)
                page.locator('.timeline-section').wait_for()
                page.locator('#departureTimeline .hour-block').nth(9).wait_for()
                page.locator('#departureTimeline .hour-block').nth(9).scroll_into_view_if_needed()
                capture('synthetic-timeline.png')
                page.goto(base + '/scenario?job_id=' + job_id)
                page.wait_for_function('window.FlightRecovery', timeout=10000)
                page.fill('#recoveryStart', '2030-01-15T08:00:00+08:00')
                page.fill('#recoveryEnd', '2030-01-15T20:00:00+08:00')
                page.fill('#recoveryDepartureCapacity', '1')
                page.fill('#recoveryArrivalCapacity', '1')
                page.locator('#recoverySubmit').click()
                expect(page.locator('#recoveryStatus')).to_contain_text('SUCCEEDED', timeout=15000)
                expect(page.locator('#situationSummary')).to_contain_text('调度前预测态势：HIGH')
                expect(page.locator('#recoverySummary')).to_contain_text('方案校验：通过')
                expect(page.locator('#recoveryResults tbody tr')).to_have_count(4)
                page.locator('#situationSummary').scroll_into_view_if_needed()
                capture('synthetic-recovery.png')
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()
