import os
import threading

import pytest
from werkzeug.serving import make_server

from apps.flight.clients.inference import InferenceClient
from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.redis_fixtures import redis_socket, queue
from tests.integration.test_durable_storage import repo
from tests.integration.test_async_api import async_app
from tests.integration.test_durable_queue import publisher
from tests.integration.test_durable_execution import executor
from tests.integration.test_prediction_flow import gateway_address, FLIGHT, excel_file

pytestmark = pytest.mark.skipif(os.environ.get('FLIGHT_BROWSER_TESTS') != '1', reason='opt-in browser acceptance')


@pytest.fixture
def async_browser(async_app):
    from playwright.sync_api import sync_playwright
    server = make_server('127.0.0.1', 0, async_app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
            context = browser.new_context(timezone_id='Asia/Shanghai')
            try:
                yield context, f'http://127.0.0.1:{server.server_port}'
            finally:
                context.close()
                browser.close()
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()


def open_page(browser):
    context, base = browser
    page = context.new_page()
    page.goto(base + '/flight_input')
    page.wait_for_function('window.FlightAsync', timeout=5000)
    return page


def run_prediction(repo, queue, address):
    publisher(repo, queue).run_once()
    client = InferenceClient(address, 1)
    try:
        executor(repo, queue, client).run_once()
    finally:
        client.close()


def test_json_page_submits_once_refreshes_and_recovers_results(async_browser, repo, queue, gateway_address):
    import json
    from playwright.sync_api import expect
    page = open_page(async_browser)
    requests = []
    page.on('request', lambda r: requests.append(r.url) if r.method == 'POST' else None)
    page.evaluate("switchMode('json')")
    page.fill('#jsonInput', json.dumps([FLIGHT]))
    job = page.evaluate('processJsonData()')
    assert job['accepted'] and job['status'] == 'QUEUED'
    expect(page.locator('#asyncTaskStatus')).to_contain_text('QUEUED')
    page.reload()
    page.wait_for_function('window.FlightAsync')
    expect(page.locator('#asyncTaskStatus')).to_contain_text('QUEUED')
    assert len([url for url in requests if url.endswith('/prediction-jobs')]) == 1
    run_prediction(repo, queue, gateway_address)
    expect(page.locator('#asyncTaskStatus')).to_contain_text('SUCCEEDED')
    expect(page.locator('#asyncResults article')).to_have_count(1)
    expect(page.locator('#asyncResults')).to_contain_text('2025-05-02T09:40:00+08:00')
    expect(page.locator('#prediction-source')).to_contain_text('TEST')
    page.locator('#asyncDownload').click()


def test_page_cancel_and_error_values_are_safe_text(async_browser, repo):
    from playwright.sync_api import expect
    page = open_page(async_browser)
    bad = {**FLIGHT, '机尾号': '<img src=x onerror=window.__asyncXss=1>'}
    job = page.evaluate('flights => FlightAsync.submitFlights(flights)', [FLIGHT, bad])
    page.locator('#asyncCancel').click()
    expect(page.locator('#asyncTaskStatus')).to_contain_text('CANCELLED')
    expect(page.locator('#asyncResults article')).to_have_count(2)
    assert page.evaluate('window.__asyncXss') is None
    assert repo.get(job['job_id'])['cancelled_count'] == 1


def test_excel_page_uses_async_upload(async_browser, repo, queue, gateway_address):
    from playwright.sync_api import expect
    page = open_page(async_browser)
    buffer, name = excel_file([FLIGHT])
    page.locator('input[type=file]').set_input_files({'name': name, 'mimeType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'buffer': buffer.getvalue()})
    job = page.evaluate('uploadAndPredict()')
    assert job['accepted'] and job['status'] == 'QUEUED'
    run_prediction(repo, queue, gateway_address)
    expect(page.locator('#asyncTaskStatus')).to_contain_text('SUCCEEDED')
    expect(page.locator('#asyncResults article')).to_have_count(1)


def test_retry_after_accepted_response_loss_reuses_idempotency_key(async_browser, repo):
    page = open_page(async_browser)
    captured = []
    def lose_response(route):
        captured.append(route.request.headers['idempotency-key'])
        assert route.fetch().status == 202
        route.abort('failed')
    page.route('**/api/v1/prediction-jobs', lose_response)
    with pytest.raises(Exception, match='fetch|Failed|Network'):
        page.evaluate('flights => FlightAsync.submitFlights(flights)', [FLIGHT])
    page.unroute('**/api/v1/prediction-jobs', lose_response)
    job = page.evaluate('flights => FlightAsync.submitFlights(flights)', [FLIGHT])
    assert len(repo.list_jobs()) == 1
    assert repo.list_jobs()[0]['job_id'] == job['job_id']


def test_async_selection_is_independent_between_tabs(async_browser):
    from playwright.sync_api import expect
    a = open_page(async_browser)
    b = open_page(async_browser)
    one = a.evaluate('flights => FlightAsync.submitFlights(flights)', [FLIGHT])
    two = b.evaluate('flights => FlightAsync.submitFlights(flights)', [{**FLIGHT, '航班号': 'CZ2222'}])
    assert one['job_id'] != two['job_id']
    assert a.evaluate('FlightJobs.selected()') == one['job_id']
    assert b.evaluate('FlightJobs.selected()') == two['job_id']


def test_accepted_job_recovers_when_first_status_read_fails(async_browser, repo):
    from playwright.sync_api import expect
    page = open_page(async_browser)
    calls = []
    def fail_once(route):
        calls.append(route.request.url)
        if len(calls) == 1:
            route.abort('failed')
        else:
            route.continue_()
    page.route('**/api/v1/prediction-jobs/*', fail_once)
    job = page.evaluate('flights => FlightAsync.submitFlights(flights)', [FLIGHT])
    repo.cancel(job['job_id'])
    expect(page.locator('#asyncTaskStatus')).to_contain_text('CANCELLED', timeout=7000)


def test_insecure_origin_reports_supported_access_method(async_browser):
    page = open_page(async_browser)
    page.evaluate("Object.defineProperty(crypto, 'subtle', {value: undefined}); Object.defineProperty(crypto, 'randomUUID', {value: undefined})")
    with pytest.raises(Exception, match='localhost.*HTTPS'):
        page.evaluate('flights => FlightAsync.submitFlights(flights)', [FLIGHT])
