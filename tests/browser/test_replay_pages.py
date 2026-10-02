"""Browser contract for registered historical replay submissions."""
import os
import threading

import pytest
from werkzeug.serving import make_server

from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.test_durable_storage import repo
from tests.integration.test_prediction_flow import gateway_address
from tests.integration.test_replay_submission import replay_app, FLIGHT
from tests.integration.test_production_composite import composite_stack, _run
from tests.integration.production_fixtures import nginx_binary
from tests.integration.redis_fixtures import redis_socket, queue


pytestmark = pytest.mark.skipif(os.environ.get('FLIGHT_BROWSER_TESTS') != '1',
                                reason='opt-in browser acceptance')


def test_browser_replay_selection_and_refresh(replay_app, repo):
    from playwright.sync_api import expect, sync_playwright

    app, _ = replay_app
    server = make_server('127.0.0.1', 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=['--no-sandbox'])
            try:
                page = browser.new_page()
                page.goto(f'http://127.0.0.1:{server.server_port}/flight_input')
                page.wait_for_function('window.FlightAsync')
                expect(page.locator('#asyncReplayDataset')).to_have_value('may-v1')
                expect(page.locator('#asyncReplayNote')).to_contain_text('历史回放')
                job = page.evaluate('flights => FlightAsync.submitFlights(flights)', [FLIGHT])
                assert job['accepted']
                assert repo.get(job['job_id'])['submission_context']['replay_dataset_id'] == 'may-v1'
                page.reload()
                expect(page.locator('#asyncTaskStatus')).to_contain_text('QUEUED')
                expect(page.locator('#asyncReplayNote')).to_contain_text('快照')
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()


def test_browser_replay_to_download_with_composite(composite_stack, repo, queue):
    from playwright.sync_api import expect, sync_playwright

    stack, _, address = composite_stack
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=['--no-sandbox'])
        try:
            page = browser.new_page(accept_downloads=True)
            page.goto(stack.url + '/flight_input')
            page.wait_for_function('window.FlightAsync')
            expect(page.locator('#asyncReplayDataset')).to_have_value('may-v1')
            job = page.evaluate('flights => FlightAsync.submitFlights(flights)', [FLIGHT])
            assert job['status'] == 'QUEUED'
            _run(repo, queue, address)
            expect(page.locator('#asyncTaskStatus')).to_contain_text('SUCCEEDED', timeout=7000)
            expect(page.locator('#prediction-source')).to_contain_text('历史基线模型')
            expect(page.locator('#asyncResults')).to_contain_text('15 分钟累计')
            expect(page.locator('#asyncResults')).to_contain_text('预测实际离港时间')
            page.reload()
            expect(page.locator('#asyncTaskStatus')).to_contain_text('SUCCEEDED')
            with page.expect_download() as result:
                page.locator('#asyncDownload').click()
            assert result.value.suggested_filename.endswith('.xlsx')
        finally:
            browser.close()
