from tests.browser.test_async_pages import pytestmark
from tests.integration.production_fixtures import (
    nginx_binary, mysql_server, mysql_settings, mysql_config, redis_socket,
    queue, repo, gateway_address, production_stack, stop_process)
from tests.integration.test_production_http import submit, run_prediction


def test_browser_proxy_recovers_instance_and_generates_versioned_recovery(production_stack, repo, queue, gateway_address):
    from playwright.sync_api import sync_playwright, expect
    stack = production_stack
    job = submit(stack)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        try:
            page = browser.new_page()
            with page.expect_response(lambda response: '/events?after=' in response.url) as reply:
                page.goto(stack.url + '/flight_input?job_id=' + job['job_id'])
            origin = reply.value.header_value('x-flight-instance')
            expect(page.locator('#asyncTaskStatus')).to_contain_text('QUEUED')
            stopped = 0 if origin == 'web-a' else 1
            stop_process(stack.web[stopped])
            run_prediction(repo, queue, gateway_address)
            expect(page.locator('#asyncTaskStatus')).to_contain_text('SUCCEEDED', timeout=7000)
            stack.start_web(stopped)
            page.goto(stack.url + '/scenario?job_id=' + job['job_id'])
            page.wait_for_function('window.FlightRecovery')
            expect(page.locator('.container > .content')).not_to_be_visible()
            page.fill('#recoveryStart', '2025-05-02T08:00:00+08:00')
            page.fill('#recoveryEnd', '2025-05-02T16:00:00+08:00')
            page.locator('#recoverySubmit').click()
            expect(page.locator('#recoveryStatus')).to_contain_text('SUCCEEDED')
            rid = page.evaluate("sessionStorage.getItem('flight.selectedRecoveryId')")
            assert rid
            page.reload()
            expect(page.locator('#recoveryStatus')).to_contain_text(rid)
        finally:
            browser.close()
