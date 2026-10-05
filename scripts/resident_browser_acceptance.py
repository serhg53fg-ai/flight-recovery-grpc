"""Run a real Resident business demonstration through Chromium pages."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from scripts.benchmark_workflow import HttpClient
from scripts.resident_ops import run_demo


def run_browser_ui(base_url: str, job_id: str, screenshot: Path) -> dict:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=['--no-sandbox'])
        try:
            page = browser.new_page(viewport={'width': 1440, 'height': 1000},
                                    timezone_id='Asia/Shanghai')
            response = page.goto(f'{base_url}/flight_input?job_id={job_id}')
            if response is None or response.status != 200:
                raise RuntimeError('flight input page failed')
            page.locator('#asyncTaskStatus').wait_for(state='visible', timeout=10000)
            page.wait_for_function(
                "document.querySelector('#asyncTaskStatus').textContent.includes('SUCCEEDED')",
                timeout=15000)
            task_status = page.locator('#asyncTaskStatus').inner_text()
            response = page.goto(f'{base_url}/scenario?job_id={job_id}')
            if response is None or response.status != 200:
                raise RuntimeError('scenario page failed')
            page.wait_for_function('window.FlightRecovery', timeout=10000)
            page.fill('#recoveryStart', '2025-05-02T08:00:00+08:00')
            page.fill('#recoveryEnd', '2025-05-02T16:00:00+08:00')
            page.locator('#recoverySubmit').click()
            page.wait_for_function(
                "document.querySelector('#recoveryStatus').textContent.includes('SUCCEEDED')",
                timeout=15000)
            page.screenshot(path=str(screenshot), full_page=True)
            return {
                'title': page.title(),
                'airport': page.locator('#recoveryAirport').input_value(),
                'task_status': task_status,
                'recovery_status': page.locator('#recoveryStatus').inner_text(),
                'situation': page.locator('#situationSummary').inner_text(),
            }
        finally:
            browser.close()


def run_browser_acceptance(client, base_url: str, output: Path, *,
                           browser_runner=run_browser_ui, poll_seconds=.25,
                           timeout_seconds=180) -> dict:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('browser acceptance output already exists')
    if not output.parent.is_dir():
        raise ValueError('browser acceptance parent is missing')
    output.mkdir()
    demo = run_demo(client, output / 'demo.json', poll_seconds=poll_seconds,
                    timeout_seconds=timeout_seconds)
    browser = browser_runner(base_url.rstrip('/'), demo['job_id'], output / 'scenario.png')
    valid = (browser.get('airport') == 'ZGGG' and
             'SUCCEEDED' in browser.get('task_status', '') and
             'SUCCEEDED' in browser.get('recovery_status', '') and
             '运行态势' in browser.get('situation', '') and
             (output / 'scenario.png').is_file())
    if not valid:
        raise RuntimeError('browser acceptance result is incomplete')
    report = {'status': 'PASS', 'created_at': datetime.now(timezone.utc).isoformat(),
              'demo': demo, 'browser': browser, 'screenshot': 'scenario.png'}
    with (output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Resident Chromium business acceptance')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url')
    parser.add_argument('--browsers-path')
    parser.add_argument('--library-path')
    parser.add_argument('--fontconfig-file')
    parser.add_argument('--cache-path')
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text())
        base_url = args.base_url or f"http://127.0.0.1:{config.get('http_port', 8080)}"
        for name, value in (
            ('PLAYWRIGHT_BROWSERS_PATH', args.browsers_path),
            ('LD_LIBRARY_PATH', args.library_path),
            ('FONTCONFIG_FILE', args.fontconfig_file),
            ('XDG_CACHE_HOME', args.cache_path),
        ):
            if value:
                os.environ[name] = value
        report = run_browser_acceptance(HttpClient(base_url), base_url, args.output)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError,
            json.JSONDecodeError) as exc:
        print(f'browser acceptance failed: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
