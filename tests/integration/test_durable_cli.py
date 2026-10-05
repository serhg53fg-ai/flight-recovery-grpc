import json

import pytest

from tests.integration.mysql_fixtures import mysql_server, mysql_settings, mysql_config
from tests.integration.redis_fixtures import redis_socket
from tests.integration.test_prediction_flow import gateway_address, FLIGHT


def test_cli_initialize_submit_execute_and_status(mysql_config, redis_socket, gateway_address, tmp_path, monkeypatch, capsys):
    from scripts.durable_prediction import main
    for key, value in {'DATABASE': mysql_config.database, 'USER': mysql_config.user,
                       'UNIX_SOCKET': mysql_config.unix_socket}.items():
        monkeypatch.setenv('FLIGHT_MYSQL_' + key, value)
    monkeypatch.setenv('FLIGHT_REDIS_UNIX_SOCKET', redis_socket)
    monkeypatch.setenv('FLIGHT_REDIS_STREAM', 'cli:' + mysql_config.database)
    monkeypatch.setenv('FLIGHT_GATEWAY_ADDRESS', gateway_address)
    assert main(['initialize']) == 0
    assert json.loads(capsys.readouterr().out)['initialized'] is True
    source = tmp_path / 'flights.json'
    source.write_text(json.dumps([FLIGHT]), encoding='utf-8')
    command = ['submit', '--input', str(source), '--key', 'cli', '--model-version', 'test-v1',
               '--source', 'TEST', '--prompt-version', 'test-v1']
    assert main(command) == 0
    job_id = json.loads(capsys.readouterr().out)['job_id']
    assert main(['publisher', '--once']) == 0
    capsys.readouterr()
    assert main(['executor', '--owner', 'cli-test', '--once']) == 0
    capsys.readouterr()
    assert main(['status', job_id]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status['status'] == 'SUCCEEDED' and status['llm_success_count'] == 0
    assert main(['status', job_id, '--include-results']) == 0
    detailed = json.loads(capsys.readouterr().out)
    assert detailed['results'][0]['source'] == 'TEST'
    assert detailed['results'][0]['prediction_data']['实际离港时间'] == '2025-05-02T09:40:00+08:00'
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)['job_id'] == job_id


def test_cli_invalid_config_does_not_print_secrets(monkeypatch, capsys):
    from scripts.durable_prediction import main
    monkeypatch.setenv('FLIGHT_MYSQL_PASSWORD', 'sensitive-password')
    monkeypatch.delenv('FLIGHT_MYSQL_DATABASE', raising=False)
    assert main(['initialize']) == 2
    output = capsys.readouterr().out
    assert 'sensitive-password' not in output and '失败' in output
