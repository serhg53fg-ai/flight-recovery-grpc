import os
from pathlib import Path
import subprocess
import sys

import pytest

from apps.flight.clients.inference import InferenceClient, PredictionError


@pytest.mark.parametrize('module', ['scripts.durable_prediction', 'scripts.migrate_jobs'])
def test_operational_cli_help_without_pythonpath(module):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-m', module, '--help'], env=env,
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_invalid_remaining_budget_does_not_start_rpc(timeout):
    client = InferenceClient.__new__(InferenceClient)
    client.timeout = 20
    with pytest.raises(PredictionError) as error:
        client.predict(None, timeout=timeout)
    assert error.value.code == 'DEADLINE_EXCEEDED'
