import json
import pytest

from training.cli import main
from tests.training.test_dataset import row, source


@pytest.mark.parametrize('bad_hash,missing', [(False, False), (True, False), (False, True)])
def test_evaluate_external_predictions(tmp_path, bad_hash, missing):
    from training.dataset import verify_dataset
    from training.baselines import ScheduleZeroBaseline
    source_path = source(tmp_path, [row(day) for day in range(1, 13)])
    dataset = tmp_path / 'dataset'
    assert main(['build', '--source', str(source_path), '--output', str(dataset),
                 '--min-test-rows', '1']) == 0
    manifest = verify_dataset(dataset)
    rows = [json.loads(line) for line in (dataset / 'test.jsonl').read_text().splitlines()]
    model = ScheduleZeroBaseline()
    model.fit([json.loads(line) for line in (dataset / 'train.jsonl').read_text().splitlines()])
    predictions = model.predict(rows)
    if missing:
        predictions.pop()
    path = tmp_path / 'predictions.jsonl'
    path.write_text(''.join(json.dumps(item) + '\n' for item in predictions))
    output = tmp_path / 'evaluation'
    result = main(['evaluate', '--dataset', str(dataset), '--predictions', str(path),
                   '--prediction-label-hash', 'wrong' if bad_hash else manifest['manifest_hash'],
                   '--output', str(output)])
    assert result == (2 if bad_hash or missing else 0)
    if result == 0:
        report = json.loads((output / 'metrics.json').read_text())
        assert report['evaluation']['sample_count'] == len(rows)
        assert report['dataset_manifest_hash'] == manifest['manifest_hash']
        assert (output / 'evaluation.md').is_file()
    else:
        assert not output.exists()


def test_build_and_train_baseline_cli_stays_in_runtime_root(tmp_path):
    source_path = source(tmp_path, [row(day, suffix) for day in range(1, 13) for suffix in range(2)])
    dataset = tmp_path / "runtime/dataset"
    models = tmp_path / "runtime/models"
    assert main(["build", "--source", str(source_path), "--output", str(dataset),
                 "--min-test-rows", "1", "--minimum-dates", "10"]) == 0
    assert main(["train-baseline", "--dataset", str(dataset), "--output", str(models)]) == 0
    metrics = json.loads((models / "metrics.json").read_text())
    assert set(metrics["models"]) == {"schedule_zero", "historical_median", "hist_gradient_boosting"}
    assert metrics["dataset_manifest_hash"]
    assert (models / "predictions/hist_gradient_boosting.jsonl").is_file()


def test_train_cli_rejects_tampered_dataset(tmp_path):
    source_path = source(tmp_path, [row(day) for day in range(1, 13)])
    dataset = tmp_path / "dataset"
    assert main(["build", "--source", str(source_path), "--output", str(dataset),
                 "--min-test-rows", "1"]) == 0
    (dataset / "test.jsonl").write_text((dataset / "train.jsonl").read_text(), encoding="utf-8")
    assert main(["train-baseline", "--dataset", str(dataset),
                 "--output", str(tmp_path / "models")]) == 2


def test_cli_failed_predictions_evaluate_and_gate(tmp_path):
    from training.dataset import verify_dataset
    source_path = source(tmp_path, [row(day) for day in range(1, 13)])
    dataset, models = tmp_path / 'dataset', tmp_path / 'models'
    assert main(['build', '--source', str(source_path), '--output', str(dataset), '--min-test-rows', '1']) == 0
    assert main(['train-baseline', '--dataset', str(dataset), '--output', str(models)]) == 0
    reference = models / 'predictions/schedule_zero.jsonl'
    items = [json.loads(line) for line in reference.read_text().splitlines()]
    failed = tmp_path / 'failed.jsonl'
    failed.write_text(json.dumps({'record_id': items.pop()['record_id'], 'error_type': 'ValueError'}) + '\n')
    predictions = tmp_path / 'candidate.jsonl'
    predictions.write_text(''.join(json.dumps(item) + '\n' for item in items))
    common = ['--dataset', str(dataset), '--predictions', str(predictions), '--failures', str(failed),
              '--prediction-label-hash', verify_dataset(dataset)['manifest_hash']]
    assert main(['evaluate', *common, '--output', str(tmp_path / 'eval')]) == 0
    assert main(['gate', *common, '--reference', str(reference), '--output', str(tmp_path / 'gate')]) == 3
    result = json.loads((tmp_path / 'gate/gate.json').read_text())
    assert result['gate']['passed'] is False
    reference_items = [json.loads(line) for line in reference.read_text().splitlines()]
    reference.write_text(''.join(json.dumps({**item, 'off_block_delay_min': 10000}) + '\n'
                                 for item in reference_items))
    # Arbitrary reference files must not replace the independently recomputed schedule.
    assert main(['gate', *common, '--reference', str(reference), '--output', str(tmp_path / 'bad-gate')]) == 2


def test_qwen_cli_rejects_invalid_dataset_before_model_load(tmp_path, monkeypatch):
    from training import batch
    monkeypatch.setattr(batch, 'OfflineQwen', lambda *args, **kwargs: pytest.fail('model loaded before validation'))
    assert main(['predict-qwen', '--dataset', str(tmp_path / 'missing'), '--model-path', str(tmp_path),
                 '--output', str(tmp_path / 'run')]) == 2


def test_qwen_cli_rejects_adapter_dataset_mismatch_before_model_load(tmp_path, monkeypatch):
    from training import batch
    from services.inference.adapters import qwen
    dataset = tmp_path / 'dataset'
    source_path = source(tmp_path, [row(day) for day in range(1, 13)])
    assert main(['build', '--source', str(source_path), '--output', str(dataset), '--min-test-rows', '1']) == 0
    monkeypatch.setattr(qwen, 'load_adapter_metadata', lambda *args: {'dataset_manifest_hash': 'wrong'})
    monkeypatch.setattr(batch, 'OfflineQwen', lambda *args, **kwargs: pytest.fail('model loaded before adapter validation'))
    assert main(['predict-qwen', '--dataset', str(dataset), '--model-path', str(tmp_path / 'model'),
                 '--adapter-path', str(tmp_path / 'adapter'), '--output', str(tmp_path / 'run')]) == 2


def test_qwen_cli_selects_legacy_prompt_for_legacy_dataset(tmp_path, monkeypatch):
    from training import batch
    from training.prompt import LEGACY_PROMPT_VERSION
    dataset = tmp_path / 'dataset'
    source_path = source(tmp_path, [row(day) for day in range(1, 13)])
    assert main(['build', '--source', str(source_path), '--output', str(dataset), '--min-test-rows', '1']) == 0
    class FakeQwen:
        def __init__(self, *args, **kwargs):
            assert kwargs['prompt_version'] == LEGACY_PROMPT_VERSION
            self.identity = {'kind': 'fake'}
        def __call__(self, features):
            return {'taxi_out_duration_min': 1, 'airborne_duration_min': 1,
                    'taxi_in_duration_min': 1}
    monkeypatch.setattr(batch, 'OfflineQwen', FakeQwen)
    assert main(['predict-qwen', '--dataset', str(dataset), '--model-path', str(tmp_path),
                 '--output', str(tmp_path / 'run')]) == 0
    assert json.loads((tmp_path / 'run/run.json').read_text())['prompt_version'] == LEGACY_PROMPT_VERSION
