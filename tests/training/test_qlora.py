import json
from pathlib import Path

import pytest

from training.qlora import TrainingConfig, build_training_spec, preflight, write_adapter_metadata


class FakeCuda:
    def __init__(self, available=True): self.available = available
    def is_available(self): return self.available
    def device_count(self): return 1
    def is_bf16_supported(self): return True


class FakeTorch:
    def __init__(self, available=True): self.cuda = FakeCuda(available)


def setup_files(tmp_path: Path):
    model = tmp_path / "model"; model.mkdir()
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (model / name).write_text("{}", encoding="utf-8")
    data = tmp_path / "sft"; data.mkdir()
    hashes = {}
    for split in ("train", "validation", "test"):
        text = json.dumps({"record_id": split, "prompt": "p", "response": "{}"}) + "\n"
        (data / f"{split}.jsonl").write_text(text, encoding="utf-8")
        import hashlib; hashes[split] = hashlib.sha256(text.encode()).hexdigest()
    (data / "manifest.json").write_text(json.dumps({"dataset_manifest_hash": "dataset-hash", "prompt_version": "flight-duration-prompt-v1", "splits": hashes}))
    return model, data


def config(tmp_path, **changes):
    model, data = setup_files(tmp_path)
    values = dict(model_path=model, data_dir=data, output_dir=tmp_path / "output",
                  resume_from_checkpoint=None, epochs=2.0, learning_rate=2e-4,
                  batch_size=1, gradient_accumulation_steps=8, seed=42,
                  final_fit=False)
    values.update(changes)
    return TrainingConfig(**values)


def test_preflight_validates_local_model_data_hashes_and_cuda(tmp_path):
    environment = preflight(config(tmp_path), FakeTorch())
    assert environment.dataset_manifest_hash == "dataset-hash"
    assert environment.prompt_version == "flight-duration-prompt-v1"


def test_preflight_rejects_missing_cuda_and_tampered_split(tmp_path):
    values = config(tmp_path)
    with pytest.raises(ValueError, match="CUDA"): preflight(values, FakeTorch(False))
    (values.data_dir / "train.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"): preflight(values, FakeTorch())


def test_rejects_output_overlap_and_invalid_hyperparameters(tmp_path):
    values = config(tmp_path)
    with pytest.raises(ValueError): preflight(TrainingConfig(**{**values.__dict__, "output_dir": values.data_dir}), FakeTorch())
    with pytest.raises(ValueError): preflight(TrainingConfig(**{**values.__dict__, "batch_size": 0}), FakeTorch())


def test_builds_deterministic_four_bit_lora_spec(tmp_path):
    spec = build_training_spec(config(tmp_path))
    assert spec["quantization"] == {"load_in_4bit": True, "quant_type": "nf4", "double_quant": True}
    assert spec["lora"]["target_modules"] == "all-linear"
    assert spec["training"]["seed"] == 42
    assert "test" not in spec["training"]["data_splits"]


def test_final_fit_uses_train_and_validation_without_test(tmp_path):
    spec = build_training_spec(config(tmp_path, final_fit=True))
    assert spec["training"]["fit_scope"] == "train+validation"
    assert spec["training"]["data_splits"] == ("train", "validation")


def test_writes_explicit_adapter_metadata_without_paths(tmp_path):
    values = config(tmp_path)
    environment = preflight(values, FakeTorch())
    path = write_adapter_metadata(values, environment, base_model_version="Qwen-test")
    metadata = json.loads(path.read_text())
    assert metadata["output_mode"] == "duration_components"
    assert metadata["base_model_version"] == "Qwen-test"
    assert metadata["adapter_version"] == "output"
    assert metadata["fit_scope"] == "train"
    assert len(metadata["base_model_fingerprint"]) == 64
    assert str(values.model_path) not in path.read_text()


def test_final_fit_metadata_records_validation_use(tmp_path):
    values = config(tmp_path, final_fit=True)
    environment = preflight(values, FakeTorch())
    metadata = json.loads(write_adapter_metadata(
        values, environment, base_model_version="Qwen-test"
    ).read_text())
    assert metadata["fit_scope"] == "train+validation"


def test_v2_metadata_records_zggg_weather_contract(tmp_path):
    values = config(tmp_path)
    manifest = json.loads((values.data_dir / "manifest.json").read_text())
    manifest.update({"prompt_version": "zggg-weather-duration-prompt-v2",
                     "airport_scope": "ZGGG",
                     "weather_feature_version": "aviation-weather-v1",
                     "temporal_fit_filter": {
                         "train": {"rows": 1, "eligible_rows": 1, "excluded_rows": 0,
                                   "reasons": {}, "cutoff": "2025-05-09T00:00:00Z"},
                         "validation": {"rows": 1, "eligible_rows": 1, "excluded_rows": 0,
                                        "reasons": {}, "cutoff": "2025-05-11T00:00:00Z"}}})
    (values.data_dir / "manifest.json").write_text(json.dumps(manifest))
    environment = preflight(values, FakeTorch())
    metadata = json.loads(write_adapter_metadata(
        values, environment, base_model_version="Qwen-test").read_text())
    assert metadata["schema_version"] == "flight-adapter-v2"
    assert metadata["airport_scope"] == "ZGGG"
    assert metadata["weather_feature_version"] == "aviation-weather-v1"


def test_v2_preflight_rejects_unsafe_or_inconsistent_temporal_filter(tmp_path):
    values = config(tmp_path)
    path = values.data_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update({"prompt_version": "zggg-weather-duration-prompt-v2",
                     "airport_scope": "ZGGG", "weather_feature_version": "aviation-weather-v1"})
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="temporal"):
        preflight(values, FakeTorch())
    manifest["temporal_fit_filter"] = {
        "train": {"rows": 2, "eligible_rows": 2, "excluded_rows": 0,
                  "reasons": {}, "cutoff": "2025-05-09T00:00:00Z"},
        "validation": {"rows": 1, "eligible_rows": 1, "excluded_rows": 0,
                       "reasons": {}, "cutoff": "2025-05-11T00:00:00Z"}}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="temporal"):
        preflight(values, FakeTorch())
