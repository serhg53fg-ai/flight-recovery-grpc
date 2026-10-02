"""AutoDL-only QLoRA training entrypoint with no-GPU testable preflight."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path

from .prompt import PROMPT_VERSION, SUPPORTED_PROMPT_VERSIONS
from .model_identity import model_fingerprint


@dataclass(frozen=True)
class TrainingConfig:
    model_path: Path
    data_dir: Path
    output_dir: Path
    resume_from_checkpoint: Path | None = None
    epochs: float = 2.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    seed: int = 42
    final_fit: bool = False


@dataclass(frozen=True)
class TrainingEnvironment:
    dataset_manifest_hash: str
    prompt_version: str
    airport_scope: str | None = None
    weather_feature_version: str | None = None


def _hash(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try: path.relative_to(parent); return True
    except ValueError: return False


def preflight(config: TrainingConfig, torch_module=None) -> TrainingEnvironment:
    model, data, output = (Path(config.model_path).expanduser().resolve(), Path(config.data_dir).expanduser().resolve(), Path(config.output_dir).expanduser().resolve())
    if not model.is_dir() or not (model / "config.json").is_file() or not any(model.glob("*.safetensors")):
        raise ValueError("local model directory is incomplete")
    if not data.is_dir() or _inside(output, data) or _inside(output, model) or output in {data, model}:
        raise ValueError("output directory must be separate from model and data")
    if not config.epochs > 0 or not 0 < config.learning_rate <= 0.01 or type(config.batch_size) is not int or config.batch_size < 1 or type(config.gradient_accumulation_steps) is not int or config.gradient_accumulation_steps < 1:
        raise ValueError("invalid training hyperparameters")
    try: manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error: raise ValueError("invalid SFT manifest") from error
    prompt_version = manifest.get("prompt_version")
    if prompt_version not in SUPPORTED_PROMPT_VERSIONS or not manifest.get("dataset_manifest_hash"):
        raise ValueError("SFT prompt or dataset manifest is incompatible")
    if prompt_version == PROMPT_VERSION and (
            manifest.get("airport_scope") != "ZGGG" or
            not manifest.get("weather_feature_version")):
        raise ValueError("ZGGG SFT weather contract is incomplete")
    for split in ("train", "validation", "test"):
        path = data / f"{split}.jsonl"
        if not path.is_file() or manifest.get("splits", {}).get(split) != _hash(path):
            raise ValueError(f"{split} hash mismatch")
    if prompt_version == PROMPT_VERSION:
        audits = manifest.get("temporal_fit_filter")
        if not isinstance(audits, dict):
            raise ValueError("ZGGG temporal fit filter is required")
        cutoffs = []
        for split in ("train", "validation"):
            audit = audits.get(split)
            count = sum(1 for _ in (data / f"{split}.jsonl").open(encoding="utf-8"))
            if not isinstance(audit, dict) or any(
                type(audit.get(key)) is not int or audit[key] < 0
                for key in ("rows", "eligible_rows", "excluded_rows")
            ) or audit["rows"] != audit["eligible_rows"] + audit["excluded_rows"] or audit["eligible_rows"] != count:
                raise ValueError("ZGGG temporal fit filter row counts are inconsistent")
            try:
                cutoff = datetime.fromisoformat(audit["cutoff"].replace("Z", "+00:00"))
                if cutoff.tzinfo is None or cutoff.utcoffset() is None:
                    raise ValueError("naive cutoff")
            except (KeyError, AttributeError, ValueError, TypeError) as error:
                raise ValueError("ZGGG temporal fit filter cutoff is invalid") from error
            cutoffs.append(cutoff)
        if cutoffs[0] >= cutoffs[1]:
            raise ValueError("ZGGG temporal fit filter cutoffs are out of order")
    if config.resume_from_checkpoint is not None:
        checkpoint = Path(config.resume_from_checkpoint).expanduser().resolve()
        if not checkpoint.is_dir() or not _inside(checkpoint, output):
            raise ValueError("checkpoint must exist inside output directory")
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() < 1:
        raise ValueError("CUDA is unavailable")
    return TrainingEnvironment(manifest["dataset_manifest_hash"], prompt_version,
                               manifest.get("airport_scope"),
                               manifest.get("weather_feature_version"))


def build_training_spec(config: TrainingConfig):
    return {
        "quantization": {"load_in_4bit": True, "quant_type": "nf4", "double_quant": True},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
        "training": {"epochs": config.epochs, "learning_rate": config.learning_rate,
                     "batch_size": config.batch_size, "gradient_accumulation_steps": config.gradient_accumulation_steps,
                     "seed": config.seed, "data_splits": ("train", "validation"),
                     "fit_scope": "train+validation" if config.final_fit else "train"},
    }


def write_adapter_metadata(config, environment, *, base_model_version):
    output = Path(config.output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    is_v2 = environment.prompt_version == PROMPT_VERSION
    metadata = {"schema_version": "flight-adapter-v2" if is_v2 else "flight-adapter-v1",
                "output_mode": "duration_components",
                "prompt_version": environment.prompt_version, "dataset_manifest_hash": environment.dataset_manifest_hash,
                "base_model_version": str(base_model_version)[:128],
                "base_model_fingerprint": model_fingerprint(config.model_path),
                "adapter_version": output.name[:128],
                "fit_scope": "train+validation" if config.final_fit else "train"}
    if is_v2:
        metadata.update({"airport_scope": environment.airport_scope,
                         "weather_feature_version": environment.weather_feature_version})
    path = output / "adapter_metadata.json"; temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)
    return path


def run_training(config: TrainingConfig):
    import torch
    environment = preflight(config, torch)
    from datasets import concatenate_datasets, load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    spec = build_training_spec(config)
    tokenizer = AutoTokenizer.from_pretrained(str(config.model_path), local_files_only=True, trust_remote_code=False)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    dataset = load_dataset("json", data_files={name: str(Path(config.data_dir) / f"{name}.jsonl") for name in ("train", "validation")})
    from .chat import tokenize_example
    dataset = dataset.map(lambda row: tokenize_example(row, tokenizer),
                          remove_columns=dataset['train'].column_names)
    model = AutoModelForCausalLM.from_pretrained(str(config.model_path), local_files_only=True,
                                                trust_remote_code=False, quantization_config=quant, device_map="auto")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, target_modules="all-linear", task_type="CAUSAL_LM")
    arguments = SFTConfig(output_dir=str(config.output_dir), num_train_epochs=config.epochs,
                          learning_rate=config.learning_rate, per_device_train_batch_size=config.batch_size,
                          gradient_accumulation_steps=config.gradient_accumulation_steps, seed=config.seed,
                          gradient_checkpointing=True, report_to="none",
                          completion_only_loss=True, max_length=None, packing=False,
                          dataset_kwargs={"skip_prepare_dataset": True})
    train_dataset = (concatenate_datasets([dataset["train"], dataset["validation"]])
                     if config.final_fit else dataset["train"])
    eval_dataset = None if config.final_fit else dataset["validation"]
    trainer = SFTTrainer(model=model, args=arguments, train_dataset=train_dataset, eval_dataset=eval_dataset,
                         peft_config=lora, processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=str(config.resume_from_checkpoint) if config.resume_from_checkpoint else None)
    trainer.save_model(str(config.output_dir)); tokenizer.save_pretrained(str(config.output_dir))
    return write_adapter_metadata(config, environment, base_model_version=Path(config.model_path).name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a flight-duration QLoRA adapter on AutoDL")
    parser.add_argument("--model-path", type=Path, required=True); parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--epochs", type=float, default=2.0); parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-fit", action="store_true")
    return TrainingConfig(**vars(parser.parse_args(argv)))


if __name__ == "__main__": run_training(parse_args())
