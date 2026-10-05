"""Offline frozen-test predictions; no GPU imports until explicitly requested."""
from pathlib import Path
import json
import time

from .dataset import SCHEMA_VERSION, verify_dataset
from .prompt import (ALLOWED_FEATURES, LEGACY_PROMPT_VERSION, PROMPT_VERSION,
                     SUPPORTED_PROMPT_VERSIONS, parse_duration_output, render_duration_prompt)


ZGGG_FEATURES = frozenset({"airport_direction", "prediction_cutoff_time",
                           "departure_weather", "arrival_weather"})


def verify_prediction_dataset(dataset):
    dataset = Path(dataset).resolve()
    try:
        schema = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))['schema_version']
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError('invalid dataset manifest') from error
    if schema == 'zggg-flight-weather-v1':
        from .zggg_dataset import verify_zggg_dataset
        return verify_zggg_dataset(dataset), PROMPT_VERSION
    if schema == SCHEMA_VERSION:
        return verify_dataset(dataset), LEGACY_PROMPT_VERSION
    raise ValueError('unsupported prediction dataset schema')


def export_predictions(dataset, output, predict, identity):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    manifest, prompt_version = verify_prediction_dataset(dataset)
    if output.exists():
        raise ValueError('prediction output must be a new directory')
    rows = [json.loads(line) for line in (dataset / 'test.jsonl').read_text().splitlines()]
    output.mkdir(parents=True)
    successes = failures = 0
    with (output / 'predictions.jsonl').open('w') as predictions, (output / 'failures.jsonl').open('w') as errors:
        for row in rows:
            start = time.monotonic()
            allowed = ALLOWED_FEATURES | (ZGGG_FEATURES if prompt_version == PROMPT_VERSION else frozenset())
            features = {key: value for key, value in row['features'].items() if key in allowed}
            try:
                value = parse_duration_output(json.dumps(predict(features)))
            except Exception as error:
                failures += 1
                errors.write(json.dumps({'record_id': row['record_id'], 'error_type': type(error).__name__,
                                        'elapsed_ms': (time.monotonic() - start) * 1000}) + '\n')
                errors.flush()
            else:
                successes += 1
                predictions.write(json.dumps({'record_id': row['record_id'], **value,
                                              'elapsed_ms': (time.monotonic() - start) * 1000}) + '\n')
                predictions.flush()
    result = {'dataset_manifest_hash': manifest['manifest_hash'], 'prompt_version': prompt_version,
              'identity': identity, 'success_count': successes, 'failure_count': failures,
              'sample_count': len(rows)}
    (output / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


class OfflineQwen:
    """Both base and adapter use the training prompt; separate from online legacy mode."""
    def __init__(self, model_path, adapter_path=None, max_new_tokens=192,
                 prompt_version=PROMPT_VERSION):
        from .model_identity import model_fingerprint
        model_path = Path(model_path).resolve()
        fingerprint = model_fingerprint(model_path)
        metadata = None
        if adapter_path:
            from services.inference.adapters.qwen import load_adapter_metadata
            metadata = load_adapter_metadata(adapter_path, model_path.name, model_path)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise ValueError('offline Qwen requires CUDA')
        if not 1 <= max_new_tokens <= 4096:
            raise ValueError('invalid token budget')
        if prompt_version not in SUPPORTED_PROMPT_VERSIONS:
            raise ValueError('unsupported duration prompt version')
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                       trust_remote_code=False, torch_dtype='auto', device_map='auto').eval()
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(Path(adapter_path).resolve()),
                                                 local_files_only=True).eval()
        self.max_new_tokens = max_new_tokens
        self.prompt_version = prompt_version
        self.identity = {'kind': 'adapter' if metadata else 'base', 'base_model_version': model_path.name,
                         'base_model_fingerprint': fingerprint, 'adapter_metadata': metadata}

    def __call__(self, features):
        text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': render_duration_prompt(features, self.prompt_version)}],
                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        if inputs['input_ids'].shape[1] > 8192:
            raise ValueError('input token limit exceeded')
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                        pad_token_id=self.tokenizer.eos_token_id)
        return parse_duration_output(self.tokenizer.decode(output[0, inputs['input_ids'].shape[1]:],
                                                           skip_special_tokens=True))
