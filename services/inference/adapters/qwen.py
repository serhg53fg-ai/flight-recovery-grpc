"""Read-only local Qwen adapter. Heavy libraries are loaded only by the worker."""
from __future__ import annotations

import json
import time
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from flight.v1 import prediction_pb2 as pb
from apps.flight.domain.prediction import TIME_FIELDS, TIME_LABELS, parse_time, validate_response
from training.prompt import (LEGACY_PROMPT_VERSION, PROMPT_VERSION,
                             parse_duration_output, reconstruct_times, render_duration_prompt)
from training.model_identity import model_fingerprint


def load_adapter_metadata(adapter_path, base_model_version, base_model_path):
    path = Path(adapter_path).resolve()
    if not path.is_dir() or not (path / 'adapter_config.json').is_file():
        raise ValueError('adapter 目录无效')
    try:
        metadata = json.loads((path / 'adapter_metadata.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('adapter 元数据无效') from exc
    required = {'schema_version', 'output_mode', 'prompt_version', 'dataset_manifest_hash',
                'base_model_version', 'base_model_fingerprint', 'adapter_version'}
    allowed_extra = {'fit_scope', 'airport_scope', 'weather_feature_version'}
    schema = metadata.get('schema_version')
    prompt_version = metadata.get('prompt_version')
    version_valid = ((schema == 'flight-adapter-v1' and prompt_version == LEGACY_PROMPT_VERSION) or
                     (schema == 'flight-adapter-v2' and prompt_version == PROMPT_VERSION and
                      metadata.get('airport_scope') == 'ZGGG' and
                      bool(metadata.get('weather_feature_version'))))
    if not required.issubset(metadata) or set(metadata) - required - allowed_extra or \
       metadata.get('fit_scope', 'train') not in {'train', 'train+validation'} or \
       not version_valid or metadata['output_mode'] != 'duration_components' or \
       metadata['base_model_version'] != base_model_version or \
       metadata['base_model_fingerprint'] != model_fingerprint(base_model_path) or \
       not metadata['dataset_manifest_hash'] or \
       not metadata['adapter_version']:
        raise ValueError('adapter 元数据与基础模型不兼容')
    return metadata


def duration_prediction(text, flight):
    components = parse_duration_output(text)
    planned = flight.planned_off_block.ToJsonString()
    prediction = pb.Prediction()
    for field, value in zip(TIME_FIELDS, reconstruct_times(planned, components)):
        getattr(prediction, field).FromJsonString(value)
    return prediction


def parse_output(text):
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('模型未返回 JSON 对象')
    try:
        data = json.loads(text[start:end+1])
        prediction = pb.Prediction()
        for field, label in zip(TIME_FIELDS, TIME_LABELS):
            getattr(prediction, field).FromDatetime(parse_time(data[label], 'UTC'))
        validate_response(pb.PredictResponse(trace_id='validation', prediction=prediction,
                          source=pb.LLM, model_version='validation', worker_id='validation'))
        return prediction
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('模型预测字段或时间顺序无效') from exc


def resolve_output_mode(mode, has_adapter):
    if mode not in (None, 'absolute_times', 'duration_components'):
        raise ValueError('无效输出模式')
    if has_adapter and mode == 'absolute_times':
        raise ValueError('duration adapter 与 absolute_times 不兼容')
    return mode or ('duration_components' if has_adapter else 'absolute_times')


class QwenBackend:
    source = pb.LLM

    def __init__(self, model_path, max_new_tokens=192, max_seconds=20, adapter_path=None, output_mode=None):
        mode = resolve_output_mode(output_mode, adapter_path is not None)
        if not model_path:
            raise ValueError('Qwen 必须配置本地模型路径')
        path = Path(model_path).resolve()
        if not path.is_dir() or not (path/'config.json').is_file():
            raise FileNotFoundError('模型目录不存在或缺少 config.json')
        if not 1 <= max_new_tokens <= 4096 or not 0 < max_seconds <= 300:
            raise ValueError('生成预算超出范围')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
            torch_dtype='auto', device_map='auto').eval()
        self.output_mode = mode
        self.adapter_metadata = None
        if adapter_path is not None:
            self.adapter_metadata = load_adapter_metadata(adapter_path, path.name, path)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, str(Path(adapter_path).resolve()), local_files_only=True
            ).eval()
            self.output_mode = 'duration_components'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens, self.max_seconds = max_new_tokens, max_seconds
        self.model_version = (path.name if self.adapter_metadata is None else
                              f"{path.name}+{self.adapter_metadata['adapter_version']}")[:128]
        self.prompt_version = (self.adapter_metadata['prompt_version'] if self.adapter_metadata
                               else PROMPT_VERSION)
        if self.adapter_metadata is None and self.output_mode == 'duration_components':
            self.model_version = f'{path.name}:{PROMPT_VERSION}'[:128]

    def predict(self, flight, cancelled, time_budget):
        features = MessageToDict(flight, preserving_proto_field_name=True)
        return self._predict_features(flight, features, cancelled, time_budget)

    def predict_request(self, request, cancelled, time_budget):
        features = MessageToDict(request.flight, preserving_proto_field_name=True)
        if request.HasField('zggg_context'):
            context = MessageToDict(request.zggg_context, preserving_proto_field_name=True)
            features.update({
                'airport_direction': ('DEPARTURE' if request.flight.departure_airport == 'ZGGG'
                                      else 'ARRIVAL'),
                'prediction_cutoff_time': request.zggg_context.prediction_cutoff.ToJsonString(),
                'departure_weather': context.get('departure_weather', {}),
                'arrival_weather': context.get('arrival_weather', {}),
            })
        return self._predict_features(request.flight, features, cancelled, time_budget)

    def _predict_features(self, flight, features, cancelled, time_budget):
        from transformers import StoppingCriteria, StoppingCriteriaList
        budget = min(time_budget, self.max_seconds)
        if budget <= 0 or cancelled():
            raise TimeoutError('推理预算耗尽')
        deadline = time.monotonic() + budget
        torch = self.torch

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                stop = cancelled() or time.monotonic() >= deadline
                return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)

        prompt_version = (self.adapter_metadata['prompt_version'] if self.adapter_metadata else
                          PROMPT_VERSION)
        prompt = (render_duration_prompt(features, prompt_version) if self.output_mode == 'duration_components' else
                  '根据下列航班计划、两站 METAR 和流量特征预测四个运行时刻。'
                  '所有输入时间均为 UTC；输出必须是 UTC ISO-8601 时间并带 Z。'
                  '只返回一个 JSON 对象，键为实际离港时间、实际起飞时间、实际落地时间、实际到港时间。'
                  '保持离港≤起飞≤落地≤到港，不输出思考过程或解释。\n'
                  + json.dumps(features, ensure_ascii=False))
        rendered = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(rendered, return_tensors='pt', add_special_tokens=False)
        if inputs['input_ids'].shape[1] > 8192:
            raise ValueError('模型输入超过 8192 token')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, stopping_criteria=StoppingCriteriaList([Stop()]),
                pad_token_id=self.tokenizer.pad_token_id)
        if cancelled() or time.monotonic() >= deadline:
            raise TimeoutError('推理已取消或超时')
        text = self.tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return duration_prediction(text, flight) if self.output_mode == 'duration_components' else parse_output(text)
