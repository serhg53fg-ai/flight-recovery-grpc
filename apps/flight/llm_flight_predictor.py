import json
import sys
import os
from typing import Any, Dict, List
from datetime import datetime, timedelta
import traceback
import importlib.util
import threading

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except Exception:
    SAFETENSORS_AVAILABLE = False

# 添加气象路由目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '气象路由'))

AutoModelForCausalLM = None
AutoTokenizer = None
TRANSFORMERS_AVAILABLE = False
TRANSFORMERS_VERSION = None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _ensure_transformers() -> bool:
    global AutoModelForCausalLM, AutoTokenizer, TRANSFORMERS_AVAILABLE, TRANSFORMERS_VERSION
    if TRANSFORMERS_AVAILABLE and AutoModelForCausalLM is not None and AutoTokenizer is not None:
        return True
    try:
        import transformers
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM, AutoTokenizer as _AutoTokenizer
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        TRANSFORMERS_VERSION = getattr(transformers, "__version__", "unknown")
        TRANSFORMERS_AVAILABLE = True
        return True
    except Exception:
        TRANSFORMERS_AVAILABLE = False
        return False


def _parse_version(version_str: str):
    nums = []
    for part in (version_str or "").split('.'):
        if part.isdigit():
            nums.append(int(part))
        else:
            prefix = ''
            for ch in part:
                if ch.isdigit():
                    prefix += ch
                else:
                    break
            nums.append(int(prefix) if prefix else 0)
        if len(nums) >= 3:
            break
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def _is_transformers_qwen3_ready() -> bool:
    if not _ensure_transformers():
        return False
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        if 'qwen3' in CONFIG_MAPPING_NAMES:
            return True
    except Exception:
        pass
    # 兜底按版本判断，Qwen3在较新版本中可用。
    return _parse_version(TRANSFORMERS_VERSION or "0.0.0") >= (4, 51, 0)

class LLMFlightPredictor:
    """基于大语言模型的航班预测器"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_path = None
        self.model_source_type = None
        self.last_error = None
        self.max_new_tokens = _env_int("LLM_MAX_NEW_TOKENS", 192)
        self.max_generate_time_sec = _env_float("LLM_MAX_GENERATE_TIME_SEC", 20.0)
        self.device_map = os.environ.get("LLM_DEVICE_MAP", "auto").strip()
        self.torch_dtype = os.environ.get("LLM_TORCH_DTYPE", "auto").strip().lower()
        self.local_files_only = _env_bool("LLM_LOCAL_FILES_ONLY", True)
        self.allow_remote_model = _env_bool("LLM_ALLOW_REMOTE_MODEL", True)
        self.model_id = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen3-1.7B").strip()
        self.trust_remote_code = _env_bool("LLM_TRUST_REMOTE_CODE", True)
        self.enable_thinking = _env_bool("LLM_ENABLE_THINKING", False)
        self.do_sample = _env_bool("LLM_DO_SAMPLE", False)
        self.temperature = _env_float("LLM_TEMPERATURE", 0.7)
        self.top_p = _env_float("LLM_TOP_P", 0.9)
        self.min_core_fields = max(0, _env_int("LLM_MIN_CORE_FIELDS", 1))
        self._lock = threading.RLock()

        if _ensure_transformers():
            self.load_model()
        else:
            self.last_error = (
                "缺少transformers依赖。即使有本地权重，也需要至少安装: "
                "transformers、torch、safetensors"
            )
            print("警告: transformers库未安装，将使用模拟预测")

    def dependency_status(self) -> Dict[str, Any]:
        return {
            "transformers": _module_available("transformers"),
            "torch": _module_available("torch"),
            "accelerate": _module_available("accelerate"),
            "safetensors": _module_available("safetensors"),
        }

    def runtime_status(self) -> Dict[str, Any]:
        torch_available = _module_available("torch")
        cuda_available = False
        cuda_device_count = 0
        if torch_available:
            try:
                import torch
                cuda_available = bool(torch.cuda.is_available())
                cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
            except Exception:
                pass

        model_device = None
        try:
            model_device = str(getattr(self.model, "device", None)) if self.model is not None else None
        except Exception:
            model_device = None

        return {
            "python_version": sys.version.split()[0],
            "transformers_version": TRANSFORMERS_VERSION,
            "model_loaded": self.model_loaded,
            "model_device": model_device,
            "model_source": self.model_path,
            "model_source_type": self.model_source_type,
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "allow_remote_model": self.allow_remote_model,
            "model_id": self.model_id,
            "enable_thinking": self.enable_thinking,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
        }

    def _resolve_torch_dtype(self):
        if self.torch_dtype in ("", "auto"):
            return "auto"
        if not _module_available("torch"):
            return "auto"
        try:
            import torch
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            return mapping.get(self.torch_dtype, "auto")
        except Exception:
            return "auto"

    def _is_valid_model_dir(self, model_dir: str) -> bool:
        """检查本地模型目录是否具备最基本加载条件。"""
        if not os.path.isdir(model_dir):
            return False

        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        for name in required_files:
            if not os.path.exists(os.path.join(model_dir, name)):
                return False

        has_weight = any(
            os.path.exists(os.path.join(model_dir, f))
            for f in [
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
            ]
        )
        return has_weight

    def _resolve_model_path(self):
        """自动解析可用模型来源：优先本地目录，否则回退远程模型ID。"""
        env_path = os.environ.get("LLM_MODEL_PATH", "").strip()
        if env_path and os.path.isdir(env_path):
            if self._is_valid_model_dir(env_path):
                return env_path, True
            print(f"LLM_MODEL_PATH目录缺少关键文件，将继续自动探测: {env_path}")

        base_models_dir = os.path.join(os.path.dirname(__file__), '气象路由', 'models')
        # 恢复为qwen3优先，避免自动命中已损坏的qwen3_1目录。
        candidates = ['qwen3', 'qwen3-1']

        for name in candidates:
            path = os.path.join(base_models_dir, name)
            if self._is_valid_model_dir(path):
                return path, True

        if self.allow_remote_model and self.model_id:
            return self.model_id, False

        return None, True

    def _quarantine_corrupt_single_safetensor(self, model_path: str):
        """若发现单文件 model.safetensors 损坏，则重命名隔离，避免阻塞分片权重加载。"""
        single_file = os.path.join(model_path, 'model.safetensors')
        index_file = os.path.join(model_path, 'model.safetensors.index.json')

        if not os.path.exists(single_file):
            return

        # 只有在存在分片索引时才隔离单文件，避免误删唯一权重来源
        if not os.path.exists(index_file):
            return

        if not SAFETENSORS_AVAILABLE:
            return

        try:
            with safe_open(single_file, framework='pt'):
                pass
        except Exception as e:
            quarantined = f"{single_file}.corrupt"
            try:
                if os.path.exists(quarantined):
                    os.remove(quarantined)
                os.replace(single_file, quarantined)
                print(f"检测到损坏的单文件权重，已隔离: {quarantined}。原因: {e}")
            except Exception as move_e:
                print(f"检测到损坏单文件权重但隔离失败: {move_e}")

    def load_model(self):
        """加载大语言模型"""
        with self._lock:
            try:
                if not _ensure_transformers():
                    self.model_loaded = False
                    self.last_error = (
                        "transformers未安装，无法加载本地权重。"
                        "请安装: transformers torch safetensors"
                    )
                    return False

                model_path, is_local_model = self._resolve_model_path()

                if not model_path:
                    self.model_loaded = False
                    self.last_error = (
                        "未找到可用模型来源。请设置LLM_MODEL_PATH指向本地权重目录，"
                        "或设置LLM_MODEL_ID并开启LLM_ALLOW_REMOTE_MODEL=1"
                    )
                    print(f"模型路径不存在: {self.last_error}")
                    return False

                self.model_path = model_path
                self.model_source_type = "local" if is_local_model else "remote"
                if is_local_model:
                    self._quarantine_corrupt_single_safetensor(model_path)
                print(f"正在加载大语言模型({self.model_source_type}): {model_path}")

                model_type = None
                if is_local_model:
                    cfg_path = os.path.join(model_path, 'config.json')
                    try:
                        with open(cfg_path, 'r', encoding='utf-8') as f:
                            cfg = json.load(f)
                        model_type = cfg.get('model_type')
                    except Exception:
                        model_type = None
                else:
                    if "qwen3" in str(model_path).lower():
                        model_type = "qwen3"

                if model_type == 'qwen3' and not _is_transformers_qwen3_ready():
                    self.model_loaded = False
                    self.last_error = (
                        f"检测到模型类型为qwen3，但当前transformers版本({TRANSFORMERS_VERSION})"
                        "不支持。请升级到>=4.51.0。"
                    )
                    print(self.last_error)
                    return False

                dtype = self._resolve_torch_dtype()
                primary_kwargs = {
                    "trust_remote_code": self.trust_remote_code,
                    "local_files_only": self.local_files_only if is_local_model else False,
                    "torch_dtype": dtype,
                }
                if self.device_map:
                    primary_kwargs["device_map"] = self.device_map

                # 优先按平台配置加载；失败时去掉device_map等强约束，提升兼容性。
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **primary_kwargs,
                    )
                except Exception as first_error:
                    fallback_kwargs = {
                        "trust_remote_code": self.trust_remote_code,
                        "local_files_only": self.local_files_only if is_local_model else False,
                    }
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **fallback_kwargs,
                    )
                    print(f"主加载参数失败，已回退兼容参数加载: {first_error}")

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=self.local_files_only if is_local_model else False,
                )

                # 设置填充令牌以避免警告
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model_loaded = True
                self.last_error = None
                print(f"大语言模型加载成功！当前模型目录: {model_path}")
                return True

            except Exception as e:
                self.model_loaded = False
                self.last_error = str(e)
                print(f"大语言模型加载失败: {str(e)}")
                traceback.print_exc()
                return False

    def _normalize_airport_code(self, value: Any) -> Any:
        if value is None:
            return value
        text = str(value).strip().upper()
        if len(text) == 4 and text.isalnum():
            return text
        return value

    def _normalize_time_text(self, value: Any) -> Any:
        if value is None:
            return value
        text = str(value).strip()
        if not text:
            return value

        fmts = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%dT%H:%M',
        ]
        for fmt in fmts:
            try:
                dt = datetime.strptime(text, fmt)
                return dt.strftime('%Y-%m-%dT%H:%M:%S')
            except Exception:
                continue
        return value

    def _normalize_formatted_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(data)

        for key in ['departure_airport', 'actual_departure_airport', 'arrival_airport', 'actual_arrival_airport']:
            if key in out:
                out[key] = self._normalize_airport_code(out.get(key))

        for key in ['planned_departure_time', 'planned_arrival_time']:
            if key in out:
                out[key] = self._normalize_time_text(out.get(key))

        if 'flight_number' in out and out['flight_number'] is not None:
            out['flight_number'] = str(out['flight_number']).strip().upper()

        if 'tail_number' in out and out['tail_number'] is not None:
            out['tail_number'] = str(out['tail_number']).strip().upper()

        return out

    def _build_input_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        core_fields = [
            'flight_number',
            'tail_number',
            'departure_airport',
            'arrival_airport',
            'planned_departure_time',
            'planned_arrival_time',
        ]
        important_fields = [
            'aircraft_type',
            'departure_metar',
            'arrival_metar',
            'planned_distance',
            'planned_flight_time',
            'planned_total_flow',
        ]

        core_present = [k for k in core_fields if data.get(k) not in (None, "")]
        important_present = [k for k in important_fields if data.get(k) not in (None, "")]

        return {
            'core_fields_total': len(core_fields),
            'core_fields_present': len(core_present),
            'core_fields_missing': [k for k in core_fields if k not in core_present],
            'important_fields_total': len(important_fields),
            'important_fields_present': len(important_present),
            'important_fields_missing': [k for k in important_fields if k not in important_present],
        }

    def format_prompt(self, flight_data: Dict[str, Any], input_quality: Dict[str, Any]) -> str:
        """格式化提示词"""

        TASK_INSTRUCTION = """
你是一个航班预测的助手，你的任务是根据用户的问题，预测用户的航班信息。
<conversation>

{conversation}

</conversation>
"""

        FORMAT_PROMPT = """
你的任务是根据<conversation></conversation> XML标签中的的航班相关信息，预测航班的"实际离港时间"，"实际起飞时间"，"实际落地时间"，"实际到港时间"这四个量。请按照以下指示操作：
1. 你必须重复分析<conversation></conversation> XML标签中的航班相关信息，预测出该航班的"实际离港时间"，"实际起飞时间"，"实际落地时间"，"实际到港时间"这四个量。
2. 你必须综合考虑所给的机尾号、航班号、机型、计划离港时间、计划到港时间、METAR气象报文、流量、航距等信息。
3. 你必须根据所给的信息，预测出该航班的"实际离港时间"，"实际起飞时间"，"实际落地时间"，"实际到港时间"这四个量。
    4. 你必须只输出一个JSON对象，不要输出解释文字、前后缀、代码块标记。
    5. 如果关键字段缺失，不要编造不存在的机场/航班事实；请基于已给信息给出保守预测，并保持时间格式一致。

根据你的分析，如果你已经预测出了时间，请按照以下JSON格式提供你的响应：
{"实际离港时间": "2023-08-01T10:00:00", "实际起飞时间": "2023-08-01T10:15:00", "实际落地时间": "2023-08-01T11:00:00", "实际到港时间": "2023-08-01T11:15:00"}
"""

        quality_block = """
输入质量评估（用于约束回答保守性）：
{quality}
""".format(quality=json.dumps(input_quality, ensure_ascii=False, indent=2))

        # 构建对话数据
        conversation = [
            {
                "role": "user",
                "content": json.dumps(flight_data, ensure_ascii=False, indent=2)
            }
        ]

        return (
            TASK_INSTRUCTION.format(
                conversation=json.dumps(conversation, ensure_ascii=False, indent=2)
            )
            + quality_block
            + FORMAT_PROMPT
        )

    def predict(self, flight_data: Dict[str, Any]) -> Dict[str, str]:
        """使用大语言模型进行航班预测"""
        try:
            with self._lock:
                if not self.model_loaded:
                    # 支持依赖补齐后的热重载，无需重启进程。
                    self.load_model()

                if not self.model_loaded:
                    print(f"LLM未加载，跳过LLM预测并交由上层回退。原因: {self.last_error}")
                    return None

                # 格式化输入数据
                formatted_data = self.format_flight_data(flight_data)
                input_quality = self._build_input_quality(formatted_data)

                if input_quality['core_fields_present'] < self.min_core_fields:
                    self.last_error = (
                        "input core feature coverage too low: "
                        f"{input_quality['core_fields_present']}/{input_quality['core_fields_total']}"
                    )
                    print(f"输入关键字段过少，跳过LLM预测并交由上层回退。详情: {self.last_error}")
                    return None

                if input_quality['core_fields_present'] < 3:
                    print(
                        "输入关键字段覆盖率较低，继续LLM预测但结果可信度有限。"
                        f"详情: {input_quality['core_fields_present']}/{input_quality['core_fields_total']}"
                    )

                # 生成提示词
                prompt = self.format_prompt(formatted_data, input_quality)

                messages = [
                    {"role": "user", "content": prompt},
                ]

                # 不同transformers版本对apply_chat_template返回类型不一致，统一规整为Tensor。
                chat_out = None
                try:
                    chat_out = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt",
                        enable_thinking=self.enable_thinking,
                    )
                except TypeError:
                    try:
                        chat_out = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_tensors="pt",
                        )
                    except TypeError:
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                        chat_out = self.tokenizer(text, return_tensors="pt")

                input_ids = chat_out
                if isinstance(input_ids, dict):
                    input_ids = input_ids.get("input_ids")
                elif hasattr(input_ids, "input_ids"):
                    input_ids = input_ids.input_ids

                if input_ids is None:
                    raise RuntimeError("tokenizer未返回input_ids")

                if isinstance(input_ids, str):
                    input_ids = self.tokenizer(input_ids, return_tensors="pt").input_ids

                if isinstance(input_ids, list):
                    import torch
                    input_ids = torch.tensor(input_ids, dtype=torch.long)

                if not hasattr(input_ids, "shape"):
                    raise TypeError(f"unexpected input_ids type: {type(input_ids)}")

                if len(input_ids.shape) == 1:
                    input_ids = input_ids.unsqueeze(0)

                input_ids = input_ids.to(self.model.device)

                # 创建注意力掩码
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.model.device)

                generate_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": self.do_sample,
                    "max_time": self.max_generate_time_sec,
                    "use_cache": True,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                if self.do_sample:
                    generate_kwargs["temperature"] = self.temperature
                    generate_kwargs["top_p"] = self.top_p

                # 生成预测
                generated_ids = self.model.generate(**generate_kwargs)

                # 提取生成的内容
                prompt_lengths = input_ids.shape[1]
                generated_only = [
                    output_ids[prompt_lengths:]
                    for output_ids in generated_ids
                ]

                # 解码响应
                response = self.tokenizer.batch_decode(generated_only, skip_special_tokens=True)[0]

                # 解析JSON响应
                parsed = self.parse_prediction_response(response)
                if parsed is None:
                    self.last_error = "llm response is not valid JSON"
                return parsed

        except BaseException as e:
            print(f"大语言模型预测失败: {str(e)}")
            traceback.print_exc()
            self.last_error = str(e)
            return None

    def format_flight_data(self, flight_data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化航班数据为标准格式"""
        formatted = {}
        scenario_dict = {}

        # 字段映射
        field_mapping = {
            'tail_number': ['机尾号', 'tail_number', 'tailNo', 'tail_no'],
            'flight_number': ['航班号', 'flight_number', 'flightNo', 'flight_no'],
            'aircraft_type': ['机型', 'aircraft_type', 'plane_type'],
            'nature': ['性质', 'nature'],
            'departure_airport': ['计划起飞站四字码', 'departure_airport', '起飞站四字码', 'airport', 'dep_airport', 'origin_airport'],
            'actual_departure_airport': ['实际起飞站四字码', 'actual_departure_airport'],
            'arrival_airport': ['计划到达站四字码', 'arrival_airport', '到达站四字码', 'dest_airport', 'destination_airport'],
            'actual_arrival_airport': ['实际到达站四字码', 'actual_arrival_airport'],
            'planned_departure_time': ['计划离港时间', 'planned_departure_time', 'etd', 'scheduled_departure_time'],
            'planned_arrival_time': ['计划到港时间', 'planned_arrival_time', 'eta', 'scheduled_arrival_time'],
            'planned_distance': ['计划地面航程_Mile', 'planned_distance'],
            'actual_distance': ['实际航程_Mile', 'actual_distance'],
            'planned_flight_time': ['计划航段时间（24年同航季平均值）', 'planned_flight_time'],
            'actual_flight_time': ['实际航段时间（24年同航季平均值）', 'actual_flight_time'],
            'planned_takeoff_count': ['计划起飞数', 'planned_takeoff_count'],
            'planned_landing_count': ['计划降落数', 'planned_landing_count'],
            'planned_total_flow': ['计划总流量', 'planned_total_flow'],
            'actual_takeoff_count': ['实际起飞数', 'actual_takeoff_count'],
            'actual_landing_count': ['实际降落数', 'actual_landing_count'],
            'actual_total_flow': ['实际总流量', 'actual_total_flow'],
            'departure_metar': ['起飞站METAR', 'departure_metar'],
            'arrival_metar': ['到达站METAR', 'arrival_metar']
        }

        # 兼容当前RPC链路：scenario_json通常携带主体航班上下文。
        raw_scenario = flight_data.get('scenario_json') if isinstance(flight_data, dict) else None
        if isinstance(raw_scenario, dict):
            scenario_dict = raw_scenario
        elif isinstance(raw_scenario, str) and raw_scenario.strip():
            try:
                parsed = json.loads(raw_scenario)
                if isinstance(parsed, dict):
                    scenario_dict = parsed
            except Exception:
                scenario_dict = {}

        # 映射字段
        for standard_key, possible_keys in field_mapping.items():
            for key in possible_keys:
                if key in flight_data and flight_data[key] is not None:
                    formatted[standard_key] = flight_data[key]
                    break

            if standard_key not in formatted:
                for key in possible_keys:
                    if key in scenario_dict and scenario_dict[key] is not None:
                        formatted[standard_key] = scenario_dict[key]
                        break

        # 保留链路中的上下文字段，避免模型看到空输入。
        if isinstance(flight_data, dict):
            trace_id = flight_data.get('trace_id')
            prompt = flight_data.get('prompt')
            if trace_id:
                formatted['trace_id'] = trace_id
            if prompt:
                formatted['prompt'] = prompt

        if scenario_dict:
            formatted['scenario'] = scenario_dict

        # 保留风险上下文字段，便于模型理解扰动级别。
        for key in ['impact', 'delay_ratio', 'risk_level', 'scenario_tag']:
            if key in flight_data and flight_data[key] not in (None, ""):
                formatted[key] = flight_data[key]
            elif key in scenario_dict and scenario_dict[key] not in (None, ""):
                formatted[key] = scenario_dict[key]

        # 兜底：若标准映射为空，直接透传原始入参，避免模型无上下文。
        if not formatted and isinstance(flight_data, dict):
            for k, v in flight_data.items():
                if v is not None and v != "":
                    formatted[k] = v

        formatted = self._normalize_formatted_fields(formatted)
        return formatted

    def parse_prediction_response(self, response: str):
        """解析大语言模型的预测响应"""
        try:
            # 尝试从响应中提取JSON
            import re

            required_fields = ["实际离港时间", "实际起飞时间", "实际落地时间", "实际到港时间"]

            # 1) 优先截取首尾大括号，兼容模型输出前后夹杂解释文本。
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = response[start:end + 1]
                try:
                    obj = json.loads(candidate)
                    if all(field in obj for field in required_fields):
                        return obj
                except Exception:
                    pass

            # 查找JSON格式的预测结果
            json_pattern = r'\{[^}]*"实际离港时间"[^}]*\}'
            match = re.search(json_pattern, response)

            if match:
                json_str = match.group(0)
                prediction = json.loads(json_str)
                if all(field in prediction for field in required_fields):
                    return prediction

            # 无法解析时返回None，由上层决定是否回退到模拟预测。
            print(f"无法解析大语言模型响应: {response}")
            return None

        except Exception as e:
            print(f"解析预测响应失败: {str(e)}")
            return None

    def simulate_prediction(self, flight_data: Dict[str, Any]) -> Dict[str, str]:
        """模拟预测结果（当大语言模型不可用时）"""
        try:
            # 获取计划时间
            planned_departure = flight_data.get('planned_departure_time') or flight_data.get('计划离港时间')
            planned_arrival = flight_data.get('planned_arrival_time') or flight_data.get('计划到港时间')

            if planned_departure:
                if 'T' in str(planned_departure):
                    base_departure = datetime.fromisoformat(str(planned_departure).replace('T', ' '))
                else:
                    base_departure = datetime.strptime(str(planned_departure), '%Y-%m-%d %H:%M:%S')
            else:
                base_departure = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)

            if planned_arrival:
                if 'T' in str(planned_arrival):
                    base_arrival = datetime.fromisoformat(str(planned_arrival).replace('T', ' '))
                else:
                    base_arrival = datetime.strptime(str(planned_arrival), '%Y-%m-%d %H:%M:%S')
            else:
                base_arrival = base_departure + timedelta(hours=2)

            # 模拟延误（5-15分钟）
            import random
            departure_delay = random.randint(5, 15)
            takeoff_delay = departure_delay + random.randint(5, 10)
            landing_delay = takeoff_delay + random.randint(0, 5)
            arrival_delay = landing_delay + random.randint(5, 10)

            # 计算实际时间
            actual_departure = base_departure + timedelta(minutes=departure_delay)
            actual_takeoff = base_departure + timedelta(minutes=takeoff_delay)
            actual_landing = base_arrival + timedelta(minutes=landing_delay)
            actual_arrival = base_arrival + timedelta(minutes=arrival_delay)

            return {
                "实际离港时间": actual_departure.strftime('%Y-%m-%dT%H:%M:%S'),
                "实际起飞时间": actual_takeoff.strftime('%Y-%m-%dT%H:%M:%S'),
                "实际落地时间": actual_landing.strftime('%Y-%m-%dT%H:%M:%S'),
                "实际到港时间": actual_arrival.strftime('%Y-%m-%dT%H:%M:%S')
            }

        except Exception as e:
            print(f"模拟预测失败: {str(e)}")
            # 返回默认时间
            now = datetime.now()
            return {
                "实际离港时间": now.strftime('%Y-%m-%dT%H:%M:%S'),
                "实际起飞时间": (now + timedelta(minutes=15)).strftime('%Y-%m-%dT%H:%M:%S'),
                "实际落地时间": (now + timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%S'),
                "实际到港时间": (now + timedelta(hours=2, minutes=15)).strftime('%Y-%m-%dT%H:%M:%S')
            }

# 创建全局预测器实例
llm_predictor = LLMFlightPredictor()
