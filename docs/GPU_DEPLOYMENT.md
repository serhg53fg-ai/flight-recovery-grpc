# 真实模型与多服务器部署

本文将源码启动与私有工件准备分开。命令在仓库根目录执行；没有 GPU 或自己的匹配工件时，使用 [TEST 教程](DEPLOYMENT.md)，不要用占位清单冒充真实发布。

## 1. 每台 GPU Worker 的依赖与工件

先按部署教程构建源码、生成协议。在 GPU 主机安装与 CUDA 驱动匹配的 PyTorch，再安装 Worker 依赖：

```bash
.venv/bin/python -m pip install -r requirements-worker.txt
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('gpu:', torch.cuda.get_device_name(0))
PY
```

PyTorch 的安装渠道与版本按目标服务器选定；这里不提供可能不兼容的固定 CUDA 命令。`requirements-worker.txt` 同时包含 QLoRA 所需库，不能替代完整训练数据准备。

准备以下私有工件：

| 工件 | 必要信息 |
|---|---|
| Qwen 基座 | config、tokenizer、完整权重；与 Adapter 基座一致 |
| Adapter | 权重、`adapter_config.json`、`adapter_metadata.json`；输出契约一致 |
| 历史模型 | 与当前特征契约匹配的兜底工件 |
| 流量模型 | 三个累计窗口、方向分量及数据身份 |
| 特征契约 | 训练生成的特征元数据 |
| 发布清单 | v2 版本、模型来源、组件版本、工件路径与校验摘要 |
| 评测证据 | 实验发布保留未通过的精度结论；不手工改为通过 |

发布准备入口为 `python -m scripts.zggg_release --help`。它接收已有训练评测产物，不会替你训练模型。实验发布使用 `--experimental`，正常发布需满足对应门槛。公开仓库不提供作者的权重、数据或内部发布清单。

## 2. 单机完整组合服务

先生成并停止 TEST Resident 栈。把私有配置另存为 GPU 配置，绑定自己的清单：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
src = Path('runtime/demo/resident.private.json')
config = json.loads(src.read_text())
config.update(backend='composite', experimental_model=True,
              runtime='/tmp/flight-gpu-demo',
              release_manifest_path='/path/to/your-release/candidate.json')
Path('runtime/demo/gpu.private.json').write_text(json.dumps(config, indent=2) + '\n')
PY
.venv/bin/python -m deploy.distributed.resident preflight --config runtime/demo/gpu.private.json
.venv/bin/python -m scripts.resident_ops up --config runtime/demo/gpu.private.json
.venv/bin/python -m scripts.resident_ops health --config runtime/demo/gpu.private.json
```

必须先替换清单路径和持久运行目录；若重用已有数据库，遵循备份与版本迁移流程。不要将 TEST 与 GPU 配置指向同一个正在运行的目录。Resident 从清单生成 Worker 参数，避免手工填写的模型版本与实际工件不一致。

如使用历史回放，私有配置同时设置 `replay_root` 和 `replay_dataset_id`。它们必须匹配发布清单的数据身份和特征契约。真实请求需要截止时刻前可用的计划、天气与历史上下文；合成 TEST 输入不能代替真实组合模型数据。

默认中心 Web 地址为 `http://127.0.0.1:8080`。本机浏览器可通过 SSH 转发中心 HTTP 端口访问，无需本机 GPU：

```bash
ssh -N -L 18080:127.0.0.1:8080 -p <SSH_PORT> <USER>@<CONTROL_HOST>
```

然后访问 `http://127.0.0.1:18080/flight_input`。

## 3. 增加第二台 Worker：保留完整持久链路

A 运行完整 Resident 栈；B 只执行模型。两端使用同一工件内容、版本身份与契约；工件在各端的绝对路径需按本机调整。清单本身使用绝对路径，复制清单不等于自动重定位权重。

在 B 准备自己的 `runtime/demo/gpu.private.json`，其中 project、工具及发布工件路径均指向 B 本机。下面复用 Resident 的参数生成器，**只启动 Worker**：

```bash
.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
from deploy.distributed.resident import validate, commands, environment
config = validate(json.loads(Path('runtime/demo/gpu.private.json').read_text()))
config['workers'] = [{
    'worker_id': 'worker-b', 'address': '127.0.0.1:50052',
    'capacity': 1, 'managed': True,
    'release_version': config['release_identity']['release_version'],
}]
argv = commands(config)['worker']
os.chdir(config['project'])
os.execve(argv[0], argv, {**os.environ, **environment(config)})
PY
```

在 A 建立 A→B 的 SSH 转发。事先核实主机指纹并准备私有密钥及 known_hosts；不得将这些文件提交仓库。

```bash
autossh -M 0 -N \
  -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/path/to/known-hosts \
  -i /path/to/private-key -p <B_SSH_PORT> \
  -L 127.0.0.1:15053:127.0.0.1:50052 <USER>@<B_HOST>
```

该隧道需要单独保持运行。A 的私有配置中设置以下节点列表，release_version 必须替换为自己的清单版本：

```json
{
  "workers": [
    {"worker_id": "worker-a", "address": "127.0.0.1:50052", "capacity": 1,
     "release_version": "<release_version>", "managed": true},
    {"worker_id": "worker-b", "address": "127.0.0.1:15053", "capacity": 1,
     "release_version": "<release_version>", "managed": false}
  ]
}
```

修改 A 的配置前，先使用旧配置停止旧栈。节点清单修改后，必须在同一运行目录重新 prepare，再 start；`resident_ops up` 不会自动覆盖已准备的不同配置：

```bash
# 停止时使用旧 runtime/stack.json，替换为 A 的真实运行目录。
.venv/bin/python -m scripts.resident_ops stop --config /path/to/A-runtime/stack.json
.venv/bin/python -m deploy.distributed.resident preflight --config runtime/demo/gpu.private.json
.venv/bin/python -m deploy.distributed.resident prepare --config runtime/demo/gpu.private.json
.venv/bin/python -m deploy.distributed.resident start --config runtime/demo/gpu.private.json
.venv/bin/python -m scripts.resident_ops health --config runtime/demo/gpu.private.json
```

prepare 重新生成配置并保留已初始化的项目数据库；操作前备份，不删除数据目录。B Worker 与隧道不属于 A Resident 管理的进程，需分别管理其生命周期。

`deploy/autodl/examples/cluster.example.json` 属于另一种轻量控制面配置，可用于参考本地节点、SSH 节点及转发字段。其 CLI 为 `python -m deploy.autodl.control --help`；不能直接当作 Resident 配置使用。需要 MySQL/Redis 持久闭环时，采用上述 Resident 节点方式。

## 4. 验收顺序

1. Health 通过后检查实际推理响应的 worker_id、模型和流量版本、source、降级字段；版本身份准入由完整栈执行。
2. 提交真实上下文的小批量，等待持久终态，验证四时刻、累计流量窗口及结果来源。
3. 提交恢复场景，核对风险快照、`validation_errors` 和未安排原因。
4. 增加并发，在响应中确认两端 worker_id 都出现；低并发下仅命中一个节点不自动等同于路由故障。
5. 暂停 B 或隧道，观察 Gateway 在总预算内故障转移，确认未出现重复业务终态；恢复 B 后检查健康和熔断恢复。
6. 保存脱敏报告，再分别停止中心、远端 Worker 和隧道。

以上步骤是部署验收流程。本轮仅验证 CPU 工程修正，没有重新运行真实 GPU 或双服务器实验；模型精度门槛仍维持现有状态。
