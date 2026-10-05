# 部署教程

命令均在仓库根目录执行。先跑 TEST 验证工程链路，再准备自己的模型、数据及私有配置。部署目录不要包含空格；运行配置、数据库、凭据与权重不提交 Git。

## 1. 构建公共源码

以下系统包命令适用于 Ubuntu/Debian；其他系统使用对应的包管理器。Python 使用 3.10。

```bash
sudo apt-get update
sudo apt-get install -y python3.10-venv cmake g++ pkg-config libprotobuf-dev protobuf-compiler libgrpc++-dev protobuf-compiler-grpc
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
bash scripts/generate_proto.sh
cmake -S gateway -B build/phase2 -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase2 -j2
ln -s phase2 build/phase2-release
ctest --test-dir build/phase2 --output-on-failure
```

若发行版不提供 Python 3.10，请先配置对应的软件源或安装 Python 3.10，并安装与解释器版本匹配的 venv 软件包；`ensurepip` 缺失时不能跳过这一步。

最后的软链接只在目标不存在时创建。已有构建目录时检查其内容，不覆盖。

## 2. 无 GPU 基础演示

```bash
.venv/bin/python scripts/run_local.py --backend test --worker-count 2
```

浏览器访问 `http://127.0.0.1:5000/flight_input`，导入 [合成示例](../examples/README.md)。启动器创建两个 Worker 和一个 Gateway，等待 gRPC 健康检查后启动 Web。使用 Ctrl+C 结束；生成配置和日志位于 `runtime/`。

这是同步预测演示，默认不启动 MySQL/Redis，也不开放完整持久恢复能力。TEST 不验证模型精度和机场流量。

## 3. 完整持久栈：先用 TEST

需要 **MySQL 8 的 mysqld、Redis、Nginx** 可执行文件。Ubuntu 上可安装 `mysql-server redis-server nginx`；软件包可能启动系统服务，项目自身则使用独立数据目录、Unix socket 和 Supervisor。不要把配置指向其他业务的数据库目录。

以下生成器仅为本项目创建私有配置，使用短临时路径避免 Unix socket 长度限制：

```bash
mkdir -p runtime/demo
.venv/bin/python - <<'PYTHON'
import json, os, shutil
from pathlib import Path
root = Path.cwd().resolve()
config = {
    'project': str(root),
    'runtime': f'/tmp/flight-demo-{os.getuid()}',
    'backend': 'test',
    'gateway': str(root / 'build/phase2/flight_gateway'),
}
for field, executable in [('mysqld', 'mysqld'), ('redis', 'redis-server'), ('nginx', 'nginx')]:
    path = shutil.which(executable)
    if path is None:
        raise SystemExit(f'Missing executable: {executable}')
    config[field] = path
from deploy.distributed.resident import validate
validate(config)
Path('runtime/demo/resident.private.json').write_text(json.dumps(config, indent=2) + '\n')
PYTHON
.venv/bin/python -m scripts.resident_ops up --config runtime/demo/resident.private.json
.venv/bin/python -m scripts.resident_ops health --config runtime/demo/resident.private.json
```

健康检查包含 HTTP `/ready`，同时验证数据库、队列、Worker 池及模型发布身份。显式 `TEST/test-v1` 演示无需真实模型工件；真实模型仍须通过发布清单与工件校验。自定义安装的 Nginx 等程序若不在 PATH 中，请在配置中填写其绝对路径。

默认 HTTP 入口为 `http://127.0.0.1:8080`。可在配置中调整端口，所有角色端口须互不相同且未被占用。生命周期入口依次进行预检查、初始化和启动；初始化的是新项目数据目录。此 TEST 演示使用本地隔离数据库，生产数据库权限和凭据需按环境另行配置。

### 持久预测 → 恢复

```bash
.venv/bin/python - <<'PYTHON'
import json
from pathlib import Path
flights = json.loads(Path('examples/synthetic-batch.json').read_text())
Path('runtime/demo/prediction-request.json').write_text(json.dumps({'flights': flights}, ensure_ascii=False))
PYTHON
curl --fail-with-body -sS http://127.0.0.1:8080/api/v1/prediction-jobs \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: synthetic-demo-prediction-v1' \
  --data-binary @runtime/demo/prediction-request.json
```

响应返回 `job_id` 与 `status_url`。查询返回的 `status_url`，或访问 `/api/v1/prediction-jobs/<job_id>/events` 读取 SSE；等待终态 `SUCCEEDED`。在页面打开对应任务，进入态势与恢复页面，采用示例中的时间范围及容量。

接口提交恢复时，正文为：

```json
{
  "source_job_id": "<job_id>",
  "scenario_version": "synthetic-demo-v1",
  "scenario": {
    "airport": "ZGGG",
    "horizon_start": "2030-01-15T08:00:00+08:00",
    "horizon_end": "2030-01-15T20:00:00+08:00",
    "slot_minutes": 15,
    "departure_capacity": 1,
    "arrival_capacity": 1,
    "mtt_minutes": 30,
    "closures": []
  }
}
```

POST 至 `/api/v1/recovery-jobs` 并附唯一 `Idempotency-Key`。检查 `status`、`validation_errors`、`summary` 与风险快照；同一幂等键不要用于不同请求。

停止项目进程：

```bash
.venv/bin/python -m scripts.resident_ops stop --config runtime/demo/resident.private.json
```

`/tmp` 可能在重启时清理；长期部署改用持久的独立短目录，停止和备份后再迁移。不要直接删除运行中的数据库。

## 4. 单 GPU 与真实组合模型

准备自己的完整 Qwen 基座、匹配 Adapter、历史和流量模型，以及训练评测产生的发布清单。CUDA/PyTorch 依赖按服务器配置安装，参考 `requirements-worker.txt`；源码不会自动下载权重。

独立 Worker 启动示例：

```bash
PYTHONPATH="$PWD/generated:$PWD" .venv/bin/python -m services.inference.server \
  --listen 127.0.0.1:50052 --backend qwen --worker-id worker-1 \
  --model-path /path/to/qwen-base --adapter-path /path/to/adapter \
  --output-mode duration_components --max-inflight 1
```

另开终端启动网关 `build/phase2/flight_gateway --config configs/gateway.local.json`。该配置只有一个容量为 1 的节点。这个例子用于理解 Worker 与 Gateway 的连接；完整业务栈应交给 Resident 管理，避免端口重复启动。

真实组合业务使用 `backend: composite`、`release_manifest_path` 及 `experimental_model: true`。代码从清单加载模型工件和身份；当前模型未通过全部精度门槛，必须显式允许实验部署。模型输出模式、特征契约、Adapter 版本与数据清单需一致，不能仅替换模型名称。若配置历史回放，必须同时提供 `replay_root` 和 `replay_dataset_id`，并通过清单身份校验。

使用 `deploy/distributed/` 中的模板生成自己的配置，先执行 `python -m deploy.distributed.resident preflight --config <私有配置>`，再启动并检验实际模型身份。合成 TEST 示例不包含真实组合模型需要的完整历史特征，不能直接替代真实数据验收。

## 5. 多服务器与多 Worker

业务中心运行 Web、Gateway、数据库和持久角色；每台 GPU 服务器运行一个或多个完整模型 Worker。本机可以只打开浏览器，或通过 SSH 转发访问中心 Web。

1. 每个 Worker 准备相同发布身份和可验证工件，配置不同 `worker_id`。
2. Worker 绑定回环或受控专网；通过 SSH 隧道让中心 Gateway 访问远端 gRPC。
3. 在节点清单中填中心可达地址、容量及启用状态；容量与 Worker 的并发上限一致。
4. 使用 `deploy/autodl/examples/cluster.example.json` 和相关模块配置隧道与常驻角色。真实端点、密钥和 known_hosts 只保存在私有配置。
5. 验证 Health、实际模型身份、两端请求分布、单节点故障转移和恢复。Health 为 SERVING 不代表精度通过，也不单独证明两台机器都参与了请求。

这是请求级并行，每个 Worker 持有完整模型；不是把一个模型跨机器切分。仅在真实双端环境中才能验收跨服务器行为，普通 CI 不承担这项验证。

排查入口见 [常见问题](TROUBLESHOOTING.md)，配置字段见 [配置说明](CONFIGURATION.md)。

完整真实工件准备、Resident 双服务器节点配置和操作顺序见 [GPU 与多服务器教程](GPU_DEPLOYMENT.md)。
