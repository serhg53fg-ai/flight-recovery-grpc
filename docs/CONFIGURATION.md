# 配置说明

## 基础启动器

`python scripts/run_local.py --help` 查看所有参数。默认 Web 使用回环地址和 5000 端口，Gateway 为 50051，首个 Worker 为 50052。启动器按 Worker 数量生成节点清单，并把容量与 `--worker-max-inflight` 对齐。

## 网关

配置参考 `configs/gateway.local.json`。修改副本后通过 `--config` 指定，不把真实部署清单提交仓库。

| 字段 | 含义 |
|---|---|
| `listen` | Gateway 监听地址 |
| `rpc_timeout_ms` | 请求总预算 |
| `health_interval_ms` / `health_timeout_ms` | 健康探测周期与预算 |
| `failure_threshold` | 达到失败次数后开启熔断 |
| `open_cooldown_ms` | 开路冷却时间 |
| `minimum_retry_budget_ms` | 故障转移所需最小剩余预算 |
| `workers[].id` / `address` | 节点身份和中心可达 gRPC 地址 |
| `workers[].capacity` / `enabled` | 准入容量与启用状态 |

容量应与 Worker `--max-inflight` 一致。更多节点不会自动增加单张卡能容纳的模型大小。远程地址填写隧道在中心暴露的端口时，保持监听与转发方向一致。

## Resident 持久栈

`deploy.distributed.resident.validate` 是配置校验入口。基本字段为 `project`、`runtime`、`mysqld`、`redis`、`nginx`、`gateway` 和 `backend`，前六项均为绝对路径。生命周期命令见 [部署教程](DEPLOYMENT.md)。

`backend` 支持 `test`、`qwen`、`composite`。真实 Qwen 需要模型路径；组合模型通过发布清单加载工件。实验发布必须显式设置 `experimental_model: true`。工件摘要由自己的发布流程产生，模板占位字符串不能充当真实身份。

发布身份关联模型、Adapter、历史/流量工件、数据与特征契约。启用回放时的数据身份必须匹配；天气和历史输入遵守预测截止时刻，不读取之后的观测或真实运行标签。

实际口令、SSH 私钥、主机授权和运行数据库均只在私有环境维护。忽略规则只是防误提交，发布前仍要执行 `scripts.release_audit` 并人工检查。
