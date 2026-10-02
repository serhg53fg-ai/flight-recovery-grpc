# 航班预测多 Worker gRPC 网关

当前活动实现位于 `src/`，使用 Google gRPC 将航班预测请求路由到一个或多个 Python 推理 Worker。本公开包排除了旧 mprpc 与 HTTP 桥接示例，只保留当前活动网关。

## 构建与启动

```bash
cmake -S gateway -B build/phase2 -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase2 --parallel 2
build/phase2/flight_gateway --config configs/gateway.local.json
```

网关只接受 `flight_gateway --config PATH`。参考配置 `configs/gateway.local.json` 包含监听地址、统一 RPC 预算、健康周期、熔断阈值、冷却时间、最低重试预算以及 Worker 的 ID、地址和容量。未知字段和不安全的公网地址会导致启动失败。

## 路由与故障策略

路由候选必须启用、健康、未被熔断并且有空余容量。网关从候选中用 P2C 比较 `inflight/capacity`。首次下游调用只有返回 `UNAVAILABLE` 且剩余预算充足时才会换一个节点重试一次；`RESOURCE_EXHAUSTED`、`DATA_LOSS`、超时、取消和输入错误不会重试。

标准 `grpc.health.v1.Health` 表示集群是否至少有一个可路由节点。`flight.v1.GatewayAdmin/GetClusterStatus` 返回只读节点健康、熔断、inflight、容量和调用计数快照。

只读 Prometheus 导出器可独立启动：

```bash
PYTHONPATH=.:generated .venv/bin/python -m deploy.metrics.exporter \
  --gateway-address 127.0.0.1:50051 --listen-port 9097
curl http://127.0.0.1:9097/metrics
```

导出器仅监听回环地址；每次抓取实时查询 `GatewayAdmin`，查询失败返回 HTTP 503。它不参与预测转发，也不保存航班输入。`flight_gateway_worker_circuit_state` 使用 protobuf 枚举值，`1=CLOSED`、`2=OPEN`、`3=HALF_OPEN`。

本阶段使用 insecure 连接，仅用于本机或受信私网。双真实GPU工程链路已验收；TLS、动态服务发现及长期多GPU性能不作为已完成能力。
