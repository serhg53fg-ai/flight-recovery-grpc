# 当前入口与历史代码

当前版本的主入口是 `apps/flight/app.py`，经统一预测服务调用 C++ Gateway 与 Python Worker。新持久 API、态势和恢复分别位于 `apps/flight/api/`、`apps/flight/situation/` 和 `apps/flight/recovery/`。

| 目录或脚本 | 定位 |
|---|---|
| `services/inference/` | 当前 gRPC Worker 和模型适配器 |
| `training/` | 当前数据契约、训练与统一评测 |
| `apps/flight/legacy_routes.py` | 旧页面兼容路由，仍参与应用注册 |
| `apps/flight/new_Greedy_algorithm/` | 旧调度实现，兼容路由仍有导入；不等同于新恢复引擎 |
| `apps/flight/` 中历史训练和本地模型脚本 | 原实验参考；先检查输入和依赖，不作为新部署启动入口 |
| `scripts/` / `deploy/` | 当前启动、检查及常驻部署工具；按对应命令 `--help` 使用 |

保留历史代码用于理解演进和兼容已有页面。由于兼容模块仍有依赖，本次没有直接删除或移动这些文件。新恢复流程以不可变态势输入、确定性调度与独立校验为准，旧脚本的指标不能直接用作当前版本验收结果。

阅读顺序：[README](../README.md) → [架构](architecture.md) → [部署](DEPLOYMENT.md) → [验证](VALIDATION.md)。仅体验公开版本时，从合成示例开始，无需逐一运行历史实验脚本。
