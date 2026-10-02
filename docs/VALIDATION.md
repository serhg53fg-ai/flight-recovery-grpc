# 模型与验收边界

## 工程

真实学校GPU栈已验证预测→态势→恢复及进程级停止/恢复。双AutoDL RTX4090已验证真实分流、隧道故障转移与熔断恢复；自动准入8请求全部成功、覆盖两个节点且未降级，并拒绝错误版本。

这些是有限验收证据，不代表长期QPS、生产SLA或任意镜像恢复成功。公开仓库不包含内部原始验收报告；`reports/` 为输出占位目录。可复现的公开示例见 `examples/`，本地验证命令见 README。

## 精度

既有五月3266条回顾性测试，分钟MAE如下：

| 指标 | 历史基线 | Qwen1.7B Adapter | Qwen4B Adapter |
|---|---:|---:|---:|
| 四事件综合 | 12.602 | 14.309 | 14.067 |
| 离港推出 | 15.718 | 20.909 | 20.732 |
| 到港上轮挡 | 27.243 | 27.550 | 27.786 |

4B综合略好但关键分组更差且更慢，未替换现有1.7B实验主模型。兜底解决推理失败，不保证自动修正数值不准的输出。原项目8.58分钟与66%调度改善未有当前同口径复现，不作为本版本结果。

此前训练边界审计发现部分标签尚未完成，后续拟合需要过滤。只有五月数据、缺少部分接收时间且测试多次查看，不能宣称严格全历史无泄漏或跨月泛化已验证。


## 如何补验默认跳过项

默认回归将数据库、浏览器、部署和真实 GPU 专项设置为显式启用。跳过不计为通过。运行前完成 README 的构建与协议生成，使用独立虚拟环境。

### 本地 CPU 完整专项

原神经网络/随机森林回归另外需要 CPU PyTorch 和 scikit-learn；这些不是 TEST Web 的必需依赖。下面的版本已在 Python 3.10 独立测试环境使用，不应用 CPU PyTorch 替换 GPU Worker 环境。

```bash
.venv/bin/python -m pip install 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install 'scikit-learn==1.5.2'
.venv/bin/python -m playwright install chromium
FLIGHT_MYSQL_TESTS=1 FLIGHT_REDIS_TESTS=1 FLIGHT_HTTP_TESTS=1 \
FLIGHT_BROWSER_TESTS=1 FLIGHT_PUBLIC_SCREENSHOTS=1 \
  .venv/bin/pytest -q
```

还需 MySQL 8 的 mysqld、redis-server 和 Nginx 可执行文件。自定义安装路径应加入本次测试进程的 PATH；Nginx 也可通过 FLIGHT_NGINX_BINARY 指定。浏览器使用自定义安装目录时，设置对应 PLAYWRIGHT_BROWSERS_PATH。测试自行创建临时数据库、socket、高端口和进程，不连接运行中的业务数据库。

### GPU 专项

使用真实 GPU、兼容 PyTorch/Transformers/PEFT、本地 Qwen 基座和匹配 Adapter，显式配置 FLIGHT_GPU_TESTS、FLIGHT_GPU_MODEL_PATH、FLIGHT_GPU_ADAPTER_PATH，并启用 MySQL/Redis/HTTP 专项。GPU测试入口为 tests/integration/test_gpu_business_chain.py，验证临时 Nginx、双 Web、持久任务、真实模型推理、态势与恢复。

此测试默认写 runtime/server-gpu-production-acceptance；已有该目录时先归档，或通过测试运行器为该模块指定新的 OUT 路径。不要覆盖先前证据。需要确保 GPU 剩余显存足够加载独立模型；不能为测试停止其他业务 Worker。

GPU专项通过说明其工程链路与断言通过，不代表模型精度门槛通过。CPU和GPU跨环境执行时逐项核对测试标识，不能只把若干批次计数相加当作覆盖率。

## 2026-10-02 补验结果

修正恢复页面测试的过时文案断言后，独立 CPU 环境完整运行 743 passed、1 skipped（322.58 秒），唯一跳过为 test_real_gpu_nginx_mysql_redis_closed_loop。同一源码的该 GPU 用例在学校 A100 的独立测试栈运行 1 passed（86.69 秒）；已按测试标识核对全部 744 个 Python 用例实际执行，无未执行项。学校常驻 11 角色及 readiness 在专项后保持正常。

默认 CI 未开启这些可选环境，仍显示相应跳过；不应为了消除跳过提示而让每次普通检查强制加载模型。上述结果不是语句/分支覆盖率，也不是模型精度评测。
