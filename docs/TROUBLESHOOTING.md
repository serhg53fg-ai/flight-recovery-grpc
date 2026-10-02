# 常见问题

| 现象 | 检查和处理 |
|---|---|
| CMake 找不到 Protobuf/gRPC | 安装 C++ 开发包与 `grpc_cpp_plugin`；Python 的 grpcio 不替代系统库 |
| 找不到 `generated` 协议模块 | 在仓库根目录执行 `bash scripts/generate_proto.sh`；确认 `.venv` 已安装 dev 依赖 |
| 测试找不到 Gateway | 构建 `build/phase2`，按教程准备 `build/phase2-release`；不要覆盖已有目录 |
| 启动等待或端口已占用 | 检查 `runtime` 中对应角色日志、监听地址与端口；停止旧项目进程后再启动 |
| `RESOURCE_EXHAUSTED` | Worker 或网关容量已满；先确认并发上限和显存，再调整容量 |
| gRPC 不可用或请求超时 | 检查 Health、节点身份、隧道和 deadline；不能通过无限重试掩盖节点故障 |
| Qwen 加载失败 | 核对模型文件完整性、CUDA/PyTorch、显存及 Adapter 与基座匹配关系 |
| 工件或回放身份不匹配 | 使用同一次发布生成的数据/工件清单；不要跳过哈希和特征契约校验 |
| 模型加载成功但输出不合法 | 核对 prompt/output schema 和 Adapter 训练方式；观察解析失败和兜底原因 |
| 恢复 API 未启用或返回 503 | 基础 TEST 启动器不启用持久恢复；按教程启动完整 Resident 栈 |
| Resident 数据库初始化失败 | 使用 MySQL 8，独立新数据目录；检查执行权限，避免复用其他服务数据目录 |
| Unix socket 路径过长 | 给 `runtime` 指定独立短绝对路径；三个 socket 路径须满足校验 |
| 浏览器测试缺少 Chromium | 安装 Playwright 浏览器和系统依赖；只有显式开启的浏览器测试才运行 |
| 默认测试有跳过项 | 查看 pytest 跳过原因；MySQL、浏览器和 GPU 项需独立环境，不能算作通过 |
| TEST 能预测但没有流量 | TEST 仅返回确定性四时刻，真实流量来自组合后端的独立流量模型 |

查看日志时先遮盖连接端点和凭据。截图中的 TEST/合成数据标签表示工程演示，不是部署故障，也不是模型准确率证据。
