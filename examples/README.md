# 合成航班演示

这些文件由本项目人工构造，不包含真实航班记录、天气记录或模型训练数据。时间为 2030 年，机场代码用于说明 ZGGG 业务范围。

| 文件 | 用途 |
|---|---|
| `synthetic-single.json` | 单航班请求 |
| `synthetic-batch.json` | 四航班、两条机尾串飞链 |
| `expected-test-times.json` | TEST 后端的确定性参考输出 |
| `recovery-scenario.json` | ZGGG 的 15 分钟容量和最小过站时间场景 |
| `generate_spreadsheet.py` | 从批量 JSON 生成 Excel，不提交生成文件 |

## 预测

先按 [部署教程](../docs/DEPLOYMENT.md)启动基础 TEST 链路，在仓库根目录执行：

```bash
curl --fail-with-body -sS http://127.0.0.1:5000/predict_flight \
  -H 'Content-Type: application/json' --data-binary @examples/synthetic-single.json
curl --fail-with-body -sS http://127.0.0.1:5000/predict_flights_batch \
  -H 'Content-Type: application/json' --data-binary @examples/synthetic-batch.json
.venv/bin/python examples/generate_spreadsheet.py
```

也可把 JSON 粘贴到 `/flight_input` 的 JSON 输入页，或上传 `runtime/demo/synthetic-flights.xlsx`。输入只含计划和可用气象字段，不携带实际起降标签。

TEST 将计划离港时间加 10 分钟作为推出时间、加 20 分钟作为起飞时间；正常示例的落地时间等于计划到港时间，上轮挡时间再加 10 分钟。它验证工程协议，**不会调用 Qwen，也不输出真实流量预测**。切换真实模型后，不应再使用 TEST 参考时刻验收精度。

## 态势与恢复

基础启动器不启用完整持久恢复 API。按部署教程启动 Resident 持久栈后，用同一批合成航班提交 `/api/v1/prediction-jobs`，待任务进入 `SUCCEEDED`，把其 `job_id` 与 `recovery-scenario.json` 组合为恢复请求。

恢复容量按 ZGGG 的推出/上轮挡时刻计入 15 分钟时槽。示例刻意让两次推出及两次上轮挡分别落入相同槽，便于观察容量冲突与延后调整。查看响应中的 `validation_errors` 和 `summary`，不要把生成成功等同于求得全局最优。

## 本地验证与截图

```bash
.venv/bin/pytest -q tests/integration/test_public_examples.py
# 另需 MySQL 8 与 Chromium；使用测试自建的临时数据库。
.venv/bin/playwright install chromium
FLIGHT_MYSQL_TESTS=1 FLIGHT_PUBLIC_SCREENSHOTS=1 \
  .venv/bin/pytest -q tests/browser/test_public_screenshots.py
```

截图保存在 `images/`，带合成数据和 TEST 水印。测试不会使用服务器上的运行数据库。测试构建路径与可选依赖见 [验证说明](../docs/VALIDATION.md)。
