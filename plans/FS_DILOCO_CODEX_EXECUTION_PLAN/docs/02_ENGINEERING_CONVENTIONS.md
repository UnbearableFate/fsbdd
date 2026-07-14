# 02. 工程约定

## 2.1 语言、依赖与风格

- Python 为主；shell 只负责环境、PBS 和轻量编排。
- Python 版本由 Miyabi 上 PyTorch/CUDA/HF 实测兼容性决定，写入 `.python-version`、`pyproject.toml` 与 `uv.lock`。
- 使用 `uv` 管理项目环境；执行位置按 `miyabi-development` skill 的 host routing。
- 推荐 `ruff format`、`ruff check`、`pytest`、`hypothesis`；类型检查选用项目已验证工具。
- 所有 public protocol dataclass/schema 有类型、版本和显式验证。
- 不捕获并吞掉异常；后台线程/进程错误必须传播到 role 状态与日志。

## 2.2 配置

- 算法与实验语义只来自冻结 resolved config，不从隐式环境变量读取。
- 环境变量只用于路径、PBS、cache、端口/launcher 控制面和密钥。
- `S_max`、`lambda_s`、`Q`、`Q_fresh`、grace、`H`、offsets、fragment count、model/data、dtype、seed、progress cap、stop condition 在启动前冻结。
- 启动时 schema 校验并打印 config digest。
- 所有角色验证相同 run/model/fragment-map/config identities；不匹配立即失败。

## 2.3 身份与完整性

每个 payload/record 至少携带：

- schema version；
- run identity；
- code commit；
- model config identity；
- fragment-map identity；
- role/learner/fragment identity；
- version 或 sequence；
- base version 与 base content identity；
- dtype、shape、byte size；
- payload content identity；
- local steps/tokens；
- creation timestamp 与 monotonic event sequence。

默认 content identity 为 SHA-256。序列化默认：

- 参数和纯 tensor state：`safetensors` 或等价非 pickle、可校验格式；
- metadata：canonical JSON；
- 非 tensor optimizer scalar metadata：JSON；
- 任何偏离写 ADR。

## 2.4 时间与日志

- manifest 中时间使用 UTC ISO-8601；
- 性能使用 monotonic clock；
- 每个角色独立写结构化 JSONL，避免多进程共享 append；
- 每条关键事件包含 `run_id`, `role`, `host`, `pid`, `event`, `fragment_id`, `version`, `local_step`, `global_cycle`（适用时）；
- stdout 供人工诊断，JSONL 才是分析输入；
- telemetry 丢失不能决定 state 是否 committed。

## 2.5 并发与后台工作

- 一次只允许一个正式 publication in flight / learner / fragment；
- 未开始的 pending snapshot latest-wins；
- 已开始 payload 不可变；
- adoption 只在完整 optimizer step 后；
- 后台 worker 使用有界 queue；
- shutdown 时有明确 drain/cancel 策略；
- 后台异常触发 role failure，不得永久 silent stall。

## 2.6 数值

- oracle 与 merge accumulation 使用 fp32，容差由测试预注册；
- 训练 dtype 可为 bf16/fp32，必须记录；
- pseudo-gradient 始终为 `G_base - L_local`；
- normalized weights 非负且和为 1；
- outer optimizer 作用于 current `G^v/O^v`；
- seed 分层：`run_seed`、`learner_seed = f(run_seed, learner_id)`、data shard seed、fault injection seed；
- 只有 oracle/profile A 需要严格可复现；性能 run 不强制全局 deterministic kernels，但必须记录相关 flags。

## 2.7 测试命名

```text
test_<requirement-or-acceptance-id>__<behavior>.py
```

例如：

- `test_inv_07__stale_uses_declared_base.py`
- `test_a_prop_03__repeated_polling_is_once_only.py`
- `test_a_rec_03__visibility_swap_crash_matrix.py`

测试输出写明对应 ID，便于自动生成 traceability。

## 2.8 禁止快捷方式

- 不违反 `miyabi-development` skill host routing 运行 runtime 工作；
- 不用完整模型 checkpoint 交换假装 fragment 协议；
- 不把 learner 进程组成 DDP world；
- 不用 telemetry/目录历史推断 authority；
- 不扫描历史修复 state；
- 不通过提高通信量掩盖 stale 对照；
- 不修改阈值以让失败 run 事后通过；
- 不把 synthetic/mock 门禁称为真实 8+1 训练。
