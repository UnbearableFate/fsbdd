# 03. Loop Engineering 执行协议

## 3.1 Loop 定义

loop 是一个能够由单一主要失败事实定义、由单一最小实现差距修复、并产生可检查证据的工作单元。一个 loop 不同时改变：

- fragment 语义；
- staleness policy；
- merge policy；
- outer optimizer；
- publication 模型；
- 拓扑；
- 数据/模型规模；
- 性能优化策略。

需要同时变化时拆成多个 loop，并以配置开关保持可归因。

一个执行 goal/session 默认只负责一个 Maker loop。该 loop PERSIST 后先交付 compact
handoff 并结束；除非用户明确要求连续执行，不把下一 loop 继续累积到同一聊天上下文。

## 3.2 状态机

```text
NOT_STARTED
  -> ORIENTED
  -> RED
  -> GREEN_LOCAL
  -> HARDENED
  -> MIYABI_PENDING       # S1-13 及以后
  -> CHECK_PENDING
  -> PASS | PASS_WITH_FOLLOWUPS | BLOCKED
```

只有 `PASS` 或允许关闭的 `PASS_WITH_FOLLOWUPS` 才能解锁依赖 loop。

## 3.3 ORIENT

必须持久化：

- 当前 loop ID、目标与非目标；
- source 规格哈希和版本；
- base branch/commit、工作树状态；
- 前置 loop 与 acceptance 状态；
- 当前环境、可用 Miyabi 资源、model/data/cache；
- 最近失败证据；
- 本 loop 唯一 failing gap；
- 预计修改文件；
- 运行与节点预算；
- 是否触发 9 节点门禁。
- context input plan：先读哪些 index/state，哪些 raw evidence 仅在异常时展开；
- runtime attempt budget：每级最多几次、失败后降到哪个最小复现。
- goal token 起点、按 scope 设定的 soft token budget 和阶段采样点。

若无法用一句可证伪陈述描述 gap，继续拆分，不进入 RED。

ORIENT 后还必须生成 requirement-to-evidence matrix。它对本 loop 的每个 requirement
和 acceptance 记录 assertion、至少一个反例、目标 API、权威观察点、measurement
interval/aggregation、证据文件和关闭条件。完整性集合必须来自 loop 卡/source
traceability，不得让测试和被测 CSV 共享一个手写漏项列表。

## 3.4 SPECIFY/RED

RED 证据必须：

1. 在缺失或刻意错误实现上失败；
2. 对应明确 requirement/acceptance；
3. 失败原因可读，不依赖偶然 timeout；
4. 尽可能小；
5. 保存命令、exit code、stdout/stderr、fixture/config、seed。
6. 到达目标 API/assertion；import/collection failure 只能算环境或 scaffolding RED，
   不能代替算法、协议或测量语义 RED。

常见 RED：

- golden transition；
- property test；
- partial publication reader；
- duplicate consumption；
- delayed latest regression；
- wrong-base stale counterexample；
- queue-bound invariant；
- timing trace；
- kill point matrix；
- loss/runtime gate parser fixture。

先有 RED，后写生产实现。纯文档/基础 scaffolding loop 也要有静态 schema/checker RED。

## 3.5 IMPLEMENT/GREEN

- 只修当前 RED；
- 先选择最简单正确实现；
- 不做未证实的性能重构；
- 保留 reference path 与 production path 可比较；
- 新默认必须在 config 中显式；
- 代码注释解释不变量和边界，不复述语法；
- GREEN 后立即保存最小通过证据。

## 3.6 HARDEN

按 loop 卡选择：

- 边界矩阵；
- property/fuzz；
- 并发与重排；
- slow/fault injection；
- 长时间有界性；
- model naming/tied weights；
- dtype/shape/corruption；
- 1-node real model；
- 2-node storage/distributed launcher；
- performance profile。

HARDEN 不扩大算法范围，只证明当前实现不脆弱。

## 3.7 Miyabi 验证阶梯

每次只升一级：

1. 静态检查与纯 Python 无重依赖检查；
2. 本地 unit/oracle/property；
3. Miyabi 侧静态检查（执行位置按 `miyabi-development` skill 的 host routing）；
4. 1-node compute targeted runtime；
5. 1-node real Torch/HF model/data 10-step smoke；
6. 2-node cross-node FS/launcher smoke（行为需要时）；
7. 9-node 8+1/50×10 final gate；
8. 阶段级长跑/多种子/故障矩阵。

前一级失败时不上一级。9 节点不是调试沙箱。

同一 commit/config 的多次 focused 检查优先复用一个有效的 1-node
interactive/debug allocation，减少排队、环境启动和 evidence wrapper 重复成本。
multi-node、长时、需要 durable scheduler evidence 的正式 run 使用 batch。任何 L2/L3/L4
run 首次失败后必须保存原始包并降级复现；只有 code/config/environment 有可识别变化
且低层验证通过后才重跑。

昂贵 run 前必须通过两项 admissibility：

1. frozen package validator：clean commit、source/skill/config identity、scheduler/run
   identity、实际 role map、路径、fail-if-exists、manifest 字段和 checksum lifecycle；
2. Checker Phase A：只审查 requirement completeness、measurement semantics、反例和
   package contract，不等待昂贵结果。

## 3.8 CHECK

每个 loop 默认最多一个 Checker context。低成本 loop 直接执行 Phase B final check；
L2/L3/L4 loop 的同一个 Checker 先执行 Phase A precheck，运行后再执行 Phase B。blocked
修复通过 follow-up 继续，不为重复确认新建多个 Checker。

Checker 使用新且不继承 Maker 聊天的上下文，输入仅限：

- source 规范；
- 当前 loop 卡；
- diff/commit；
- RED/GREEN/HARDEN 原始证据；
- 适用时 9 节点 run package；
- acceptance traceability。

Checker 必须：

- 从不变量反推至少一个 Maker 未列出的反例；
- 确认 RED 在错误实现上失败；
- 对照 reference math；
- 检查无界历史/队列；
- 检查数据面无网络捷径；
- 检查运行拓扑和节点真实性；
- 检查 loss/runtime gate 未事后改阈值；
- 输出 `PASS`、`PASS_WITH_FOLLOWUPS` 或 `BLOCKED`。

Phase A 可以输出 `ADMISSIBLE` 或 `BLOCKED_PRECHECK`，不能关闭 loop；Phase B 才能输出
最终 verdict。Checker subagent 不承担普通检索、总结或实现工作。

`PASS_WITH_FOLLOWUPS` 只有在 follow-up 不影响当前 requirement/acceptance 正确性且已登记后才能关闭。

## 3.9 PERSIST

每个 loop 产生：

```text
$EVIDENCE_ROOT/<loop-id>/<run-or-check-id>/
  manifest.yaml
  commands.log
  stdout/
  stderr/
  tests/
  configs/
  env/
  runtime/
  analysis/
  checker/
  checksums.sha256
```

并更新：

- `$PLAN_ROOT/plans/PROGRESS.yaml`；
- `$PLAN_ROOT/plans/ACCEPTANCE_TRACEABILITY.csv`；
- `$PLAN_ROOT/plans/loop_states/<loop-id>.yaml`；
- `$REPO_ROOT/evidence/indexes/<loop-id>.md`，其中记录 raw
  `$EVIDENCE_ROOT` 位置与校验和；
- ADR（如有）；
- 阶段 checkpoint（阶段关闭时）；
- 下一最小 failing gap。

聊天总结不能替代 PERSIST。

raw evidence 不通过 Git 在本地与 Miyabi 间同步。Git 只跟踪 index、manifest 摘要、
配置和足以复核结论的小型证据；大日志、metrics、payload 与 storage snapshot 保留在
resolved `EVIDENCE_ROOT`。

PERSIST 前必须运行 evidence finalizer/validator，并记录 validator 版本、命令、结果和
manifest hash。validator 至少拒绝 placeholder、未列/多列文件、checksum mismatch、
不一致的 code/config/source/skill/scheduler/role identity 以及不安全的目录复用。

## 3.10 失败与重试

- 第一次失败：保存原始证据，分类为 code/config/environment/resource/spec。
- 环境/资源失败不得伪装成代码失败或通过。
- 9 节点失败后用最小拓扑复现；只有新 commit/config/环境修复后才重跑 9 节点。
- 同一 loop 连续三次在相同根因失败，必须写 blocker/ADR 并重新 ORIENT。
- 不删除失败 run；以状态和哈希区分。

## 3.11 上下文与 token 控制

- 每次恢复先读 `PROGRESS.yaml`、当前 loop state、stage checkpoint、evidence index 和
  最新 Checker；只有索引不充分或 hash/claim 异常才打开 raw evidence。
- 用窄 `rg` 定位后读取命中范围；大 CSV/log 用确定性程序提取 cardinality、hash、
  extrema 和 anomaly，不把整份内容放入上下文。
- loop state 的 handoff 必须包含已证实事实、唯一 gap、checked commit、selected
  evidence、attempt/cost 统计和下一命令，聊天历史不是恢复依赖。
- token soft budget 只用于发现上下文膨胀；超出时先停止批量读取、审计输入并压缩
  handoff，不能跳过测试、Checker 或证据来“省 token”。
- 同一 session 中 hash 未变化的 authority 不重复全文读取。工具调用应设置与问题相称
  的输出上限。
- subagent 仅用于独立 Checker 或用户明确要求的并行工作；启动时不继承 Maker 聊天，
  只传最小文件/commit 清单。
