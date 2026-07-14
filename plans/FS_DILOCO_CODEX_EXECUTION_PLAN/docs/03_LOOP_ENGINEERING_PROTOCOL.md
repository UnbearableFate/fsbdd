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

若无法用一句可证伪陈述描述 gap，继续拆分，不进入 RED。

## 3.4 SPECIFY/RED

RED 证据必须：

1. 在缺失或刻意错误实现上失败；
2. 对应明确 requirement/acceptance；
3. 失败原因可读，不依赖偶然 timeout；
4. 尽可能小；
5. 保存命令、exit code、stdout/stderr、fixture/config、seed。

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
3. Miyabi login 节点静态检查；
4. 1-node compute targeted runtime；
5. 1-node real Torch/HF model/data 10-step smoke；
6. 2-node cross-node FS/launcher smoke（行为需要时）；
7. 9-node 8+1/50×10 final gate；
8. 阶段级长跑/多种子/故障矩阵。

前一级失败时不上一级。9 节点不是调试沙箱。

## 3.8 CHECK

Checker 使用新上下文，输入仅限：

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

## 3.10 失败与重试

- 第一次失败：保存原始证据，分类为 code/config/environment/resource/spec。
- 环境/资源失败不得伪装成代码失败或通过。
- 9 节点失败后用最小拓扑复现；只有新 commit/config/环境修复后才重跑 9 节点。
- 同一 loop 连续三次在相同根因失败，必须写 blocker/ADR 并重新 ORIENT。
- 不删除失败 run；以状态和哈希区分。
