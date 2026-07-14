# 13. Stage 0 复盘与 Stage 1 执行改进

本文件记录 Stage 0 已证实的结论、耗时和 token 成本的根因，以及从 Stage 1
开始必须执行的改进。它不改变 source spec 的算法语义。

## 13.1 Stage 0 做了什么

Stage 0 关闭了七个 loops 和八个 required acceptances：

- 建立 fresh/stale weighting、consumption frontier 与 outer optimizer 的可执行数学
  oracle；
- 建立有界、确定性的离散事件模拟器，并完成精确 7,776 行参数扫描和中断恢复；
- 在目标 Lustre 上验证跨节点可见性、100,000 次原子替换、并发 reader、metadata
  容量、16-stream I/O 与 bounded cleanup；
- 用 loop-level Checker 和 stage-level Checker 独立复算后关闭 Stage 0。

最终事实以 `reports/stage_checkpoints/STAGE_0.md` 和
`reports/checkers/STAGE_0.md` 为准。Stage 0 没有实现 learner/syncer，没有运行
Torch/HF 模型、GPU 训练或 loss，也没有建立 9 节点能力。

## 13.2 主要发现

- Stage 1 Profile A 冻结为 `M=8`, `Q=Q_fresh=M`, `S_max=0`, grace `0`,
  `lambda_s=1`, `F=4`, staggered `H=50`。
- Stage 4 的 stale anchor 只有约 `5.31%` accepted stale token rate，且相对 matched
  fresh control 的恢复为 `-0.0446` 个百分点；它是必须实跑的 ablation anchor，不是
  stale 有益的结论。
- 目标路径是跨节点共享 Lustre；distinct-host `/tmp` negative control 正确失败。
- 跨节点可见性最坏 p99 为 `7.950762ms`，远低于 `2.5s` 门槛；两个正式 run 在
  256/4096-byte record、每种 100,000 次替换、四个 readers 下均为零 violation。
  unsafe overwrite 产生 5,208 个 detector hits，说明 detector 不是静默失效。
- 修正后的同步 metadata 下界为 `39,403.447605/s`，高于 `5,120/s` 门槛；最坏
  16-stream write 为 `0.323807765s`，低于 `5s`。固定发现面、history/control 和
  cleanup 均保持有界。
- 可选 object-store `BENCH-05` 未运行，因此没有 S3/object-store 结论。

## 13.3 为什么花了很长时间

执行包准备到 Stage 0 checkpoint 的 Git 时间窗约 5 小时 43 分；以 Stage 0
operational baseline `cd4ae9d` 到关闭 `0585a2a` 计约 5 小时 24 分。完整 goal
还包含规格阅读、排队、日志检查、Checker 等待和最终审计，记录约 7 小时 04 分。
该范围有 74 个 commits，说明成本主要来自多轮短迭代和证据重建，而不是单次 benchmark
本身。

必要成本应保留：真实 PBS compute-node 验证、不可变证据、反例和独立 Checker
确实发现了真实错误。可避免成本主要有四类：

1. **Checker 介入过晚。** 多数语义缺陷在 GREEN/HARDEN 甚至完整集群证据产生后才被
   发现，导致代码、run package、checksum、索引和 Checker 全部重做。
2. **证据契约边做边补。** run identity、skill provenance、实际 role map、resume
   identity、checksum 完整性和 scheduler 时间来源没有在首个 runtime run 前冻结。
3. **测量定义不够早。** S0B-03 曾把不同 rank 的非同步一秒窗口相加，再除以最大
   local elapsed，得到不可接受的合成 rate；昂贵运行前缺少“权威观察点、共同区间、
   聚合公式”的审查。
4. **PBS 迭代粒度太小。** login 节点不能运行测试是必要限制，但多个 focused test
   被拆成独立 batch jobs，反复支付排队、环境启动、证据封装和复核成本。

历史 blocked Checker 报告还分别发现：RED 只在 import 阶段失败、requirement map
漏项、float 排序丢失大整数精度、隐藏容器无界增长、resume 未绑定 code/config/
generator、skill/manifest provenance 不实。这些都是把审查前移可以低成本发现的问题。

## 13.4 为什么 token 用量很大

token 成本与运行时间相关，但不是同一件事。主要放大器是：

- 七个 loops、多个 blocked→corrected cycles 和最终 stage audit 都反复加载大规格、
  loop 卡、diff、manifest、日志和 CSV 摘要；
- 每个 fresh Checker 为保证独立性重新构建上下文；若 subagent 继承 Maker 的完整聊天，
  还会重复支付无关上下文；
- 证据缺少统一 admissibility validator，Agent 多次用自然语言逐文件核对 qtime、角色、
  checksums 和 listed/actual 文件；
- 广泛 `sed`/`rg`、整份历史报告和大工具输出进入上下文，之后的每轮推理都继续携带；
- 关键决策虽已落盘，但没有一个面向下一 loop 的短 handoff，恢复工作时需要从历史材料
  重新归纳。

独立性和证据完整性不能通过少读来牺牲；正确的降 token 方法是把机械核对自动化、让
上下文按索引逐层展开，并减少重复 agent/session。

## 13.5 从 Stage 1 起的强制改进

### A. 先冻结语义和证据，再运行

ORIENT 后、RED 前必须形成 requirement→assertion→counterexample→authoritative
observation→aggregation→evidence 的完整矩阵。RED 必须执行目标 API 或明确的错误
surrogate；仅 import/collection failure 不构成语义 RED。

任何 L2/L3/L4 run 前必须通过 run-package preflight：clean commit、source/skill
identity、resolved config、run identity time source、scheduler identity、实际 roles、
paths、fail-if-exists、checksum finalization 和 listed/actual 一致性。S1-00 实现通用
builder/finalizer/validator，后续 loop 复用，不再手工拼装。

### B. Checker 前移但不增加 subagent

每个 loop 最多使用一个独立 Checker context。普通低成本 loop 只在关闭时调用；需要
L2/L3/L4 的 loop 让同一个 Checker 先做 Phase A 静态 precheck，Maker 修正后运行，
再由该 Checker 做 Phase B final check。修复 blocked candidate 时继续同一 Checker，
除非 Checker 独立性已被破坏；不为“再确认一次”另起多个 final Checker。

Checker subagent 使用无聊天继承的最小上下文，只收到 source/loop 路径、checked
commit、diff、evidence index 和 manifest。subagent 不用于普通检索、总结或可由主 Agent
串行完成的工作。

### C. 降低 PBS 和集群试错成本

遵守 static→targeted→1-node real→2-node behavior→9-node gate→long run 阶梯。
在同一 commit/config 的 focused 迭代中优先复用一个有效的 1-node interactive/debug
allocation；multi-node、长时或需持久证据的最终 run 才用 batch。昂贵 run 首次失败后
立即保存证据并降级复现，不连续提交同类集群试错。

### D. 降低上下文和 token 成本

- 首先读取 `PROGRESS.yaml`、当前 loop state、stage checkpoint、evidence index 和最后
  Checker；只有索引不足或校验失败时读取 raw evidence。
- 搜索先用窄 `rg`，再打开命中附近行；禁止为定位一个事实批量 dump 多份大文件。
- 每个阶段结束更新 loop state 的短 handoff：已证实事实、当前唯一 gap、checked
  commit、选择的 evidence、下一命令。不要依赖聊天历史。
- 一个执行 goal/session 默认只关闭一个 Maker loop；记录 token 起点、soft budget 和
  各阶段增量，PERSIST 后先交付 handoff，不把整个 Stage 继续堆进同一上下文。
- 工具输出默认限制到回答当前问题所需的行/字段；大 CSV/log 用程序生成计数、hash 和
  anomaly summary。
- 同一 session 中按 hash 记录已读且未变化的 authority，不重复全文读取；新 Checker
  仍按独立性要求读取自己的最小权威输入。

### E. Stage 1 成本闸门

- `S1-00` 只做 skeleton/config/identity/evidence tooling，不加载真实模型。
- `S1-01..S1-05` 先关闭结构、partition、storage 和 protocol 语义，再扩大真实训练。
- `S1-06..S1-12` 按 1-node 后 2-node 建立功能和数值 E2E。
- `S1-13` 的 9N gate 和 M=4/160M/≥1B-token long run 都必须在较小 smoke、package
  preflight 和 Checker Phase A 通过后提交；long run 不是调试环境。

## 13.6 预期效果

这些改进不减少 correctness gates；它们减少“发现问题时已经支付的成本”。预期直接
下降的是无效 PBS 次数、重复 evidence packages、重复 Checker 上下文、人工日志核对和
大输出 token。Stage 1 是否真正改善，应在每个 loop 记录 attempts、queue/active time、
Checker cycles、Agent token 起止/阶段增量、tool-output 摘要量和返工根因，并在 Stage 1
checkpoint 复盘。
