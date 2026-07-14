# 执行顺序与并行规则

## 规范 DAG

Stage 0 的 0-A/0-B/0-C 可并行。下面是单一 Codex 会话的默认串行顺序；资源等待时只能切换到依赖已满足的 ready loop，并在 `PROGRESS.yaml` 记录。

```text
01. S0C-01  Fresh/Stale 数学 oracle 与 weighting
02. S0C-02  Consumption 语义、Outer Optimizer reference transitions 与证据映射
03. S0A-01  离散事件模拟器内核与 oracle 对齐
04. S0A-02  参数扫描、默认 Profile 与 stale 挽回预测
05. S0B-01  Miyabi 与目标共享 FS 能力发现/Benchmark Harness
06. S0B-02  跨节点可见性与 10^5 次原子发布验证
07. S0B-03  Metadata/Bandwidth 压测与 Stage 1 FS 决策
08. S1-00  项目 Skeleton、Typed Config、Identity 与 Run Manifest
09. S1-01  Hugging Face Model Logical-Layer Registry
10. S1-02  Layer-aligned 连续均衡 Fragment Map
11. S1-03  STOR-01 抽象与原子 Publication Primitive
12. S1-04  Bootstrap、Per-fragment Current 与 Bounded Global State
13. S1-05  Proposal Schema、Latest-wins Discovery 与通用 Eligibility
14. S1-06  Torch/HF Learner Inner Training Core
15. S1-07  H/Offset Fragment Snapshot 与 Proposal Publish
16. S1-08  Latest-only Adoption、Mixed-version Model 与 Inner Moments
17. S1-09  Syncer Readiness、Quorum、Grace 与 Fair Scheduling
18. S1-10  Streaming Direct Merge 与 Fragment-wise Outer Optimizer
19. S1-11  Atomic Global Commit 与 Consumed Frontier
20. S1-12  Profile A 数值端到端与 Frozen Evaluation Snapshot
21. S1-13  基础训练系统关闭与首个 Miyabi 8+1/50×10 Baseline
22. S2-01  异步 GPU→CPU→FS Publication Pipeline
23. S2-02  异步 FS→CPU→GPU Adoption Pipeline
24. S2-03  Bounded Backpressure、Slow FS 与 GC
25. S2-04  Telemetry、Latency Decomposition 与 Gate Analyzer
26. S2-05  Goodput 优化与 Stage 2 关闭
27. S3-01  Learner Kill-Restart Recovery
28. S3-02  Syncer Kill-Restart from Per-fragment Authority
29. S3-03  Global Publication Crash Matrix
30. S3-04  Quorum Availability Boundary、Automatic Resume 与 Stage 3 关闭
31. S4-01  `S_max=1` Base History Window 与 Bounded GC
32. S4-02  Old-base Displacement 与 Staleness Boundary/Rejections
33. S4-03  Fresh Anchor、Readiness 与 Deterministic Candidate Selection
34. S4-04  Inverse-staleness Weighting、Same-base Consumption 与 Latest Reorder
35. S4-05  Profile B 实跑、Simulator 对齐与 Stale Acceptance Gate
36. S4-06  Matched-token H3、多种子与 Stage 4 关闭
```

## Ready 判定

一个 loop ready 当且仅当：

- 所有 `depends_on` 为 PASS 或允许关闭的 PASS_WITH_FOLLOWUPS；
- source spec 未发生未处理变化；
- 没有影响该 loop 的 blocker；
- 所需 Miyabi/data/model/storage 资源可用，或该 loop 不需要它们；
- 当前仅一个 Maker loop 处于 GREEN/HARDEN，避免交叉污染。

## 并行允许项

- Stage 0-B 可与 0-A/0-C 并行，由不同分支/证据目录执行。
- Stage 1 的 S1-01/02 与 S1-03 可在 S1-00 后并行，但集成前各自 Checker 必须通过。
- 后续默认串行，因为每个 loop 都要基于前一 commit 跑新的9N门禁。

## 禁止越级

- 0-B 未通过/未形成缓解，不进入 Stage1 E2E。
- S1-13 未通过，不启用后续9N基准比较。
- 任何 loop 的低层验证失败，不直接提交9节点。
- Stage3未关闭，不开始Stage4正式 stale实验。

## Stage 1 成本里程碑

Stage 1 不新增 loop，但按以下里程碑升级资源：

1. `S1-00` 先交付 typed config、identity 和通用 run-package
   builder/finalizer/validator；后续正式 run 不再各自拼装证据契约。
2. `S1-01..S1-05` 先关闭 model/fragment/storage/protocol 的结构与语义反例；没有必要
   不上 L2。
3. `S1-06..S1-12` 先通过 targeted/1-node real，再做 2-node functional/numerical E2E。
4. `S1-13` 只有在 package preflight、Checker Phase A、1-node 和 2-node smoke 全通过后
   才提交 9N gate；M=4/160M/≥1B-token long run 还必须在短跑 loss/runtime/storage
   证据通过后提交。

每个里程碑从 `PROGRESS.yaml`、loop state 和 evidence index 恢复；不要求重读全部历史
Checker 或 raw logs。
