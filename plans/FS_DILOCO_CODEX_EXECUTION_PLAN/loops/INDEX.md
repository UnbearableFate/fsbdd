# Loop 索引

基础系统完成点为 `S1-13`。从 S2-01 起，每个实现 loop 都有新的 9 节点门禁。

## Stage 0

| ID | 标题 | 前置 | Acceptance | 9N |
|---|---|---|---|---|
| [S0C-01](stage0/S0C-01.md) | Fresh/Stale 数学 oracle 与 weighting | — | A-ALG-00 | 否 |
| [S0C-02](stage0/S0C-02.md) | Consumption 语义、Outer Optimizer reference transitions 与证据映射 | S0C-01 | A-ALG-00 | 否 |
| [S0A-01](stage0/S0A-01.md) | 离散事件模拟器内核与 oracle 对齐 | S0C-02 | A-SIM-01 | 否 |
| [S0A-02](stage0/S0A-02.md) | 参数扫描、默认 Profile 与 stale 挽回预测 | S0A-01 | A-SIM-02, A-SIM-03 | 否 |
| [S0B-01](stage0/S0B-01.md) | Miyabi 与目标共享 FS 能力发现/Benchmark Harness | — | 组成证据 | 否 |
| [S0B-02](stage0/S0B-02.md) | 跨节点可见性与 10^5 次原子发布验证 | S0B-01 | A-BENCH-01, A-BENCH-02 | 否 |
| [S0B-03](stage0/S0B-03.md) | Metadata/Bandwidth 压测与 Stage 1 FS 决策 | S0B-02 | A-BENCH-03, A-BENCH-04 | 否 |

## Stage 1

| ID | 标题 | 前置 | Acceptance | 9N |
|---|---|---|---|---|
| [S1-00](stage1/S1-00.md) | 项目 Skeleton、Typed Config、Identity 与 Run Manifest | S0A-02, S0B-03, S0C-02 | 组成证据 | 否 |
| [S1-01](stage1/S1-01.md) | Hugging Face Model Logical-Layer Registry | S1-00 | A-FRAG-01 | 否 |
| [S1-02](stage1/S1-02.md) | Layer-aligned 连续均衡 Fragment Map | S1-01 | A-FRAG-01, A-FRAG-02, A-FRAG-03 | 否 |
| [S1-03](stage1/S1-03.md) | STOR-01 抽象与原子 Publication Primitive | S1-00, S0B-03 | A-PROP-01, A-GLOBAL-01 | 否 |
| [S1-04](stage1/S1-04.md) | Bootstrap、Per-fragment Current 与 Bounded Global State | S1-02, S1-03 | A-GLOBAL-02 | 否 |
| [S1-05](stage1/S1-05.md) | Proposal Schema、Latest-wins Discovery 与通用 Eligibility | S1-03, S1-04, S0C-02 | A-PROP-01, A-PROP-02, A-PROP-03 | 否 |
| [S1-06](stage1/S1-06.md) | Torch/HF Learner Inner Training Core | S1-00, S1-01 | 组成证据 | 否 |
| [S1-07](stage1/S1-07.md) | H/Offset Fragment Snapshot 与 Proposal Publish | S1-05, S1-06, S1-02 | 组成证据 | 否 |
| [S1-08](stage1/S1-08.md) | Latest-only Adoption、Mixed-version Model 与 Inner Moments | S1-04, S1-06, S1-03, S1-02 | A-LEARN-02, A-LEARN-03 | 否 |
| [S1-09](stage1/S1-09.md) | Syncer Readiness、Quorum、Grace 与 Fair Scheduling | S1-05, S1-04 | A-PROP-02, A-PROP-03 | 否 |
| [S1-10](stage1/S1-10.md) | Streaming Direct Merge 与 Fragment-wise Outer Optimizer | S1-09, S0C-02 | A-ALG-01, A-PERF-01 | 否 |
| [S1-11](stage1/S1-11.md) | Atomic Global Commit 与 Consumed Frontier | S1-03, S1-05, S1-10 | A-GLOBAL-01, A-GLOBAL-02, A-PROP-03 | 否 |
| [S1-12](stage1/S1-12.md) | Profile A 数值端到端与 Frozen Evaluation Snapshot | S1-07, S1-08, S1-09, S1-11 | A-ALG-01, A-EVAL-01, A-LEARN-01, A-LEARN-03 | 否 |
| [S1-13](stage1/S1-13.md) | 基础训练系统关闭与首个 Miyabi 8+1/50×10 Baseline | S1-12 | A-LEARN-01, A-PERF-01 | 强制 |

## Stage 2

| ID | 标题 | 前置 | Acceptance | 9N |
|---|---|---|---|---|
| [S2-01](stage2/S2-01.md) | 异步 GPU→CPU→FS Publication Pipeline | S1-13 | 组成证据 | 强制 |
| [S2-02](stage2/S2-02.md) | 异步 FS→CPU→GPU Adoption Pipeline | S2-01 | 组成证据 | 强制 |
| [S2-03](stage2/S2-03.md) | Bounded Backpressure、Slow FS 与 GC | S2-01, S2-02 | A-PERF-02 | 强制 |
| [S2-04](stage2/S2-04.md) | Telemetry、Latency Decomposition 与 Gate Analyzer | S2-03 | A-PERF-06 | 强制 |
| [S2-05](stage2/S2-05.md) | Goodput 优化与 Stage 2 关闭 | S2-04 | A-PERF-04, A-PERF-05, A-PERF-06 | 强制 |

## Stage 3

| ID | 标题 | 前置 | Acceptance | 9N |
|---|---|---|---|---|
| [S3-01](stage3/S3-01.md) | Learner Kill-Restart Recovery | S2-05 | A-REC-02 | 强制 |
| [S3-02](stage3/S3-02.md) | Syncer Kill-Restart from Per-fragment Authority | S3-01 | A-REC-01 | 强制 |
| [S3-03](stage3/S3-03.md) | Global Publication Crash Matrix | S3-02 | A-REC-03 | 强制 |
| [S3-04](stage3/S3-04.md) | Quorum Availability Boundary、Automatic Resume 与 Stage 3 关闭 | S3-03 | A-REC-04 | 强制 |

## Stage 4

| ID | 标题 | 前置 | Acceptance | 9N |
|---|---|---|---|---|
| [S4-01](stage4/S4-01.md) | `S_max=1` Base History Window 与 Bounded GC | S3-04 | 组成证据 | 强制 |
| [S4-02](stage4/S4-02.md) | Old-base Displacement 与 Staleness Boundary/Rejections | S4-01 | A-STALE-01, A-STALE-02, A-STALE-03, A-STALE-04 | 强制 |
| [S4-03](stage4/S4-03.md) | Fresh Anchor、Readiness 与 Deterministic Candidate Selection | S4-02 | A-STALE-06 | 强制 |
| [S4-04](stage4/S4-04.md) | Inverse-staleness Weighting、Same-base Consumption 与 Latest Reorder | S4-03 | A-STALE-05, A-PROP-04, A-PROP-05 | 强制 |
| [S4-05](stage4/S4-05.md) | Profile B 实跑、Simulator 对齐与 Stale Acceptance Gate | S4-04 | A-STALE-07 | 强制 |
| [S4-06](stage4/S4-06.md) | Matched-token H3、多种子与 Stage 4 关闭 | S4-05 | A-ALG-02 | 强制 |
