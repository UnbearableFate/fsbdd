# Stage 2：异步管线与性能加固

## 目标

让 GPU↔CPU↔FS transfer 与 inner training 重叠，保持 queue/state 有界，补齐 telemetry，并达到 source 性能阈值。

## 进入条件

`S1-13` 通过，`CAP-BASE-TRAINING` active。

## Loop 顺序与 9 节点要求

| Loop | 主要目标 | 9N overlay |
|---|---|---|
| S2-01 | async GPU→CPU→FS publish | 证明训练与上传重叠 |
| S2-02 | async FS→CPU→GPU adoption | 证明 safe-boundary 短暂停 |
| S2-03 | latest-wins backpressure、slow FS、GC | 1/10 FS 限速 overlay |
| S2-04 | TEL-01..03 与 latency decomposition | 完整结构化 metrics |
| S2-05 | goodput 优化与阶段关闭 | 95%/2% 等正式阈值 |

每个 loop 都需要 8+1、H=50、global_cycle=10 门禁，但 `GU-S2-ASYNC`（S2-01+S2-02）
为预注册 gate 单元：S2-01 以 2-node overlap 证据先关闭，S2-02 的一次单元 run 同时
断言 upload overlap 与 adoption pause，履行两个 loop 的 9N 义务（`AGENTS.md` §5.1）。
其余 loop 各自一次新 run；不得把全部门禁推迟到 S2-05。

## 执行与成本约束

- S2-03 的 ≥2h slow-FS run 若以 9 节点 8+1 拓扑、1/10 限速 overlay 完成 50×10 且提交
  前预注册双重用途，可同时充当该 loop 的 gate run。
- goodput/pause 的正式测量定义（GPU busy 时间来源、共同区间、聚合公式）在 S2-04
  冻结；S2-05 的 95%/2% 阈值只按该定义计算，不得合成非同步 rank-local rates。
- 0.5–1B 系统 workload identity（模型/数据/tokenizer/预处理 revision）在 S2-05 任何
  L4 run 前以 ADR 冻结。
- 性能优化逐项开关、每项一个 profile 对照；无 profile 证据不得批量重构。
- Checker 按 package validator summary 与 analyzer JSON 优先复核 gate；raw 仅在校验
  失败时展开。每 loop 一个 Checker context（Phase A/B）。

## 验收覆盖

- A-PERF-02
- A-PERF-04
- A-PERF-05
- A-PERF-06

同时保持 Stage 1 全部回归。

## 阶段级实验

- 160M no-communication vs FS communication；
- 0.5–1B 同样对照；
- 慢 FS 1/10、≥2h，有界 queue/temp；
- latency 分解；
- 每 loop 的 9N loss/runtime回归。

## Stage 关闭条件

- GPU goodput ≥95%；
- snapshot+adoption pause ≤2%；
- slow FS 2h 状态有界；
- telemetry 足以重算所有报告；
- 9N baseline 根据兼容性决定保留或生成性能版 baseline；
- Checker 允许关闭。
