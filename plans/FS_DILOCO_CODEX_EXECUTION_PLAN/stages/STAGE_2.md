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

每个 loop 都需要新的 8+1、H=50、global_cycle=10 run；不得只在 S2-05 跑一次。

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
