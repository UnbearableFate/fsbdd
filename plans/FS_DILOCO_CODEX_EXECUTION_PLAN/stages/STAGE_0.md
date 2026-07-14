# Stage 0：预研验证

## 目标

在写端到端系统前关闭三个最高风险：

1. 事件/接受语义与参数选型；
2. Miyabi 目标共享 FS 的可见性、原子性、元数据与带宽；
3. fresh/stale/outer optimizer 的数学 oracle。

Stage 0 不实现训练系统，不触发 9 节点训练门禁。

## 进入条件

无。Miyabi 访问不可用时可先做 0-A/0-C，但 Stage 1 不得在 0-B 通过或形成明确缓解方案前开始。

## 默认执行顺序

| Loop | 内容 | 最高资源 |
|---|---|---|
| S0C-01 | fresh/stale pseudo-gradient 与 weighting oracle | local CPU |
| S0C-02 | consumption/reference transitions/evidence map | local CPU |
| S0A-01 | 离散事件内核与 oracle 共享决策向量 | local CPU |
| S0A-02 | 全矩阵扫描、默认 profile、预测报告 | CPU/batch |
| S0B-01 | Miyabi/FS capability discovery 与 benchmark harness | 1–2 nodes |
| S0B-02 | visibility + 10^5 atomic publication | ≥2 nodes |
| S0B-03 | metadata + bandwidth + threshold/mitigation decision | 2–16 nodes |

0-B 可与 0-A/0-C 并行。

## 关键交付

- 可复用 oracle package；
- simulator 与参数扫描报告；
- 默认 `Q/grace/S_max/lambda/H/offset` 建议；
- Miyabi target FS 一手数据；
- Stage 1 storage capability decision；
- requirement-to-evidence map 初版。

## 验收覆盖

- A-SIM-01..03
- A-BENCH-01..04
- A-ALG-00

阈值严格沿用 source spec。

## Stage 关闭条件

- 全部 7 个 loop 通过；
- Stage 0 acceptance 全部有原始证据；
- 0-B 不达标时有经 Checker 接受的缓解方案和新的 Stage 1 profile；
- simulator 的默认建议已冻结，但明确不用于 runtime correctness；
- Checker 出具 Stage 0 `PASS` 或允许关闭的 `PASS_WITH_FOLLOWUPS`；
- 产出 `templates/STAGE_CHECKPOINT.md` 的实例。

## 不得提前做

- learner/syncer生产代码；
- stale 质量结论；
- 以本地/NFS 结果代替 Miyabi target FS；
- 因 simulator 预测低而取消 Stage 4；
- 在 login 节点运行 benchmark。
