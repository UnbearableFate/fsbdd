# Stage 4：Stale-aware `S_max=1`

## 目标

在同一 binary 中把 base window 从 fresh-only 打开到 `S_max=1`，正确重建旧 base displacement，提供 fresh anchor、确定性选择、inverse-staleness weighting 和 matched-token 对照。

## 进入条件

Stage 3 通过；Stage 0-A 预测只作参数/验收锚点，不能取消本阶段。

## Loop 顺序

| Loop | 内容 | 9N overlay |
|---|---|---|
| S4-01 | base history window 与 bounded GC | `S_max=1` 保留/回收 |
| S4-02 | old-base displacement、boundary/rejection | 实际生成 s=1 |
| S4-03 | Q_fresh/readiness/deterministic selection | mixed fresh/stale quorum |
| S4-04 | weighting、same-base consumption、latest reorder | telemetry/weight mass |
| S4-05 | Profile B 与 simulator 接受率对齐 | stale accepted >5% |
| S4-06 | matched-token H3、多种子、阶段关闭 | 8+1 smoke + research runs |

每个 loop 都需要新的 8+1/50×10 run。S4-01 至少要证明 window 行为；S4-02 起必须预注册异构，使 stale proposal 实际出现。

## 验收覆盖

- A-STALE-01..07
- A-PROP-04..05
- A-ALG-02

## 消融

同一 commit/binary，只改配置：

- `S_max: 0,1`
- `lambda_s: 0.5,1,2`
- `Q_fresh: 1,0(experimental)`
- heterogeneity: `1.0,1.5,2,3`

质量比较 matched-token；效率 matched-compute 或 matched-communication。

## Stage 关闭条件

- old-base formula 与 oracle 一致；
- boundary/rejection/identity/weight/consumption 全过；
- Profile B stale accepted >5%，与 simulator 偏差 ≤2×；
- ≥3 seeds 的 matched-token H3 报告；
- 质量失败时默认回退 `S_max=0`，但机制和负面结果保留；
- Checker 允许关闭。
