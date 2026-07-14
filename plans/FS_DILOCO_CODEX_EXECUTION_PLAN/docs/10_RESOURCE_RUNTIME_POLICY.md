# 10. 资源与运行时间政策

## 10.1 原则

- 用最小拓扑调试，用最终拓扑证明；
- 9 节点只在 RED/GREEN/HARDEN 和 1/2-node real smoke 通过后使用；
- 任何 run 提交前冻结目的、预计时间、节点数、停止条件和证据；
- queue wait 与 active runtime 分开；
- 不为“看看会怎样”申请长期多节点 job。
- requirement/measurement/package preflight 与 Checker Phase A 未通过时不提交 L2–L4。

## 10.2 预算层级

| 等级 | 典型资源 | 用途 |
|---|---|---|
| L0 | local CPU | oracle、schema、simulator |
| L1 | 1 compute node | Torch/HF、单角色、10-step |
| L2 | 2 compute nodes | cross-node FS、最小 E2E |
| L3 | 9 compute nodes | 每-loop 8+1/50×10 gate |
| L4 | stage budget | 1B tokens、2h slow FS、kill matrix、多种子 |

每个 loop 卡指定最高必要等级。

L1 focused 迭代优先在一个有效的 1-node interactive/debug allocation 内批量完成；不要
为每个小测试单独提交 batch。L2 以上、长时和需要 scheduler 作为正式证据的 run 使用
batch。login 节点仍然只做 control-plane/static 工作。

每个 loop state 预登记 `attempt_budget`。L2/L3/L4 第一次失败后停止同级重试，先在最低
可复现等级定位；只有 checked commit、resolved config 或环境修复发生可记录变化且低层
阶梯通过后，才消耗下一次昂贵 attempt。

## 10.3 9 节点消耗控制

基础系统后约有 15 个实现 loops。每个 loop 至少一次 9N gate。计划时记录：

```text
node_hours = 9 × active_runtime_hours
gpu_hours  = 8 × active_runtime_hours
```

按 1 次主 run + 1 次失败重试预留，不预留无界重试。若同一根因连续失败，降级拓扑排查。

## 10.4 Runtime baseline

`S1-13` 建立首个兼容 baseline。建议同时记录：

- isolated learner step time；
- no-communication learner throughput；
- 2-node E2E update interval；
- 9-node active runtime；
- per-fragment latency breakdown；
- FS read/write/metadata；
- startup/asset loading。

后续比较不只看总时长，定位变化来源。

## 10.5 合理时间判定

同时满足：

- pre-registered absolute budget；
- compatible baseline envelope；
- PBS walltime 余量；
- no unexplained stalls；
- Stage-specific performance threshold。

如果模型首次编译/缓存造成一次性慢启动，可单独报告 cold/warm，但不能选择性只引用较快结果。门禁 profile 应在基准定义中固定 cold 或 warm 条件。

## 10.6 排队与调度

- queue wait 不计性能，但影响执行计划；
- coallocated 9N 优先用于 loop gate；
- independent jobs用于场景真实性/恢复；
- job 不能同时开始时，roles 可以在 shared FS 上等待 bootstrap，但 wait 需单独统计；
- 不因 queue skew 放宽 protocol；
- 超出 walltime 的 run 视失败或不完整证据。

## 10.7 数据与存储预算

- Stage 0-B 先测实际 target FS；
- 估计 fragment payload、proposal slots、base window、temp、telemetry；
- 运行前检查 quota；
- GC 不在 correctness critical path；
- failure run 保存 metadata/trace，不必永久复制全部 tensor payload；
- assets、protocol state、evidence 分目录，避免误 GC。
