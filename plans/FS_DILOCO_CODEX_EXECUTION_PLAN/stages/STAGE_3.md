# Stage 3：崩溃一致性与恢复

## 目标

证明共享 FS 上的 bounded authoritative state 支持 kill-restart，不需要历史 replay 或 exact learner recovery。

## 进入条件

Stage 2 通过。

## Loop 顺序

| Loop | 内容 | 9N overlay |
|---|---|---|
| S3-01 | learner restart、sequence monotonic、optimizer rebuild | kill 一个 learner，恢复后完成 10 cycles |
| S3-02 | syncer restart from current records | kill syncer，恢复后完成 |
| S3-03 | global publication interruption matrix | payload/visibility 前后 kill |
| S3-04 | quorum shortage、other-fragment progress、automatic resume | independent jobs 为主 |

每个 loop 需 8+1/50×10 门禁，但 `GU-S3-CRASH`（S3-02+S3-03）为预注册 gate 单元：
S3-02 以 2-node 缩小复现证据先关闭，S3-03 的一次 fault-tape run（含 syncer
kill+restart 与 publication kill 点）履行两个 loop 的 9N 义务（`AGENTS.md` §5.1）。
Stage 3 优先使用 independent jobs，使角色能单独 kill/restart；coallocated run 只能作为补充。

## 执行与成本约束

- restart 时间（≤5min）与恢复 interval（≤2× 正常 update interval）的起止事件定义、
  kill-point 矩阵与每点重复次数全部在提交前预注册；不得作业后定义口径。
- kill 场景先在进程内注入与 2-node 缩小复现通过后再消耗 9N；fault gate 首次失败
  立即保存证据并降级复现。
- Stage 3 全矩阵（10 次 syncer kill、learner ≤5min、publication matrix）的汇总口径
  在 S3-04 冻结，并显式引用各 loop 已有证据，不重复重跑。
- Checker 按 validator summary 与 analyzer JSON 优先复核；raw 仅在校验失败时展开。

## 验收覆盖

- A-REC-01..04

## 阶段级正式矩阵

- syncer kill -9 不同时机 10 次；
- learner restart ≤5 min；
- payload write/visibility replace 前后只出现合法旧/新 state；
- 少于 quorum 的 fragment 停、其他 fragment 继续；
- 恢复后自动续；
- loss 无非法跳变；
- 不扫描历史。

## Stage 关闭条件

- 全矩阵通过；
- restart sequence 不复用；
- current records 足以恢复 syncer；
- orphan/temp 只影响空间，不影响 correctness；
- 恢复 runtime 及 update interval 符合 source；
- Checker 允许关闭。
