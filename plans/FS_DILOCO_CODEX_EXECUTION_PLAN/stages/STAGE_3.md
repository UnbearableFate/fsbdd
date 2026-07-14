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

每个 loop 需新的 8+1/50×10 run。Stage 3 优先使用 independent jobs，使角色能单独 kill/restart；coallocated run 只能作为补充。

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
