# 08. 配置、身份与可复现性

## 8.1 配置层

推荐分层：

```text
base.yaml
model/<profile>.yaml
data/<profile>.yaml
storage/miyabi.yaml
stage/<stage>.yaml
loop/<loop-id>.yaml
run/<run-id>.yaml
```

启动时合并为一个 resolved config。只有 resolved config 与 digest 是 run authority；分层源文件仅便于维护。

## 8.2 禁止隐式默认的字段

以下字段必须在 resolved config 中出现：

- run/model/data/fragment identities；
- M/F/H/offsets；
- Q/Q_fresh/grace/S_max/lambda；
- local progress cap；
- inner/outer optimizers and schedules；
- dtypes/accumulation；
- snapshot/adoption/polling/backpressure；
- storage paths/backend/visibility delay；
- stop condition；
- seeds；
- logging/heartbeat；
- fault schedule；
- loss/runtime gate；
- Miyabi role layout。

## 8.3 Run identity

建议：

```text
<date>-<stage>-<loop>-<short-config-digest>-<short-commit>
```

identity 不依赖 mutable branch name。所有角色必须在启动前对 run identity 和 config digest 达成一致，否则 fail fast。

## 8.4 Asset manifests

模型：

- HF config JSON/revision/hash；
- tokenizer files/revision/hash；
- initial parameters hash；
- tied parameter map。

数据：

- source/revision；
- preprocessing code commit；
- tokenizer identity；
- shard list、size、token count、hash；
- learner-to-shard mapping。

fragment：

- logical layer sequence；
- parameter identities；
- byte counts；
- ownership；
- intervals；
- map digest。

## 8.5 版本兼容

protocol records 包含 schema version。规则：

- reader 明确接受的版本集合；
- 不支持时 fail，不做 best-effort 猜测；
- schema migration 通过独立 loop/ADR；
- 同一 run 不混用不兼容 code commits；
- restart 必须验证 current records 与代码兼容。

## 8.6 Seed

保存：

- run seed；
- model initialization seed；
- each learner seed；
- dataloader/shuffle seed；
- simulation seed；
- fault/latency injection seed；
- evaluation seed。

seed 不等于确定性保证；仍记录 CUDA/torch deterministic flags 和库版本。

## 8.7 Baseline compatibility key

runtime 对比只在以下关键字段一致时有效：

```text
Miyabi node/GPU class
queue/resource shape
model config + sequence length
batch/accumulation/precision
dataset/cache/local staging policy
fragment map/F/H
learner count/syncer count
core training implementation
```

不兼容时创建新 baseline，不强行比较。

## 8.8 冻结与解冻

- run 提交前 config 状态 `frozen`；
- runtime 不允许修改算法配置；
- 诊断 override 生成新 run ID；
- Stage 4 消融共享同一 binary/commit，只改允许的配置轴；
- 任何默认改变写 changelog/ADR 并更新 baseline。
