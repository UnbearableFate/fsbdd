# 05. Miyabi 运行手册

## 5.1 必须使用的 Codex skill

在涉及到代码运行,测试时使用skill miyabi-development

## 5.2 安装与确认

已经安装,直接使用

## 5.3 Host routing

login/compute 节点的分工、host 分类与在哪类节点上执行什么操作，一律以
`miyabi-development` skill 的 host routing 为准；本包不复制、不另行规定。

## 5.4 环境发现

在 skill 允许运行时检查的 allocation 中记录：

```bash
hostname
echo "$PBS_JOBID"
cat "$PBS_NODEFILE"
module list 2>&1
which python
python --version
nvidia-smi
```

Python/Torch/HF 版本检查（执行位置按 skill host routing）：

```bash
python - <<'PY'
import torch, transformers, datasets, tokenizers, safetensors
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("tokenizers", tokenizers.__version__)
print("safetensors", safetensors.__version__)
print("cuda_available", torch.cuda.is_available())
PY
```

结果写入 run manifest。项目 pin 与 skill 默认 Python 冲突时，以通过 Miyabi 兼容性验证的项目 pin 为准，并记录理由。

## 5.5 路径分类

每次 run 明确：

- `PROJECT_ROOT`：tracked checkout；
- `RUN_ROOT_BASE`：所有节点可见、用于创建独立 run root 的目标共享 FS；
- `RUN_ROOT=$RUN_ROOT_BASE/$RUN_ID`：单次 run 独占的协议 authority 根；
- `CACHE_ROOT`：HF/model/data cache；
- `LOCAL_STAGE_ROOT`：node-local 临时 staging，可丢失；
- `EVIDENCE_ROOT`：长期保留证据。

禁止把 protocol `current/latest/base/proposal` 放在 node-local scratch。Stage 0-B 必须对实际 `RUN_ROOT` 所在文件系统测量，而不是对另一个方便路径测量。

## 5.6 Git 同步

- 本地分支：`codex/<loop-id>-<slug>`；
- push tracked changes；
- Miyabi `fetch`, `switch`, `pull --ff-only`；
- 记录 commit；
- 不以 rsync 覆盖 tracked source；
- logs/大证据可以用受控 rsync/归档；
- 启动时检测并记录真实 integration branch；当前仓库为 `master`，不默认 merge。

## 5.7 验证拓扑

### 1 节点

用于：

- import/环境；
- HF model/data；
- oracle integration；
- learner/syncer单角色 smoke；
- 10-step 真实训练。

### 2 节点

用于：

- shared FS visibility；
- publication/read；
- launcher/role mapping；
- cross-node adoption；
- 最小 learner+syncer。

### 9 节点

用于：

- `S1-13` 基础系统；
- 之后每个 loop 的 final gate；
- 8 learners + 1 syncer、9 个 distinct compute hosts。

## 5.8 两种 9 节点启动模式

### Coallocated gate

一个 PBS job 请求 9 个节点，每节点一个 supervisor。rank 0 启动 syncer，rank 1–8 启动 learners。MPI/PBS 只负责进程启动和退出；应用不得使用 MPI/torch.distributed 数据通信。

优点：同时获得节点，减少 queue skew，适合每-loop gate。

### Independent jobs

一个 bootstrap job、一个 syncer job 与八个 learner jobs，共享唯一 run identity 与
独占 `RUN_ROOT`。bootstrap 成功后才能启动 roles；每个 role 必须用共享 FS 上的原子
host claim 拒绝物理 host 重复。适合：

- Stage 1 至少一次场景真实性验证；
- Stage 3 kill/restart；
- 需要证明独立作业生命周期的实验。

两种模式都必须验证 9 个 distinct host。independent 模式的 PBS resource 必须请求
每 learner 1 GPU，并使用当前 Miyabi 已验证的 exclusive-host placement；不得仅依赖
作业数量推断 host 数。阶段计划规定何时必须 independent 模式。

## 5.9 作业前检查

- queue/group/node shape 从 skill 和当前项目发现，不硬编码；
- PBS 脚本 `bash -n`；
- module 在 job shell 内重新 load；
- `PROJECT_ROOT`, `RUN_ROOT`, `CACHE_ROOT` 存在且权限正确；
- assets 可离线读取；
- config digest 与 code commit 固定；
- role map 预生成；
- walltime 和 runtime budget 冻结；
- cleanup/trap 不删除失败证据；
- stdout/stderr 指向 timestamped evidence dir。

## 5.10 作业后检查

- `qstat -f`/accounting；
- 9 个 host 与角色；
- exit codes；
- global cycle/version counts；
- loss/runtime gate；
- orphan/temp/GC；
- bytes 和 latency；
- checksums；
- Checker 输入完整。

队列等待时间不计 active runtime，但必须单独报告。
