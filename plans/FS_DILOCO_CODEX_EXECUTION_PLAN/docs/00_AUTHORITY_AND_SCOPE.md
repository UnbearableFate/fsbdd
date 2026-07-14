# 00. 权威关系、范围与用户补充约束

## 0.0 安装路径契约

本包不是仓库根。执行时：

- `REPO_ROOT=$(git rev-parse --show-toplevel)`；
- `PLAN_ROOT=$REPO_ROOT/plans/FS_DILOCO_CODEX_EXECUTION_PLAN`；
- 本包控制文件相对 `PLAN_ROOT`；
- production source、tests、runtime configs、PBS scripts 和 evidence 相对
  `REPO_ROOT` 或 resolved 外部/shared 路径。

根目录 `AGENTS.md` 必须显式引入本包契约。若 checkout 存在无关 dirty state，正式
run 从包含本包与 source 的 committed tracked baseline 创建 clean worktree，不能把
无关修改混入 run commit。

## 0.1 规范层级

| 层级 | 文档 | 负责内容 |
|---|---|---|
| A | `source/STAGE0-4_SPEC.md` | Stage 0–4 的算法语义、系统不变量、协议、验收 ID |
| B | `source/RESEARCH_PLAN.md` | 研究问题、假设、阶段目标、方法学口径、长期实验 |
| C | `AGENTS.md` 与本执行包 | Codex 工作方式、工程拆分、Miyabi/Torch-HF/9 节点门禁 |
| D | loop 卡、配置与模板 | 对 A–C 的操作化，不得改写上层含义 |

本包对原规格的唯一新增硬约束是用户补充的执行约束：

1. 计划执行者为 Codex；
2. Codex 必须使用 [UnbearableFate/miyabi-development](https://github.com/UnbearableFate/miyabi-development) 操作 Miyabi；
3. 基础训练系统以 PyTorch + Hugging Face 生态构建；
4. 基础系统完成后，每个实现 loop 最终以真实 8 learner + 1 syncer 的 Miyabi 计算节点训练收尾，运行 `H=50` local optimizer steps 与 `target_global_cycles=10`，loss 正常下降，active runtime 合理。

这些约束不改变 `INV-*`、算法公式、proposal/global publication 契约或原验收阈值。

## 0.2 Stage 0–4 范围

包括：

- Stage 0-A 离散事件仿真；
- Stage 0-B Miyabi 目标共享存储微基准；
- Stage 0-C 数学 oracle；
- Stage 1 fresh-only、stale-ready 的最小端到端系统；
- Stage 2 异步传输、backpressure、telemetry 与性能；
- Stage 3 kill-restart 崩溃一致性；
- Stage 4 `S_max=1` stale-aware 扩展与 matched-token 对照。

不包括：

- Stage 5 外部网络基线、1000+ update 长跑、RDA/balanced-tensor 主实验；
- Stage 6 论文；
- Phase 2 多 syncer；
- TP/PP/FSDP/ZeRO、动态 membership、自动选主/failover、exact replay；
- 以网络通信替代共享存储数据面。

实现不得阻碍上述后续扩展。

## 0.3 基础系统完成点

`S1-13` 是 `CAP-BASE-TRAINING` 能力标记。它要求：

- Torch/HF 真实模型与数据可训练；
- fragment map、bootstrap、learner、proposal、syncer、merge、outer optimizer、global publication、adoption 已形成闭环；
- Profile A 数值正确；
- 首次 Miyabi 8+1/50×10 run 通过并形成 runtime baseline；
- 9 个不同 compute host 被证实；
- loss 门禁与运行时间基线已建立。

`S1-13` 之前不要求每个 loop 都申请 9 节点；之后每个实现 loop 都必须重新跑门禁，不能借用旧 run。

## 0.4 “50 local × 10 global”的规范化

为避免违反 `PROG-01..06`：

- local step = 一次完整 `optimizer.step()` 后的安全边界；gradient accumulation 的 micro-batch 不计为 local step；
- `H=50` = 每个 fragment 的计划发布周期；
- global step 在日志和配置中写为 `global_cycle`；
- `target_global_cycles=10` = 所有 fragment 的 `outer_update_count_f >= 10`；
- 每 learner 的实际 local steps/tokens 单独报告；异步模式下不要求相等；
- 不引入全 fragments 统一提交的 global head。

## 0.5 不能以门禁替代的原验收

9 节点 50×10 是 loop 级真实回归，不替代：

- Stage 1 的 M=4、160M、真实 FS、≥1B token run；
- Stage 2 的 160M 与 0.5–1B goodput 对照、2 小时慢 FS；
- Stage 3 的 10 次 syncer kill 与完整中断矩阵；
- Stage 4 的 ≥3 seeds、matched-token H3 检验；
- Stage 5 才收集的 A-PERF-03 1000+ updates 长跑证据。

## 0.6 变更规则

发现原规格与实现证据不兼容时：

1. 写 ADR；
2. 明确受影响的 requirement/acceptance IDs；
3. 保留失败证据；
4. 提出最小修订和迁移；
5. 用户批准后修改规范及变更记录；
6. 重新运行受影响 loops/acceptance。

不得通过代码默认值、隐藏环境变量或报告措辞悄悄改变研究口径。
