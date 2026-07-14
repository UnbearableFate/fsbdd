# 执行包变更记录

## v1.0.2 — 2026-07-14

- 将 9 节点 8 learners + 1 syncer、H=50、global-cycle=10 的固定回归 workload 明确为 pretrained GPT-2 small + WikiText-2 raw。
- 冻结 Hugging Face model/tokenizer/dataset immutable revisions、GPT-2 架构身份、WikiText-2 packed-512 预处理和 8-way shard 规则。
- 冻结 batch、AdamW、bf16/fp32、RNG、loss warmup/min-points 与 validation 证据口径，并扩展 run manifest/checker report。
- 明确该 9N workload 不替代 source spec 的 160M/M=4/≥1B-token long run、性能与 matched-token 质量验收。
- 保留现有 Python/Torch 环境不变；仍未处理 Stage 0 前的环境 bootstrap。

## v1.0.1 — 2026-07-14

- 固定 `REPO_ROOT`/`PLAN_ROOT` 路径契约，并为当前 `master` integration branch 建立显式保护。
- 明确 tracked plan baseline、clean worktree、loop state 与 raw/tracked evidence 边界。
- 使 `S1-13 PASS` 与完整 Stage 1 closure（含 M=4/1B-token long run）等价。
- 修复 9 节点 run ID/目录覆盖风险；independent 模式增加 compute-node bootstrap、依赖提交与 distinct-host claim。
- 保留现有 Python/Torch 环境不变；未处理 Stage 0 前的环境 bootstrap。

## v1.0 — 2026-07-14

- 基于 `RESEARCH_PLAN.md` v1.2-draft 与 `STAGE0-4_SPEC.md` v2.0-draft 生成 Stage 0–4 Codex 执行计划。
- 加入用户补充约束：执行者为 Codex；Miyabi 操作必须使用 `UnbearableFate/miyabi-development`；训练系统基于 PyTorch + Hugging Face。
- 定义基础系统完成点 `S1-13`。
- 定义其后每个实现 loop 的真实 Miyabi 9 节点门禁：8 learners + 1 syncer、`H=50`、`target_global_cycles=10`、loss 正常下降、运行时间在冻结预算内。
- 加入阶段计划、36 个 loop 卡、追踪表、配置/证据/Checker/PBS 模板。
