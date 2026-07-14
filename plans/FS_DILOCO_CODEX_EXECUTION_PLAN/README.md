# FS-Based Decoupled DiLoCo：Codex Stage 0–4 执行包

本包把 `source/RESEARCH_PLAN.md` 与 `source/STAGE0-4_SPEC.md` 转化为可由 Codex 顺序执行、检查和留证的工程计划。它不替代原规格；原规格中的算法语义、系统不变量和验收 ID 仍是 Stage 0–4 的规范性依据。

## 使用入口

1. 固定仓库根 `REPO_ROOT` 与本执行包根
   `PLAN_ROOT=$REPO_ROOT/plans/FS_DILOCO_CODEX_EXECUTION_PLAN`。
2. Codex 先读取 `$REPO_ROOT/AGENTS.md` 和 `$PLAN_ROOT/AGENTS.md`。
3. 再读取 `$PLAN_ROOT/CODEX_START_HERE.md`、
   `$PLAN_ROOT/plans/EXECUTION_ORDER.md` 与当前 loop 文件。
4. 生产代码、测试、runtime configs、PBS 脚本和 raw evidence 写到
   `REPO_ROOT` 或 resolved 外部路径；不得写进 `PLAN_ROOT`。
5. 按 `ORIENT → SPECIFY/RED → IMPLEMENT/GREEN → HARDEN → CHECK → PERSIST` 执行。
6. 所有 Miyabi 操作必须使用并遵守 [UnbearableFate/miyabi-development](https://github.com/UnbearableFate/miyabi-development)。
7. 基础训练系统在 `S1-13` 完成后，后续每个实现 loop 都必须以一次新的真实 Miyabi 9 节点门禁结束：8 learners + 1 syncer、每 learner 1 GPU/1 node、pretrained GPT-2 small + WikiText-2 raw packed-512、`H=50` local optimizer steps、`target_global_cycles=10`，loss 有限且呈下降趋势，运行时间在预先冻结的预算内。
8. loop 通过后更新 `$PLAN_ROOT/plans/PROGRESS.yaml`、tracked evidence
   index 和 Checker 报告；raw evidence 不提交 Git，也不得只在聊天上下文中宣称完成。
9. Stage 1 起执行 `docs/13_STAGE0_RETROSPECTIVE_AND_STAGE1_EXECUTION.md`：先冻结
   requirement/measurement/evidence contract，再升级 PBS 资源；每个 loop 最多一个
   独立 Checker context，并使用索引优先、raw-on-demand 的低 token 工作流。

## 包结构

- `source/`：用户提供的两份原始规范及哈希。
- `docs/`：权威关系、架构、工程约定、Loop Engineering、Torch/Hugging Face、Miyabi、9 节点门禁、证据与变更控制。
- `stages/`：Stage 0–4 的阶段级计划、入口条件、loop 顺序和关闭条件。
- `loops/`：每个最小工作单元的执行卡。
- `plans/`：顺序、进度、验收追踪。
- `configs/`：计划级配置模板；实际项目须用 schema 校验并冻结 resolved config。
- `templates/`：loop、run、Checker、ADR、阶段 checkpoint 模板。
- `pbs/`：Miyabi PBS 角色启动模板；其中占位符必须由 Codex按当前系统和项目配置实测填充。
- `evidence/`：证据目录规范。

## 关键解释

用户要求的“50 local × 10 global steps”在本协议中解释为：

- `H=50`：每个 fragment 的计划发布间隔为 50 个**已完成的 local optimizer steps**，不是 micro-batches；
- `target_global_cycles=10`：`global_cycle = min_f(outer_update_count_f)` 达到 10，即每个 fragment 至少完成 10 次 outer update；
- 不引入不存在的统一全模型 global version，也不把各 learner 的 local steps 混称为 global steps。

9 节点门禁是每个 loop 的真实系统冒烟与回归门槛，不替代原规格中的 1B-token、2 小时慢 FS、10 次 kill 注入、matched-token 多种子等阶段级验收。
