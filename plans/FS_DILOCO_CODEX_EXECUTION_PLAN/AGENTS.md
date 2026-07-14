# Codex 执行契约

本仓库的默认执行者是 Codex。除用户显式改写外，Codex 必须遵守本文件。

## 1. 启动顺序与权威层级

本执行包当前安装在仓库的
`plans/FS_DILOCO_CODEX_EXECUTION_PLAN/`。执行时先固定：

- `REPO_ROOT`：`git rev-parse --show-toplevel` 的结果；
- `PLAN_ROOT`：`$REPO_ROOT/plans/FS_DILOCO_CODEX_EXECUTION_PLAN`。

所有未显式说明的生产代码、测试、runtime config、PBS 脚本和
`evidence/` 路径相对 `REPO_ROOT`；本包中的 `source/`、`docs/`、
`stages/`、`loops/`、`templates/` 与 `plans/` 相对 `PLAN_ROOT`。不得在
`PLAN_ROOT` 下创建第二套 `pyproject.toml`、`src/` 或 `tests/`。

每次开始或恢复工作时依次读取：

1. `source/STAGE0-4_SPEC.md`：Stage 0–4 算法、协议、不变量和验收的最高规范。
2. `source/RESEARCH_PLAN.md`：研究目标、阶段指标、方法学红线和研究证据。
3. 本文件与 `docs/00_AUTHORITY_AND_SCOPE.md`：用户补充的执行约束。
4. 当前 `stages/STAGE_*.md`、`loops/<stage>/<loop-id>.md`。
5. `$PLAN_ROOT/plans/PROGRESS.yaml`、
   `$PLAN_ROOT/plans/loop_states/<loop-id>.yaml` 与最近一次
   checkpoint/Checker 报告。

冲突处理：

- 算法和协议语义以 `STAGE0-4_SPEC.md` 为准。
- 执行者、Miyabi、Torch/Hugging Face 技术基线和 8+1/50×10 门禁以本执行包为准。
- Miyabi 站点安全规则与当前有效的 `miyabi-development` skill 优先于包内示例命令。
- 发现不可同时满足的约束时，停止当前 loop，写 ADR/阻塞证据，不得静默改变语义。

## 2. Miyabi 强制规则

任何涉及筑波大学 Miyabi 的操作前：

1. 确认并读取 [UnbearableFate/miyabi-development](https://github.com/UnbearableFate/miyabi-development) 的 `SKILL.md`。
2. 若未安装，安装到 `~/.codex/skills/miyabi-development`；若已安装，记录当前 commit，不得无记录地更新。
3. 执行 `hostname`，按 skill 的 host routing 选择本地、Miyabi login/control-plane 或 PBS compute-node 工作流。
4. login 节点只做查看、编辑、Git、静态 shell 检查、`qsub/qstat` 与日志检查。不得在 login 节点运行 `pytest`、导入 `torch/transformers/datasets/accelerate`、训练、推理、预处理、CUDA、MPI 或分布式 launcher。
5. 运行时检查只能在有效 PBS interactive/debug 或 batch compute allocation 中执行。
6. 读取 skill 中与任务相关的 references；本项目通常至少涉及 Python environment 与 PBS/launcher 指南。
7. 每个 run manifest 记录 skill 仓库 URL、commit、初始 hostname、节点类型、PBS job ID、queue、group、module list 和实际角色映射。
8. skill 示例中的 queue、Python 版本、module、路径和 walltime 不是项目常量；先发现并验证，再冻结入项目配置。

## 3. Torch/Hugging Face 基线

- 核心训练与张量计算使用 PyTorch。
- 模型、tokenizer、配置和数据管线基于 Hugging Face `transformers`、`tokenizers`、`datasets`；可使用 `safetensors`。
- 默认采用自定义训练循环。不得用 `Trainer` 隐藏 fragment snapshot、per-fragment counters、mixed-version adoption 或 outer update。
- 每个 learner 是独立单 GPU 进程。不得以 DDP/NCCL/RPC/parameter server 代替共享文件系统数据面。
- `accelerate` 仅可在单 learner 内用于设备/精度封装，且必须证明没有建立跨 learner 通信。
- PBS/MPI 可以作为进程启动控制面，但应用进程启动后，proposal、global state、readiness 和 adoption 只能通过规范允许的共享存储协议完成。
- merge 累加与数值 oracle 默认 fp32；模型训练精度由冻结 profile 决定并进入 manifest。

## 4. Loop Engineering

一次只执行一个 loop。每个 loop 必须经过：

`ORIENT → SPECIFY/RED → IMPLEMENT/GREEN → HARDEN → CHECK → PERSIST`

- ORIENT：确认规格版本、依赖、未通过 ID、当前 commit、环境、最近证据。
- RED：在实现前建立能对缺失/错误实现失败的证据。
- GREEN：仅解决当前最小 failing gap，不夹带新算法、拓扑或大规模重构。
- HARDEN：运行 loop 卡规定的边界、并发、恢复、性能或属性测试。
- CHECK：使用独立 Checker 上下文；Checker 不复用 Maker 的论证。
- PERSIST：落盘配置、命令、原始输出、分析、决策、哈希、进度和下一 gap。

不得把覆盖率当作需求证明；不得以“本地通过”替代 Miyabi 相关行为的真实验证。

### 4.1 运行前 admissibility

ORIENT 后、RED 前必须冻结并落盘：

- 当前 loop 覆盖的完整 requirement/acceptance IDs；
- 每个 ID 的 assertion、至少一个反例、权威观察点、聚合区间/公式与证据路径；
- run/evidence contract，包括 code/config/source/skill/scheduler/role/path identity、
  timestamp source、fail-if-exists 与 checksum finalization；
- 资源等级、单次 attempt budget、升级阶梯和失败后的最小降级复现。

RED 必须执行目标 API 或明确记录的错误 surrogate；只在 import/collection 阶段失败
不能证明目标语义。性能/并发结论必须先固定共同观测区间和聚合公式；不得把非同步
rank-local rates 直接相加。

L2/L3/L4 run 提交前，通用 validator 必须对 frozen package 返回 admissible。首次昂贵
run 失败后先保存证据并缩小复现，不得连续用相同大拓扑试错。

### 4.2 Checker 与 subagent

每个 loop 默认一个 Maker 和最多一个独立 Checker context。低成本 loop 只做最终
Checker；需要 L2/L3/L4 的 loop 由同一个 Checker 在提交前做 Phase A 静态 precheck，
运行后做 Phase B final check。blocked 修复继续该 Checker context，除非独立性已被
破坏；不得为重复确认创建多个 final Checker。

Checker 以无 Maker 聊天继承的最小输入启动，只接收 source/loop 路径、checked commit、
diff、evidence index 和 manifest。subagent 不用于普通检索、总结或主 Agent 可串行完成
的任务。

### 4.3 上下文与 token 纪律

- 一个执行 goal/session 默认只关闭一个 Maker loop；PERSIST 后先交付 compact handoff，
  不在同一长上下文自动进入下一 loop，除非用户明确要求连续执行。
- ORIENT 记录 goal token 起点和 soft budget，各阶段记录增量。超出 soft budget 触发
  context/input 审计与 handoff 压缩，不得以预算为由降低 correctness gate。
- 先读 `PROGRESS.yaml`、当前 loop state、stage checkpoint、evidence index 和最后
  Checker；索引不足或校验失败时才展开 raw evidence。
- 搜索先窄 `rg` 再读取命中范围；大 CSV/log 由程序输出计数、hash 与 anomaly summary，
  不整份注入上下文。
- 每阶段把已证实事实、唯一 gap、checked commit、selected evidence 和下一命令写入
  loop state；恢复工作以落盘 handoff 为准，不重放聊天历史。
- 同一 session 对 hash 未变化的 authority 不重复全文读取。工具输出只保留当前判断
  所需字段和行。

详细依据与 Stage 1 闸门见 `docs/13_STAGE0_RETROSPECTIVE_AND_STAGE1_EXECUTION.md`。

## 5. 真实 9 节点门禁

`S1-13` 建立基础训练系统与首个基准。自该 loop 通过后，所有后续实现 loop 的完成定义都包含一次新的门禁 run：

- 8 learners + 1 syncer；
- 9 个不同的 PBS compute host；
- 每 learner 使用 1 GPU，syncer 不使用 GPU；
- `H=50` local optimizer steps；
- `target_global_cycles=10`；
- 真实 Torch/HF 模型和真实数据路径，不用 mock 替代；
- loss 全部有限，聚合 loss 的末段稳健统计低于初段，并满足冻结阈值；
- active runtime 不超过提交前冻结的预算和兼容基准上限；
- 生成完整证据包与独立 Checker 结论。

低层检查未通过时不得消耗 9 节点。9 节点失败后先缩小复现和修复，再提交新的完整门禁 run；不得反复用集群试错。

## 6. Git 与变更权限

- 每个 loop 使用 `codex/<loop-id>-<slug>` 分支，或在用户明确指定的等价分支上工作。
- 记录开始 commit；跨本地与 Miyabi 只同步已提交的 tracked source。
- 启动时发现并记录真实 integration branch；当前仓库为 `master`。默认不合并该
  integration branch，不重写共享历史，不自动删除用户分支。
- 规范变化必须先写 ADR，列出证据、影响的 requirement/acceptance IDs、迁移和回退。
- 算法变化与系统优化必须独立开关；同一 loop 不得同时引入多个不可归因变化。

## 7. 完成声明

Codex 只有在以下材料都存在时才可把 loop 标为 `PASS`：

- RED 证据在错误实现上确实失败；
- GREEN/HARDEN 通过；
- 适用时 9 节点门禁通过；
- acceptance traceability 已更新；
- Checker 为 `PASS` 或不影响当前关闭的 `PASS_WITH_FOLLOWUPS`；
- `$PLAN_ROOT/plans/PROGRESS.yaml`、
  `$PLAN_ROOT/plans/loop_states/<loop-id>.yaml`、
  `$REPO_ROOT/evidence/indexes/<loop-id>.md` 和下一 loop 已持久化；
- 工作树状态与 commit 已记录。

无法完成时标为 `BLOCKED`，提供最小复现、原始输出、已排除原因、下一条可执行动作；不得以含糊总结代替证据。
