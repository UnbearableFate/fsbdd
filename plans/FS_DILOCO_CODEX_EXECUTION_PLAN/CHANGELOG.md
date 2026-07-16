# 执行包变更记录

## v1.2.1 — 2026-07-16

- 按用户显式指令，以 job `2394870.opbs` 的当前部分观测关闭 S1-13/Stage 1。
- 保留原 formal contract 的未通过事实：未完成 1B token、global cycle 2400、
  loss ratio ≤0.99、scheduler exit 0 和正常 final package；不得将该 run 报告为完成的
  1B-token evidence。
- 一次性例外只接受有限正且连续的四 learner loss、下降的 partial aggregate trend、
  预算内 projected runtime，并与已通过的 9N gate 和完整五节点 smoke 联合支撑
  “训练已发生”这一较窄结论；不改变算法、协议、Stage 1 acceptance IDs 或后续
  Stage 2–5 的独立实验义务。

## v1.2.0 — 2026-07-15

- 按 Stage 0 复盘把执行改进落到全部 29 张 Stage 1–4 loop 卡：删除与
  `AGENTS.md` §4–§7 / `docs/03` 重复的协议样板（ORIENT 七项、RED/GREEN/HARDEN
  收尾句、通用完成定义等），每卡新增 loop 特有的 PREFLIGHT 冻结项（测量定义、
  workload identity、注入 profile 等）、显式停止条件与可复制「启动指令」（新
  session 的最小恢复入口）。
- 新增预注册 gate 单元（`AGENTS.md` §5.1、`PROGRESS.yaml` `gate_units`）：
  `GU-S2-ASYNC`（S2-01+S2-02）、`GU-S3-CRASH`（S3-02+S3-03）、
  `GU-S4-STALE-SELECT`（S4-03+S4-04）。单元先行 loop 以 ≤L2 证据 + Checker PASS
  关闭，9N 义务由单元 gate loop 的一次 run 合并履行，analyzer 必须同时断言两个
  loop 的 overlay 判据；另允许提交前预注册的双重用途 run（loop gate + stage
  experiment，如 S2-03 slow-FS 2h run）。标准路径 9N run 由 ~15 次降到 ~12 次。
- 9N gate 的 `analysis/*.json` 定为 machine-readable PASS/FAIL 判定；Checker 按
  validator summary + analyzer JSON 优先复核，raw 仅在校验失败时展开。
- 前移测量与身份冻结点：正式 160M profile 的 model/data identity 在 `S1-06` 以
  ADR 冻结；goodput/pause 定义在 `S2-04` 冻结供 `S2-05` 使用；0.5–1B workload
  identity 在 `S2-05` L4 run 前冻结；Stage 3 restart/interval 与 kill 矩阵、
  Stage 4 异构 profile / acceptance 口径 / 实验注册表全部提交前预注册。
- `S1-00` 的 run-package validator 增加 L0/L1 local 与 L2+ PBS 双 profile，低资源
  loop 不再拼装 scheduler 证据字段。
- `CODEX_START_HERE.md` 区分「恢复 loop」与「首次进入」两种入口，恢复时只读
  PROGRESS + loop 卡 + 卡内恢复清单。
- 记录并吸收 source spec 的 frontmatter 变更（移除 `repository:` 行，无算法/协议
  语义变化）：更新 `source/SOURCE_HASHES.txt` 与 `PROGRESS.yaml` source_hashes。
- 简化 Miyabi skill 相关规定：skill 已安装，直接使用 `miyabi-development`
  （`AGENTS.md` §2、`docs/05` §5.1/§5.2）；并删除本包内所有对 login/compute 节点
  分工的自行规定（docs/02/03/04/05/10/12、pbs/README、CODEX_START_HERE、涉及的
  loop 卡），执行位置一律以 skill 的 host routing 为准。run manifest 记录 skill
  commit、“示例常量先验证再冻结”等规则保持不变。
- 保留现有 correctness gates 不变：per-loop RED/Checker、9N 拓扑/loss/runtime/
  protocol 判据、1B-token/2h/kill 矩阵/matched-token 阶段验收均未放宽。

## v1.1.0 — 2026-07-15

- 根据 Stage 0 七 loops、六份保留 blocked Checker 和 stage closure 证据完成耗时与
  token 复盘，并明确 Stage 0 没有生产训练/9N 结论。
- 在 RED 前增加 requirement/counterexample/observation/aggregation/evidence matrix；
  import-only failure 不再作为语义 RED。
- 在正式 runtime run 前后增加通用 package validator，澄清 run timestamp source 与
  PBS qtime 的关系，并禁止合成非同步 rank-local rates。
- L2–L4 使用同一个独立 Checker context 做 Phase A/Phase B；subagent 不继承 Maker
  聊天且不用于普通工作。
- focused 检查优先复用 1-node interactive/debug allocation；昂贵失败先降级复现。
- 加入 index-first/raw-on-demand、短 handoff、窄工具输出等 token 控制，并把通用
  evidence tooling 前移到 S1-00。
- 保留现有 Python/Torch 环境不变。

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
