# Codex 开始执行

## 首次进入仓库

从仓库内任意目录执行以下静态检查，不导入重型运行库：

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
PLAN_ROOT="$REPO_ROOT/plans/FS_DILOCO_CODEX_EXECUTION_PLAN"
test -d "$PLAN_ROOT"
cd "$REPO_ROOT"
git status --short --branch
git rev-parse HEAD
git symbolic-ref --quiet --short refs/remotes/origin/HEAD || true
sha256sum \
  "$PLAN_ROOT/source/RESEARCH_PLAN.md" \
  "$PLAN_ROOT/source/STAGE0-4_SPEC.md"
sed -n '1,240p' AGENTS.md
sed -n '1,260p' "$PLAN_ROOT/AGENTS.md"
sed -n '1,240p' "$PLAN_ROOT/docs/00_AUTHORITY_AND_SCOPE.md"
sed -n '1,260p' "$PLAN_ROOT/docs/03_LOOP_ENGINEERING_PROTOCOL.md"
sed -n '1,320p' "$PLAN_ROOT/docs/13_STAGE0_RETROSPECTIVE_AND_STAGE1_EXECUTION.md"
sed -n '1,260p' "$PLAN_ROOT/plans/EXECUTION_ORDER.md"
```

随后读取 `$PLAN_ROOT/plans/PROGRESS.yaml` 中第一个依赖已满足、状态为
`not_started` 或 `blocked` 的 loop。默认首项为 `S0C-01`。

若当前 checkout 有与计划无关的未提交修改，不得把它们带入正式 run commit；先在
当前 feature branch 提交计划内文件，再从该 commit 创建 clean worktree 执行正式
loop。source、执行包和实现必须是 Git tracked，并能由 Miyabi 通过同一 commit 获取。

## 每个 loop 的固定动作

1. 从已记录的 integration branch 或前一 PASS commit 创建/切换 loop 分支并记录
   base commit；当前仓库 integration branch 为 `master`，不得按字面假定为 `main`。
2. 创建 `$PLAN_ROOT/plans/loop_states/`，复制
   `$PLAN_ROOT/templates/LOOP_STATE.yaml` 为 `<loop-id>.yaml`。
3. 执行当前 loop 卡的 ORIENT。
4. 落盘完整 requirement/反例/观察点/聚合/evidence matrix 和 run-package contract；
   RED 先证明目标语义失败，而不是仅证明 import 失败。
5. 先提交或至少持久化 RED 测试与失败输出。
6. 完成最小 GREEN；按 static→targeted→1-node→2-node 阶梯 HARDEN。focused 迭代
   优先复用有效 1-node interactive/debug allocation。
7. L2/L3/L4 前由唯一 Checker context 做 Phase A precheck，并运行 package validator。
8. 若处于 `S1-13` 或之后，执行新的 Miyabi 8+1/50×10 门禁。
9. 由同一独立 Checker context 做 Phase B final check，使用
   `$PLAN_ROOT/templates/CHECKER_REPORT.md`。
10. 更新 `$PLAN_ROOT/plans/PROGRESS.yaml`、
   `$PLAN_ROOT/plans/ACCEPTANCE_TRACEABILITY.csv` 和
   `$REPO_ROOT/evidence/indexes/<loop-id>.md`。
11. 提交 loop 结果并报告 branch/commit/证据路径，不默认合并真实 integration
    branch。

主 Agent 不为普通检索/总结启用 subagent。Checker 以无聊天继承的最小输入启动；每个
loop 默认最多一个 Checker context，blocked 修复通过 follow-up 继续，而不是重建全部
上下文。

## Miyabi 前置

```bash
hostname
test -f "$HOME/.codex/skills/miyabi-development/SKILL.md"
git -C "$HOME/.codex/skills/miyabi-development" rev-parse HEAD
sed -n '1,260p' "$HOME/.codex/skills/miyabi-development/SKILL.md"
```

若 skill 不存在，按其 GitHub 仓库安装后启动新的 Codex session。安装或更新动作必须记录；不在 login 节点运行 Torch/HF 导入或测试。

## 当前计划起点

Stage 0 三线可并行，但单一 Codex 会话的默认顺序是：

`S0C-01 → S0C-02 → S0A-01 → S0A-02 → S0B-01 → S0B-02 → S0B-03 → Stage 1`

集群不可用时，可先完成不依赖 Miyabi 的 ready loops，但不得越过 Stage 1 的 Stage 0-B 进入条件。
