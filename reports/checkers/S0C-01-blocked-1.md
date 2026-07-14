# Checker Report：S0C-01/b993911

## Verdict

`BLOCKED`

当前实现的静态数学方向正确，但本 loop 不能关闭：RED 没有执行到错误公式的数值断言，三个 raw evidence package 的 checksum 均失败，且 weighting 的“单调性”测试没有调用被测实现。它们分别阻断 loop 卡要求的特异性 RED、证据不可变性和 ORACLE-02 的可执行证明。

## 独立输入

- source hashes：`RESEARCH_PLAN.md = 2c0db3148110da10ea62f445c189d2d53b405527d8da264b9af9817f5ca9d835`；`STAGE0-4_SPEC.md = fad26a8add51e70ddf50d0130b61d80fe06e1c3945221c29508412a5bcab2feb`。本地重算与三个 manifest 一致。
- loop card：`plans/FS_DILOCO_CODEX_EXECUTION_PLAN/loops/stage0/S0C-01.md`，SHA-256 `c6c20e8f2fb7580a9a15db47ec88c319fbf775ffee6bf11abf5546e7abf13da0`。
- code commit/diff：base `cd4ae9defa2688184077e09725175eb91d28b692`；checked commit `b9939119599bd7c64eb02641e36959ff0a3bd093`；8 files，464 insertions/1 deletion。`git diff --check cd4ae9d b993911` 通过。
- resolved config：stdlib-only CPU oracle，JSON fixture，fp32 rounding，tolerance `1e-6`，property seed `20260714`；不涉及模型、数据或 Torch 版本变更。
- RED evidence：`evidence/raw/S0C-01/20260714T221100Z-red-missing-oracle`。
- GREEN evidence：`evidence/raw/S0C-01/20260714T221900Z-green-oracle`。
- HARDEN evidence：`evidence/raw/S0C-01/20260714T222100Z-harden-oracle`。
- 9N package：不适用，基础训练系统尚未完成。

## 重新构建的目标

- 单一目标：建立与生产 syncer/storage 无依赖的 fresh/stale reference math；对 `s=1` 使用 `G_base - L_local`，明确拒绝 `G_current - L_stale`；证明 `tokens/(1+lambda_s*s)` 的归一化和边界性质。
- 受影响不变量：INV-07、OPT-01、STALE-05，以及 ORACLE-01/02。
- Acceptance：A-ALG-00 在 traceability 中由 S0C-02 主关闭、S0C-01 支撑；本报告只判定 S0C-01 支撑证据，不能提前关闭完整 A-ALG-00。
- 用户补充门禁：Miyabi login node 只做静态检查；Checker 未在 `miyabi-g1` 运行项目 Python。9 节点门禁不适用。

## RED 有效性

- 错误实现/缺失实现是否失败：只证明了“缺失实现”失败。RED 在 `cd4ae9d` 上 exit code 1，原因是 `ModuleNotFoundError: No module named 'fsbdd_stage0'`；测试加载阶段即终止，没有执行 stale counterexample。
- 失败是否对应需求：它对应一般性的缺失 oracle gap，但不满足本 loop SPECIFY/RED 中更强的要求——先运行 `G_current - L_stale` 错误替身，并由 `[wrong] != [reference]` 的数值断言产生失败。现有 RED 无法证明测试对该错误实现敏感。
- 是否存在偶然 timeout/脆弱断言：无 timeout；失败稳定但不具需求特异性。
- 必须修正：新增一个隔离的 RED target/错误替身，使 `G_current - L_stale` 的结果被当作 reference 检查并因具体数值差异失败；保存能看到 expected/actual 差异而不是 import error 的新 RED run。不得覆盖旧 RED 目录。

## 代码与协议检查

- scope：纯 `src/fsbdd_stage0` reference math、fixture/tests 与 CPU PBS wrapper；没有 learner/syncer/storage 实现，没有隐藏网络数据面，Torch 未改变。范围可接受。
- data plane：不适用。
- bounded state/discovery：不适用；API 只处理调用方传入的小向量。
- atomic publication：runtime 协议不适用；但 evidence publication 自身存在 checksum 缺陷，见“运行证据”。
- base/consumption：`validated_pseudo_gradient` 静态实现为先检查 identity/version/staleness，再调用 `pseudo_gradient(base, local)`；方向符合 INV-07/OPT-01。Consumption 属 S0C-02。
- numerical reference：每次基本运算经 binary32 rounding，golden fresh/stale 值与静态手算一致。尚无 merged-gradient/current-state transition API；该自动化证据应在 S0C-02 完成。
- config/identity：wrong identity、future base、too-stale 有 unit rejection；但 future/wrong-identity 只在测试代码中临时构造，未进入要求供 simulator/runtime 复用的 canonical fixture。

## Maker 未列出的反例

1. 独立 `s=1` 手算：取当前 `G^9=[10,1]`，stale base `G^8=[2,-4]`，stale local `L_s=[-1,-6]`，则正确 `g_s=G^8-L_s=[3,2]`，错误公式给出 `G^9-L_s=[11,7]`。再取 fresh local `L_f=[8,5]`，其 `g_f=[2,-4]`；令 stale/fresh tokens 分别为 60/30、`lambda_s=1`，raw mass 都为 30，故 weights 都为 `1/2`，`g_merge=[2.5,-1]`。无 momentum、lr=1 时必须作用于 current state，得到 `G^10=G^9-g_merge=[7.5,2]`；作用于旧 base 会得到 `[-0.5,-3]`，而使用错误 stale displacement 会得到 `[3.5,-0.5]`。三者明确可区分。
2. `test_oracle_02__fresh_is_undiscounted_and_staleness_is_monotone` 的 monotonic 部分只计算测试自身的 Python 公式，未调用 `inverse_staleness_weights`。例如把实现错误改成 `tokens/(1+lambda_s*(s % 3))`，现有 golden（仅覆盖 s=0/1/2）、单 contributor s=7、lambda=0、随机“非负且和为 1”检查仍可全部通过。对等 tokens、`staleness=[0,3]`、`lambda_s=1` 的正确 normalized weights 应为 `[0.8,0.2]`；该错误实现会给 `[0.5,0.5]`。必须增加直接调用实现并与独立 reference 比较的 randomized/formula/monotonic tests。
3. loop HARDEN 要求 wrong base identity 和 future base fixture 被拒绝；canonical JSON 只有正向 cases 与一个 forbidden 数值字段，没有可共享的 invalid identity/future-base fixtures。应将这些决策向量写入 canonical fixture 并由测试消费。

## 运行证据

- node/role：RED `2382230.opbs` on `mg0046`；GREEN `2382245.opbs` on `mg0056`；HARDEN `2382252.opbs` on `mg0056`。均为单 PBS compute node，符合 Stage 0 CPU 资源级别。
- GREEN：commit `c100227e89baf0a419ad20db29f842f9f3d58921`，7 tests，exit 0。
- HARDEN：checked commit `b9939119599bd7c64eb02641e36959ff0a3bd093`，8 tests，exit 0。
- loss/runtime/global cycles/local steps/fault/perf overlay：均不适用于纯数学 oracle；9N 不适用。
- raw evidence 可复算：不通过。对三个目录分别执行 `sha256sum -c checksums.sha256`，全部只有 `stdout/job.log` mismatch。checksum 记录的是空文件 SHA-256 `e3b0c442...`，实际 hashes 分别为 RED `d5d4e0e0...`、GREEN `6ebac49b...`、HARDEN `b20b6dbc...`。静态根因是 `pbs/stage0_cpu_tests.pbs` 先在 lines 101–104 生成 checksum，随后 line 105 仍向被 `tee` 的 `stdout/job.log` 写 completion message。因此三个 package 均不满足 `docs/07_TEST_EVIDENCE_CHECKER.md` §7.3 的 raw evidence 不可变性。
- run identity：三个 `run_id` 只有 UTC timestamp + phase label，没有 §7.3 要求的 PBS execution identity/submission nonce 与 config digest；manifest 内另列 PBS job ID 不能补足 run_id 组成契约。
- Checker 静态检查：`bash -n pbs/stage0_cpu_tests.pbs` 通过；未在 login node 执行 unittest 或项目 Python。

## Acceptance 判定

| ID | 证据 | 结论 |
|---|---|---|
| ORACLE-01 / INV-07 / OPT-01 | final code/fixture 的 base-relative displacement 静态正确；GREEN/HARDEN 显示 tests pass；独立反例可区分 correct/wrong/current application | 数学实现方向通过，但特异性 RED 与可信 raw package 缺失，不能关闭 |
| ORACLE-02 / STALE-05 | golden weights、500 random normalization/nonnegative checks、lambda/token/single boundaries；独立 mutation probe | `BLOCKED`：monotonic/formula property 未对被测实现建立充分约束 |
| A-ALG-00（S0C-01 支撑部分） | RED/GREEN/HARDEN 三包及 checked diff | `BLOCKED`；完整 A-ALG-00 仍需 S0C-02 transitions/golden cases |

## Follow-ups / 解除阻塞条件

以下均为解除阻塞条件，不是可延期 follow-up：

1. 提供数值失败的 wrong-formula RED，而非 import failure。
2. 修复 evidence finalization，使所有日志停止写入后再计算 checksum；用新的 fail-if-exists run IDs 重跑 RED/GREEN/HARDEN，并使每个 `sha256sum -c` 全部通过。run ID 同时加入 execution identity/submission nonce 与 config digest。
3. 增加调用真实 weighting API 的独立 formula/monotonic property test，至少覆盖 `s>=3` 的可区分 case；把 wrong-identity/future-base vectors 加入 canonical fixture。
4. 修正后由独立 Checker 重新检查；不得改写或删除当前失败 evidence。

## 签署

- Checker session/date：independent Checker `S0C-01` / 2026-07-14 Asia/Tokyo
- checked commit：`b9939119599bd7c64eb02641e36959ff0a3bd093`
- report hash：最终文件 SHA-256 应由 PERSIST 在本报告落盘后外部记录；不可自嵌入。
