# Checker Report：S0C-02/fe7e558

## Verdict

`BLOCKED`

Consumption frontier 与 outer-transition 的核心 reference 数学通过静态审查和既有 compute-node 证据，但 S0C-02 不能关闭。初版 requirement-to-evidence map 漏掉本 loop 明确受控的 `INV-05`、`PROP-06`、`OPT-04`；所谓 completeness test 自己也没有把这三个 ID 放入 required set，因此 GREEN/HARDEN 无法发现该空洞。此外，候选优先级对当前 API 接受的超大 token counter 使用 binary64 除法，会把两个不同 effective-token 值折叠成相同 key，并由 sequence 反转规范规定的选择顺序。

## 独立输入

- source hashes：`RESEARCH_PLAN.md = 2c0db3148110da10ea62f445c189d2d53b405527d8da264b9af9817f5ca9d835`；`STAGE0-4_SPEC.md = fad26a8add51e70ddf50d0130b61d80fe06e1c3945221c29508412a5bcab2feb`。Checker 在 login/control-plane 上静态重算一致。
- loop card：`plans/FS_DILOCO_CODEX_EXECUTION_PLAN/loops/stage0/S0C-02.md`，SHA-256 `2d8e494773191ae7fc007e087fa4e1100c909bda58bb6356cf22252ff5093ba4`。
- code commit/diff：base `1c41a9025a125d6b8140ddb048c49a6d9dea7028`；checked commit `fe7e558a483d222fea1a397e15f4192cafff3a47`；7 files，783 insertions。`git diff --check` 通过；工作树在 Checker 报告落盘前 clean。
- resolved config：stdlib-only CPU oracle；JSON decision vectors；显式 float32 merge/transition rounding；candidate policy `S_max/Q/Q_fresh/max_contributors/lambda_s`；无模型、数据、Torch/HF、网络或 production storage/runtime 变更。
- RED：`evidence/raw/S0C-02/20260714T230000Z-red-n2a6c-d4f45a81dbc3`，commit `b6b33053abf206ebe9c84d1e7d0e75d254841d44`。
- GREEN：`evidence/raw/S0C-02/20260714T231000Z-green-n5b1d-f2401c5c0232`，commit `f519ef6ccade1c988f6aa6c6f34be0e1d60fbe2c`。
- HARDEN：`evidence/raw/S0C-02/20260714T232000Z-harden-n7e4f-d48bf8412675`，commit `fe7e558a483d222fea1a397e15f4192cafff3a47`。
- full regression：`evidence/raw/S0C-02/20260714T232500Z-harden-regression-n9a3b-c96be5fe9ea1`，commit `fe7e558a483d222fea1a397e15f4192cafff3a47`。
- 9N package：不适用；Stage 0 且 `CAP_BASE_TRAINING` 尚未建立。

## 重新构建的目标

- 单一目标：建立 distinct-learner、consumed sequence/base、same-base once-only、publication-conditional frontier、direct averaging、PyTorch-style momentum/Nesterov 两步迁移，以及 Stage 0 初版 requirement/evidence map。
- 受影响不变量/条款：`ORACLE-03`、`ORACLE-04`、`ORACLE-05`、`INV-05`、`PROP-06`、`OPT-01`、`OPT-02`、`OPT-04`、`OPT-05`。
- Acceptance：`A-ALG-00`；S0C-01 的 fresh/stale displacement 与 weighting 是 supporting evidence，本 loop 为 primary closure。
- 用户补充门禁：Stage 0 不跑训练和 9 节点门禁；Miyabi login node 不运行项目 Python。Checker 仅作静态检查并复核既有 PBS compute-node raw evidence。

## RED 有效性

- 错误实现/缺失实现是否失败：是。隔离 surrogate 同时证明按文件数计 quorum、只检查 sequence 的 same-base replacement、使用旧 momentum buffer 的 Nesterov，以及漏掉 `ORACLE-05` 映射均会失败；4 tests 均到达特异断言，raw unittest exit code 为 1。
- 失败是否对应需求：是，分别直接对应 `INV-05`、`PROP-06`、`OPT-04`、`ORACLE-05`，不是 import error、timeout 或环境故障。
- 是否存在偶然 timeout/脆弱断言：未见。RED 用确定性小向量/集合，耗时 0.002s。PBS wrapper 将预期 unittest failure 作为有效 RED 完成。

## 代码与协议检查

- scope/data plane：diff 仅含 Stage 0 oracle、fixture/test、RED surrogate、初版 evidence map 与 orientation state；无 learner/syncer/storage、隐藏网络数据面、训练或 Torch 版本变更。
- decision semantics：eligibility 同时要求 `sequence > last_sequence` 与 `base_version > last_base_version`；future、too-stale、wrong identity 均分类拒绝。quorum 对 per-learner 去重后的候选计数，未选择同 learner proposal 不计第二票。
- one-per-learner 与确定性：普通计数域内，排序 key 明确为 lower staleness → larger effective tokens → newer sequence → learner/proposal identity，且 Maker 对一个 3-proposal case 的全部排列做了重放。超大 token counter 的优先级反例仍未处理，见下节。
- publication-conditional consumption：`commit_consumption(..., publication_succeeded=False)` 返回未推进 frontier；成功时 sequence 与 base frontier 同时推进。再次 polling 被 `consumed_sequence` 拒绝；同 base/new sequence 被 `consumed_base` 拒绝。
- bounded state：每 learner/fragment 只保存两个整数 frontier；1000 次成功 update 后仍为单个 `ConsumptionFrontier(999, 999)`。在规范固定 membership 假设下，状态为 `O(M x F)`，不随 update 历史增长。
- direct merge/current application：每个 contribution 使用 `G_base - L_local`，按 float32 顺序累加。fresh 控制中 weights `0.25/0.75` 得到 local average `[6.5, 2.5]`；mixed stale/fresh 反例得到 merged displacement `[3,3]`，明确作用于 current `[12,-2]` 得 `[9,-5]`，而非作用于 old base。
- PyTorch-style Nesterov：首步 buffer 初始化为 gradient；后续为 `b = mu*b + g`；Nesterov direction 为 `g + mu*b`，随后 `p = p - lr*direction`。fixture 两步得到 buffers `[2,-4]`、`[0.8,-0.6]` 和 parameters `[9.62,-4.24]`、`[9.648,-4.486]`，与 PyTorch SGD 的 zero-dampening transition 顺序一致。no-momentum/lr=1 control 也通过。
- requirement map：`reports/stage0/requirement_to_evidence.csv` 包含 `SIM-01..05`、`BENCH-01..05`、`ORACLE-01..05` 与 Stage 0 acceptance IDs，但没有 loop card 明列的 `INV-05`、`PROP-06`、`OPT-04`。completeness test 的 required set 也只枚举 SIM/BENCH/ORACLE/acceptance，故测试在缺少三个当前 requirement 的情况下错误通过。这违反 `ORACLE-05` 与 loop 的单一交付目标。

## Maker 未列出的反例

1. 候选顺序反例：令 `current_version=0`、`s_max=0`、`q=q_fresh=max_contributors=1`。learner `a` proposal 为 `tokens=2^53, sequence=2`；learner `b` proposal 为 `tokens=2^53+1, sequence=1`；二者均 fresh/valid。规范的 larger-effective-tokens 优先级必须选择 `b`。当前 `_candidate_key` 计算 `tokens / 1.0`，binary64 将两个值都表示为 `9007199254740992.0`，于是 sequence tie-break 选择 `a`，与 `STALE-07` 的优先级相反。`Proposal` 当前既不限制 token 上界，也不以整数交叉乘法比较 effective mass。
2. Traceability mutation：从 CSV 删除（或维持缺失）`INV-05`、`PROP-06`、`OPT-04`，当前 `test_oracle_05__initial_requirement_evidence_map_is_complete` 仍通过，因为 lines 184–196 的 required set 从未包含这些 loop requirements。该反例已由静态集合差构造，Maker tests 没有覆盖。

## 运行证据

- checksum manifests：Checker 对四个指定目录逐一执行 `sha256sum -c checksums.sha256`；每包 14/14 entries 全部 `OK`，且每个 manifest 恰覆盖除自身外的全部 14 个文件，无遗漏或额外路径。manifest SHA-256 分别为 RED `f9c1fdc0f54c8e8f97977b0b56b31456f521db32ac2ae8cc529ec2b072f1be21`、GREEN `feb53a689468771aaf21a218c33e63065d11843436736ae81fb523ba9e51780c`、HARDEN `080e05432c92a2ef2de3a5845e354b46adba341872d33532218a2b21cc566368`、full regression `e536861a3d565fb82c4655978427786a3a7601c7170f4e592cfaf3289105e2f5`。
- RED：job `2382347.opbs`，compute host `mg0067`，4 tests/4 expected assertion failures，unittest exit 1。
- GREEN：job `2382351.opbs`，compute host `mg0067`，6 tests pass。
- HARDEN：job `2382361.opbs`，compute host `mg0069`，8 tests pass，含 mixed current application 与 1000-update bounded frontier。
- full regression：job `2382392.opbs`，compute host `mg0069`，S0C-01 + S0C-02 共 17 tests pass；因此 wrong stale formula rejection、weighting、consumption、merge/current application、outer transition 的组合回归成立。
- environment：四包均记录一台 distinct PBS compute host、nodefile、modules（`nvidia/25.9`、`nv-hpcx/25.9`）、Python 3.13.13、clean branch status、source hashes、nonce/config digest、skill URL/commit。`bash -n pbs/stage0_cpu_tests.pbs` 通过。
- loss/runtime/global cycles/local steps/fault/perf：纯 Stage 0 mathematical oracle 不适用；证据没有作训练或性能 claim。

## Acceptance 判定

| ID | 证据 | 结论 |
|---|---|---|
| `ORACLE-03` / `INV-05` / `PROP-06` | shared vectors、全排列重放、same-base/repeated-poll、publication conditional commit、1000-update frontier | 核心 consumption reference `PASS`；超大 token priority domain 尚有反例 |
| `ORACLE-04` / `OPT-01` / `OPT-02` / `OPT-04` / `OPT-05` | fresh direct-average control、mixed stale/fresh current application、两步 Nesterov buffer/parameter trace | `PASS` |
| `ORACLE-05` | CSV 与 executable completeness checker | `BLOCKED`：当前 loop requirement IDs 漏映射，checker required set 同步漏项 |
| `A-ALG-00` | S0C-01 supporting evidence + S0C-02 full regression + checked diff | `BLOCKED`：primary loop 尚未满足 ORACLE-05 完成定义 |

## 解除阻塞条件

以下是关闭条件，不是可延期 follow-up：

1. 在初版 requirement/evidence map 中加入至少 `INV-05`、`PROP-06`、`OPT-04` 的真实 fixture/test/evidence 映射，并让 completeness test 从 authoritative loop requirement set 检查这些 ID，而不是维护一个漏项的手写子集。
2. 对 candidate effective-token priority 使用不会在接受域内反转顺序的比较方式（例如精确交叉乘法），或冻结、验证并映射一个足以排除 precision collapse 的 token 上限；加入上述 `2^53` 可区分反例。
3. 在新的 immutable run IDs 上重跑受影响 GREEN/HARDEN 与 S0C-01+02 full regression，验证所有 checksum manifest，再交独立 Checker 复查。不得覆盖本次四个 evidence package。

## 签署

- Checker session/date：independent Checker `/root/s0c02_checker` / 2026-07-14 Asia/Tokyo
- checked commit：`fe7e558a483d222fea1a397e15f4192cafff3a47`
- report hash：最终文件 SHA-256 由 PERSIST 在报告落盘后外部记录；不可自嵌入。
