# Checker Report：S0A-01/ecbc32a

## Verdict

`BLOCKED`

The deterministic decision path and the three supplied PBS evidence packages are internally consistent, but the checked implementation does not satisfy the loop's bounded-state requirement. `_Simulator.accepted_ids` and `_Simulator.counted_rejections` are append-only sets of historical proposal identities. Their size grows with accepted/rejected proposals and therefore with simulation history. The HARDEN test named `test_inv_08__operational_state_and_trace_are_bounded` does not observe either set, so its pass does not prove `INV-08` or the S0A-01 HARDEN condition. S0A-01 cannot close at `ecbc32a25397fe1c91e314d926c4218078c072f3`.

## 独立输入

- source hashes：`RESEARCH_PLAN.md = 2c0db3148110da10ea62f445c189d2d53b405527d8da264b9af9817f5ca9d835`；`STAGE0-4_SPEC.md = fad26a8add51e70ddf50d0130b61d80fe06e1c3945221c29508412a5bcab2feb`。Checker 在 `miyabi-g1` login/control-plane 静态重算一致。
- loop card：`plans/FS_DILOCO_CODEX_EXECUTION_PLAN/loops/stage0/S0A-01.md`，SHA-256 `65940960b1334d96799a19a7da1fda6424fa3bbdc8cee3cf3c2f48c3554009a4`。
- checker template：`templates/CHECKER_REPORT.md`，SHA-256 `0cf29aa65c6a0d691cdb2661e5cacc3f2beb1619d729e98add082e36539ae297`。
- code ancestry：base `35694ba1d3444b3791b1a6f9f5be11acc8237570` → RED/spec `c6692fde7250905aa5ef965346719fb06ae6a8d9` → GREEN `057f70da8aeef383fddfe0a5925532eec19e1cb1` → checked HARDEN `ecbc32a25397fe1c91e314d926c4218078c072f3`。两次 `merge-base --is-ancestor` 均成功；`git diff --check 35694ba..ecbc32a` 成功。
- resolved core profile：`configs/stage0/simulator_core.json`，SHA-256 `4bc130bd7f2a17f753397495a24702df5f8a0151a89e71cefedacbc563afab0b`。共享 decision vectors SHA-256 `0dd9fe699bce3eef6b3da97b67d12a755ac227a0d36132e23e6feda128cbd5a8`。
- RED：`evidence/raw/S0A-01/20260715T000500Z-red-n4b2c-96de0c07d935`，commit `c6692fd...`。
- GREEN：`evidence/raw/S0A-01/20260715T001500Z-green-n8c4f-ffcce635e89e`，commit `057f70d...`。
- HARDEN：`evidence/raw/S0A-01/20260715T003000Z-harden-n5d7a-4df02fe149c8`，commit `ecbc32a...`。
- 9N package：不适用；Stage 0 且 `CAP_BASE_TRAINING` 尚未建立。

## 重新构建的目标

- 单一目标：用显式、稳定排序的离散事件队列模拟 `M` learners / `F` fragments 的 `H/offset`、constant/lognormal speed、upload/visibility、distinct-learner `Q`、fresh anchor、grace、`S_max`、one-per-learner 与 same-base once-only consumption，并与 Stage 0-C shared decision vectors 使用同一决策语义。
- 受影响不变量：`SIM-01`、`SIM-02`、`STALE-01`、`STALE-07`、`STALE-08`、`STALE-09`，以及显式 HARDEN 所要求的 `INV-08` bounded state。
- Acceptance：`A-SIM-01` 要求全部共享决策向量一致；本 loop 还要求 deterministic replay、真实 stale timeline 与 bounded operational state。
- 用户补充门禁：不得在 Miyabi login node 运行项目 Python；Checker 仅做 Git/file/hash/static shell inspection。运行证据来自已完成的 PBS compute jobs。

## RED 有效性

- 错误实现/缺失实现是否失败：是。RED surrogate 在 `c6692fd` 上以 unittest exit `1` 失败 3/3：按文件数误形成 quorum、visibility 时重写 snapshot base、按 insertion order 选 winner。
- 失败是否对应需求：是，分别对应 distinct learner quorum / `SIM-02`、`STALE-01` 的 snapshot base identity、`STALE-07` deterministic selection。
- 是否存在偶然 timeout/脆弱断言：未见。三项是 0.002s 的确定性断言，不依赖时间或外部 I/O；wrapper 将预期测试失败保留为 raw exit `1`。
- 局限：RED 是隔离的故意错误 surrogate，并未保存“production simulation test suite 在缺失实现上失败”的输出；其反例本身有效，但不能替代后述 bounded-state proof。

## 代码与协议检查

- scope/data plane：实现只在 `src/fsbdd_stage0/simulation.py` 和 Stage 0 exports/tests/config 中；没有训练、storage、Torch/HF 或网络数据面。
- shared vectors：simulation evaluator 与 event-loop selection 均调用 Stage 0-C `select_proposals`；7 个 JSON cases覆盖 duplicate learner、fresh-over-stale、consumed base、repeated sequence、new-base sequence jump、future/too-stale/wrong identity、empty boundary。GREEN 与 HARDEN 均通过。
- deterministic ordering：event key 为 `(time, priority, identity)`；proposal identity 含固定-width fragment/learner/sequence。候选 key 为 lower staleness → exact `Fraction` effective tokens → newer sequence → learner identity → proposal identity。per-learner RNG seed 为 `seed*1_000_003 + learner*97`，不依赖初始化顺序。相同 seed 的完整 result、反向 learner initialization、不同 seed 的差异均有测试。
- H/offset/speed/delay：offset 为 `floor(f*H/F)`；constant-ratio 与 seeded lognormal-jitter 两条路径均由 M=4 tests执行；upload 与 visibility delay 分开进入 proposal visibility，global adoption另经 visibility delay并在下一安全 step boundary 应用。`S_max` 配置限制为 `{0,1,2}`。
- quorum/grace/consumption：readiness由 eligible distinct learners和 fresh distinct learners共同决定；nonzero grace deadline稳定排队，deadline时重新选择；frontier只有每 learner 的 last sequence/base，成功 update后才推进。未选择 latest proposal不被消费，same-base replacement由 `consumed_base` 拒绝。
- rejection semantics：shared vectors证明 `consumed_sequence`、`consumed_base`、`base_identity_mismatch`、`future_base`、`too_stale` 与 `duplicate_learner`；模拟器内部生成的 proposals恒为identity-valid且不含可配置 progress cap，因此 missing-base/over-progress不属于本 core生成路径。
- **blocking bounded-state failure**：`simulation.py:283-284` 建立 `accepted_ids` 和 `counted_rejections`；`simulation.py:407` 对每个首次拒绝 proposal identity永久追加，成功 update路径也对每个 selected identity永久追加。两个集合没有 eviction/window/frontier压缩，故在固定 `M/F/S_max` 下仍随 outer updates单调增长。`tests/simulation/test_a_sim_01__event_semantics.py:124-148` 只检查 exposed `latest/frontier/trace/event_queue` maxima，无法检测这两个集合。此项直接违反 loop card“state/history 不随模拟更新无界增长”和 source `INV-08`。
- serialization：config/result/trace可经 canonical JSON；HARDEN 调用 `json.dumps(..., allow_nan=False)`。内部 queue/state不是公开持久化 schema，本 loop 当前没有 replay-authority claim。
- runtime import isolation：`git grep` 在 checked commit 中未发现 runtime production module导入 `fsbdd_stage0.simulation`；目前仓库尚无 learner/syncer production runtime。`src/fsbdd_stage0/__init__.py` 会为 Stage 0 package root eager-export simulator，故未来 runtime应避免依赖该 package root或拆分exports，但这在当前 commit不是 runtime use。

## 手工 grace-window 重放

使用 `test_stale_01__visibility_and_heterogeneity_create_real_stale_events` 的 M=4/F=2 constant profile：`H=10`、fragment offsets `(0,5)`、learner step intervals `(1, 5/3, 7/3, 3)` seconds、upload `0.5s`、visibility `5s`、`Q=2`、`Q_fresh=1`、grace `0.1H=1s`、`S_max=1`。

1. fragment 1 在 local step 5 snapshot。learner 0 于 `t=5` snapshot base 0，`t=10.5` visible；仅 1 distinct learner，未 ready。learner 1 于 `t=25/3` snapshot base 0，`t=83/6 = 13.833...` visible；此时两项均 fresh，distinct=2、fresh=2，打开 grace，deadline `t=89/6 = 14.833...`。
2. `(83/6, 89/6]` 内没有新的 fragment-1 proposal visible。deadline update event重新选择 learner 0/1，均 base 0；commit consumption frontier `(seq=1, base=0)`，fragment version `0→1`。这符合“grace结束时冻结当时全部 eligible distinct proposals”。
3. learner 2 的 step-5 base-0 snapshot 于 `t=35/3`，到 `t=103/6 = 17.166...` 才 visible；current version已为 1，因此 `s=1`，不是被visibility重写为fresh。learner 3 的对应 proposal于 `t=20.5` visible，也为 `s=1`。二者虽可满足 total Q，但 fresh=0，不能单独开启下一次 grace，符合 `STALE-09/Q_fresh`。
4. learner 0 在 adoption可见前于 `t=15` 产生的 sequence 2仍声明 base 0；`t=20.5` visible后因该 learner的 base 0 frontier已消费而以 `consumed_base` 拒绝。该路径同时验证 snapshot-base不变与 same-base consumption。

## Maker 未列出的反例

1. 将 duration扩大而保持 `M/F/H/delays` 固定：每轮成功 update都会把新 proposal ID加入 `accepted_ids`，每个首次终局拒绝加入 `counted_rejections`；即使 `latest`、frontier、trace和queue maxima恒定，这两个集合仍线性增长。当前 HARDEN assertion会继续通过，构成假阴性。
2. M=1分支在 `_step_interval` 中被特判，但 supplied GREEN/HARDEN没有 M=1测试；loop card明确将 M=1作为小边界。修复 bounded state时应增加 M=1 constant/lognormal smoke，并直接暴露/断言所有 operational identity-tracking容器的上界。

## 运行证据

- checksum manifests：Checker 对三个 run package逐项执行 `sha256sum -c checksums.sha256`，每包全部 14 entries为 `OK`。manifest/checksum SHA-256分别为 RED `c6d88960...` / `280aa043...`，GREEN `17972650...` / `de3d9da4...`，HARDEN `a5f0d7fb...` / `4e6d5c36...`。
- commit binding：RED manifest/env bind `c6692fd...`，GREEN bind `057f70d...`，HARDEN bind checked `ecbc32a...`；各包保存 clean branch status、compute hostname/nodefile、PBS job ID、modules、Python version、source hashes与skill identity。
- RED：job `2382610.opbs` / `mg0010`，3 expected failures，raw exit 1。
- GREEN：job `2382625.opbs` / `mg0014`，5 tests passed。
- HARDEN：job `2382636.opbs` / `mg0012`，S0C oracle + simulation共24 tests passed，并保存12-event sample。
- manifest caveat：每个 manifest的 `config_digest`只由submission environment传入，wrapper仅检查其12字节前缀出现在run ID；run package的空 `configs/` 目录没有保留resolved config或digest preimage。因此 checksum完整性、run-ID自一致性和Git binding可验证，但三个 manifest config digest不能从所留证据独立复算。run IDs以 `Z` 标记但文件mtime/commit时序为 Japan local 2026-07-14 23:00–23:06（UTC 14:00–14:06），命名时间也不应作为独立时钟证据。
- Checker runtime：初始 host `miyabi-g1`，无 PBS allocation；未运行项目 Python、pytest、torch或其他runtime。所有检查均为静态文件/Git/hash/shell inspection。
- loss/runtime/global cycles/local steps/fault/perf：Stage 0 event/count simulator不训练模型；9N、loss与GPU性能不适用。

## Acceptance 判定

| ID | 证据 | 结论 |
|---|---|---|
| `SIM-01` | M=4 constant/lognormal profiles，H/offset，upload/visibility，Q/grace，S_max，event trace | `PASS` for modeled core behavior；M=1 evidence缺失 |
| `SIM-02` / `A-SIM-01` | shared fixture 7/7 cases；GREEN/HARDEN conformance | `PASS` for decision-table conformance |
| `STALE-01/07/08/09` | exact candidate key、frontier、fresh quorum；automatic + manual grace replay | `PASS` |
| `INV-08` / S0A-01 HARDEN bounded state | append-only `accepted_ids` / `counted_rejections`; current test false-negative | `FAIL` |
| S0A-01 loop | RED/GREEN/HARDEN evidence plus all loop conditions | `BLOCKED` |

## Follow-ups

No non-blocking follow-up can close this loop. Required next action: replace the two historical-ID sets with bounded accounting tied to latest slots/frontiers (or another fixed-size design), expose a state-size diagnostic that includes every operational container, add a duration-scaling regression that fails the current implementation, add the M=1 boundary case, then produce fresh GREEN/HARDEN evidence at the corrected commit. The new evidence package should also retain a resolved config/digest preimage and use an accurate timestamp convention.

## 签署

- Checker session/date：independent Checker `/root/s0a01_checker` / 2026-07-14 Asia/Tokyo
- checked commit：`ecbc32a25397fe1c91e314d926c4218078c072f3`
- report hash：由外部对本文件完整字节计算；不可自嵌入。
