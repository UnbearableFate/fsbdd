# Checker Report：S0A-02/e3b7613

## Verdict

`BLOCKED`

The 7,776-row sweep and its SIM-03/04/05 reports are internally consistent and independently recomputable, so `A-SIM-02` and the report-content portion of `A-SIM-03` pass. The loop nevertheless cannot close because its single objective explicitly requires a resumable sweep runner, while the supplied PBS entrypoint cannot resume an interrupted run and no RED/GREEN/HARDEN evidence exercises resumption. `pbs/stage0_simulation_sweep.pbs:35-40` refuses any existing run directory, yet a failed run necessarily leaves that directory and its checkpointed shard CSVs behind. In addition, `run_shard` trusts existing metric rows after checking only matrix index/modulo and configuration `row_key`; it does not bind retained rows to the generating code commit or revalidate their metrics. A restarted/mixed partial result could therefore be silently accepted even if orchestration were changed to reuse the directory.

## 独立输入

- source hashes：`RESEARCH_PLAN.md = 2c0db3148110da10ea62f445c189d2d53b405527d8da264b9af9817f5ca9d835`；`STAGE0-4_SPEC.md = fad26a8add51e70ddf50d0130b61d80fe06e1c3945221c29508412a5bcab2feb`。
- loop card：`plans/FS_DILOCO_CODEX_EXECUTION_PLAN/loops/stage0/S0A-02.md`，SHA-256 `d08c1b33023770c240f7b56847ae014966900abdd9747b6d1e7cd955040cb5a3`。
- checker template：SHA-256 `0cf29aa65c6a0d691cdb2661e5cacc3f2beb1619d729e98add082e36539ae297`。
- unresolved plan config：`plans/FS_DILOCO_CODEX_EXECUTION_PLAN/configs/stage0_simulation_scan.yaml`，SHA-256 `e176201d75fc95b0e7754174db341da9991954a25d65cebc7df6bac85e684e56`；resolved config：`configs/stage0/simulation_sweep.json`，SHA-256 `2722cdedce0b37de551b9a9fabed2e37338b35259030248bd725183e7b8f2723`。
- code ancestry：base `dd21542401ef1ea66392ea18e2eb702fe6a1244c` → ORIENT `2093d19` → RED `081f6ed19693ef614a5e3ae233bf47bc7fa7ebbe` → implementation `7799503` → corrected GREEN code `178b38040da6ff6daa0aee3d3f45f50a5772845f` → published results `037c7c1` → checked HARDEN `e3b7613359adbe0dd490793404d8ca61e5a6b593`。`git diff --check dd215424..e3b7613` 通过。
- RED：`evidence/raw/S0A-02/20260714T142600Z-red-n6c4e-2bf344f3d830`，commit `081f6ed...`。
- corrected full sweep/GREEN：`evidence/raw/S0A-02/20260714T143800Z-green2-n8d3c-51edd8599de6`，commit `178b380...`。
- HARDEN：`evidence/raw/S0A-02/20260714T144400Z-harden-n5a7e-705f86d5e2e5`，commit `e3b7613...`。
- 9N package：不适用；Stage 0 且 `CAP_BASE_TRAINING` 尚未建立。

## 重新构建的目标

- 单一目标：以可并行、可恢复的 runner 覆盖 SIM-03 全矩阵，输出 SIM-04 全部 token-weighted/interval/stale/recovery 指标，并冻结 Stage 1/4 profile 与 Stage 5 delay 建议。
- 受影响条款：`SIM-03`、`SIM-04`、`SIM-05`；Acceptance `A-SIM-02`、`A-SIM-03`。
- 扩展矩阵：SIM-03 的 `M × Q/M × heterogeneity × grace/H × visibility` 之外，显式加入 `S_max={0,1,2}`、`speed_model={constant_ratio,lognormal_jitter}`、`seed={17,29,43}`；固定 `F=4`、`H=50`、`lambda_s=1`、`Q_fresh=1`。
- 用户门禁：本 Checker 位于 `miyabi-g1` login/control-plane，仅执行 Git/file/hash、Ruby CSV/JSON 和静态 shell 检查；未运行项目 Python、pytest、torch、仿真或训练。

## RED 有效性

- RED 的三个确定性 surrogate 均以 exit `1` 失败：遗漏一个 visibility 值得到 `5,832 != 7,776`；报告缺少 interval/recovery 字段；错误 symmetric interval 得到 `200 != 50`。
- 三项分别对应 SIM-03 完整性、SIM-04 schema、解析 symmetric sanity；运行仅 `0.002s`，不是 timeout、import 或环境故障。
- RED 没有覆盖本报告发现的 resumability 缺口；现有测试中也没有 partial-checkpoint/restart 或 corrupted-retained-row 反例。

## 代码与协议检查

- scope/data plane：base-to-checked diff 只有 Stage 0 sweep/config/PBS/tests/reports/orientation state；未修改 learner/syncer、训练、storage protocol、Torch/HF 版本或网络数据面。
- exact matrix：独立从 resolved axes 以声明顺序重建 Cartesian index，全部 `0..7775` 与 CSV `matrix_index` 一致；全部 7,776 `row_key` 和 simulation `config_digest` preimage 由 canonical JSON 静态重算一致。四个 shard 各 1,944 rows，合并并按 index 排序后与 `simulation_raw.csv` 的全部字段逐行相同。
- full axes：行数为 `3×3×4×3×4×3×2×3 = 7,776`。每个去掉 `S_max` 的 key 恰有 `{0,1,2}` 三臂，共 2,592 paired seed rows；每个去掉 seed 的 profile 恰有三个 seeds，共 2,592 aggregate rows。
- metric units：全部 raw rows满足 `token_opportunities=processed_tokens×F`、`accepted_token_efficiency=accepted_tokens/token_opportunities`、`discard_rate=discarded/(accepted+discarded)`、`stale_acceptance_rate=stale_accepted_tokens/accepted_tokens`、`visibility_delay_over_h=visibility_seconds/50`。interval JSON 的 bucket sum/count/mean 与列值一致，`global_cycle=min(update_counts)`；四项 slot maxima均不超过 `M×F`。
- aggregate math：Checker 对全部 2,592 seed groups独立重算 n/mean/sample-stdev/normal-95%-CI/min/max，逐字段匹配 `simulation_aggregates.csv`。paired `S_max=1−0` recovery 为 `n=2592`、mean `0.038065253125540446` fraction（`3.8065253125540446` pp）、sample stdev `0.08904561475518057`、95% CI half-width `0.003428081270430378`；summary 一致。
- symmetric sanity：`Q=M`、heterogeneity 1、grace 0、visibility 0、`S_max=0`、constant 的 3 M × 3 seeds 共 9 rows；全部 interval mean `50.0`、discarded tokens `0`、stale accepted proposals `0`，与解析 H 周期一致。
- candidate ranking：Checker按冻结顺序独立重算四项候选。rank 1 是 `M=8,Q/M=0.5,Q=4,grace=0.1H`，region mean recovery `0.03362411430894311`、accepted efficiency `0.35827672261088284`、interval `41.20516111281898`，36 raw/36 paired rows；其余 rank 次序与 summary 完全一致。
- anchor/classification：固定 rank-1 profile 的 heterogeneity `2.0`、visibility `1s`、constant、3 seeds，paired recovery `-0.0004456725198324074` fraction（`-0.04456725198324074` pp），stale accepted-token rate `0.053107946226540664`，accepted efficiency `0.3348337641501025`；按 `<3pp` 权威表分类为 `ablation_or_discussion`。864 profile-level classifications独立重算为 `654/110/100`（ablation/compressed/full）。
- seed sensitivity：anchor 的 constant model按设计不使用 seed并报告零方差；同一默认 region 的 lognormal arm三种 seed结果不同：efficiency `0.3540325465/0.3501960260/0.3749776826`，stale rate `0.0654754388/0.0589021055/0.0827877634`，interval `39.779389/40.118269/40.707071`。HARDEN 因而确实观察 seed sensitivity。
- SIM-05 boundary：config、manifest、summary与 recommendation 均把预测限定为参数选择、Stage 4 later-measurement anchor和 Stage 5 delay points；报告明确写明不是 runtime correctness/training quality、Stage 4 remains mandatory。仓库 diff中没有把预测接入 correctness、runtime selection或 Stage 4 cancellation 的路径。
- unit clarification：模拟器把 fastest learner 的 nominal step interval设为 `1.0s`，所以当前 `visibility_seconds/H_steps` 数值等于相对 nominal-fastest `H=50s` 的比例。报告应在修复版明确这个基准；当前数值可复算，但不能被误读为任意真实训练 step time 下都不变的物理比值。
- **blocking resumability failure**：`run_shard` 每 25 rows原子 checkpoint，并会读取已有 CSV；但 production PBS wrapper在任何 shard启动前对 existing `RUN_DIR` 直接 exit 73。故 job interruption 后没有受支持的 resume path。即便绕过 wrapper，已有行只验证 index/modulo/row_key，不验证 `config_digest`、code commit或指标公式，可能把旧代码/损坏指标与新行混合。当前 full run是一次性完整结果，不是 resume proof。

## Maker 未列出的反例

1. PBS job在 shard写出第一个 25-row checkpoint后被终止；用相同 RUN_ID重新提交。预期 resumable runner保留并继续，实际在 `pbs/stage0_simulation_sweep.pbs:36-38` 因目录存在 exit 73，所有 checkpoint不可由入口点恢复。
2. partial shard中保留一个 matrix index与正确 row_key但把 `accepted_tokens`/efficiency改错，或该行来自同 config的旧 code commit。`run_shard` 会将其放入 `existing` 并跳过重算；aggregate只复核 index/row_key，也会接受混合结果。需要 per-row generation identity/metric validation或安全重算策略。

## 运行证据

- checksum manifests逐项 `sha256sum -c` 全部通过，且 manifest path set与实际除自身外文件精确同集：RED 19 entries，GREEN2 43，HARDEN 28。checksum-manifest SHA-256分别为 RED `96bc9ff3194113b89568316582df6481d130eaa8f6f0c9546bf9fb403f2f1a68`、GREEN2 `cec66b3fa97ef649e03369c93e4ea4b4e26a812a68e0a54af8e8cf8c0e3e033f`、HARDEN `b611d50d8b133045508b2f6cdab051241a00e1f6e02e3ae32c41af70be71d280`。
- digest preimages独立复算为 RED `2bf344f3d8306b565075729197a2538403ad77f4c3b1c261aedf2535d65af666`、GREEN2 `51edd8599de6d74ff4869f1911b5cb9e46c156eb51ce60845eb0265843e8dc00`、HARDEN `705f86d5e2e5103828c06f1959e1dec7e93d2541bc2b592322c0ff8f7c7b0b4e`；均与 manifest/run ID 一致。全部 3/6/12 input snapshots同时匹配 preimage hashes与各自 recorded Git commit blob。
- commit/environment binding：RED `081f6ed` on `mg0034`/job `2382753`；GREEN2 `178b380` on `mg0016`/job `2382794`；HARDEN `e3b7613` on `mg0036`/job `2382824`。三包均保存 clean branch status、source hashes、nodefile、modules、Python version与 skill commit `ad1fd34...`。
- GREEN2：四 shard exit 0，aggregate输出 `{complete:true,rows:7776}`，6 tests pass。HARDEN：S0C oracle + S0A-01/02 共32 tests pass，包括 unit/state-bound和lognormal seed-sensitivity checks。published manifest/summary/recommendation与 immutable GREEN2 result逐字相同。
- loss/runtime/global cycles/local steps/fault/perf：Stage 0 synthetic event/count sweep不训练模型；loss、GPU、9N与training runtime gate不适用。

## Acceptance 判定

| ID | 证据 | 结论 |
|---|---|---|
| `SIM-03` / `A-SIM-02` | exact 7,776 Cartesian rows；S_max/speed/seeds扩展；row/config identities；four shards | `PASS` |
| `SIM-04` | independently recomputed token units、interval distribution、2,592 paired recovery与2,592 aggregate rows | `PASS` |
| `SIM-05` / `A-SIM-03` | frozen Stage 1/4/5 recommendation、anchor、classification、mandatory Stage 4 boundary | `PASS` |
| loop single objective: parallel **resumable** runner | entrypoint rejects interrupted run directory；retained rows lack code/metric binding；no resume test/evidence | `FAIL` |
| `S0A-02` closure | report acceptance content passes, but required runner behavior and proof are incomplete | `BLOCKED` |

## Follow-ups

No non-blocking follow-up can close this loop. Required next action: add an explicit resume mode that reuses an existing run only after verifying frozen config, code/generator identity and immutable input snapshots; reject or safely recompute retained rows whose full identity/metrics cannot be trusted. Add RED coverage for interrupted partial shard and corrupted/old-code retained rows, then produce fresh corrected GREEN/HARDEN evidence demonstrating actual partial-run resume and unchanged full 7,776-row outputs. Also document that delay/H uses the nominal fastest 1-second step in this simulation.

## 签署

- Checker session/date：independent Checker `/root/s0a02_checker` / 2026-07-14 Asia/Tokyo
- checked commit：`e3b7613359adbe0dd490793404d8ca61e5a6b593`
- report hash：由本文件落盘后外部计算；不可自嵌入。
