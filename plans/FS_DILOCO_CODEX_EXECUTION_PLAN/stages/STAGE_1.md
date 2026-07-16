# Stage 1：Fresh-only 最小端到端系统

## 目标

以 `S_max=0` 运行同一套通用 stale-ready 机制，实现可数值验证的 FS-only fragment training 闭环。技术底座为 PyTorch + Hugging Face，自定义 learner loop。

## 进入条件

- Stage 0-B 通过或缓解 profile 已冻结；
- Stage 0-C oracle 全部通过；
- Stage 0-A 默认参数建议可用；
- Miyabi `miyabi-development` skill 已安装/记录。

## Loop 顺序

| Loop | 内容 |
|---|---|
| S1-00 | 项目 skeleton、typed config、identity、manifest |
| S1-01 | HF model logical-layer registry |
| S1-02 | layer-aligned deterministic fragment map |
| S1-03 | STOR-01 interface 与 payload-first/visibility-last |
| S1-04 | bootstrap、per-fragment current/base bounded state |
| S1-05 | proposal schema、latest-wins、eligibility、consumption metadata |
| S1-06 | Torch/HF learner inner training core |
| S1-07 | H/offset snapshot 与 proposal publish |
| S1-08 | latest-only adoption、mixed-version、inner moments |
| S1-09 | syncer readiness、quorum、grace、公平调度 |
| S1-10 | base-relative math、direct merge、outer optimizer、streaming reduction |
| S1-11 | atomic global commit 与 consumed frontier |
| S1-12 | Profile A 数值 E2E 与 frozen evaluation snapshot |
| S1-13 | 基础系统关闭；首次 Miyabi 8+1/50×10 baseline |

## 技术约束

- GPT-2 small (`openai-community/gpt2`) + WikiText-2 raw 作为固定 9N 回归门禁 workload；
- source 要求的 160M/M=4/≥1B-token run 使用独立正式验收 profile，不复用 9N runtime baseline；
- custom training loop；
- learner 不组成 DDP world；
- FS 是唯一算法数据面；
- `S_max=0` 只是 config；
- proposal 从第一天携带 base version/content identity/tokens/steps；
- weighting 从第一天使用统一函数；
- base window 从第一天按 `S_max+1`；
- steady-state 不传完整模型。

## 执行与成本约束

- `S1-00` 必须先冻结 manifest/run identity time-source 语义，并提供通用 evidence
  builder/finalizer/validator。validator 覆盖 clean commit、source/skill/config、PBS
  scheduler、实际 roles、路径、fail-if-exists、placeholder、checksums 与 listed/actual。
- 每个 loop 的 RED 前先完成 requirement/counterexample/observation/aggregation/evidence
  matrix。性能 rate 使用 coordinator 共同区间，不合成非同步 rank-local rates。
- focused runtime 检查复用有效的 1-node interactive/debug allocation；L2/L3/L4 使用
  batch，并在首次失败后降级复现。
- 每 loop 默认最多一个独立 Checker context。L2–L4 由该 context 在昂贵提交前做
  Phase A、运行后做 Phase B；普通工作不使用 subagent。
- Agent 从 checkpoint/state/index 逐层读取，raw evidence on demand；大日志/CSV 先生成
  deterministic summary，以降低重复上下文和 token。
- 正式 160M profile（Pythia 风格 ~160M + FineWeb-Edu/C4 独立 shard）的
  model/tokenizer/dataset identity 必须在 `S1-06` 内以 ADR 冻结（不迟于其 HARDEN）；
  `S1-13` 的 M=4/≥1B-token long run 直接使用该 identity，不得在 S1-13 才开始选型。
- `S1-13` 起 loss/runtime/topology/protocol analyzer 输出 machine-readable JSON；
  Checker 按 validator summary + analyzer JSON 优先复核，raw 仅在校验失败时展开。
- 每张 loop 卡末尾的「启动指令」是 session 恢复入口；恢复时只加载卡内「恢复只读」
  清单，不重读全部 stage 文档。

## 验收覆盖

- A-FRAG-01..03
- A-PROP-01..03
- A-GLOBAL-01..02
- A-LEARN-01..03
- A-PERF-01
- A-ALG-01
- A-EVAL-01

默认另需完成 source 规定的 160M、M=4、真实 FS、≥1B token run。`S1-13` 的
9 节点 50×10 不替代它。2026-07-16 用户批准的单次 S1-13 early-close 例外允许以
已通过的 9N、完整五节点同 profile smoke 和 job `2394870.opbs` 的部分
loss/runtime 观测关闭；它不构成完成的 1B-token evidence。

## 基础能力标记

只有全部 Stage 1 acceptance、默认 M=4/160M/真实 FS/≥1B-token long run、首次
9N gate 和 Stage 1 Checker 都通过，`S1-13` 才能标记 PASS 并写入；本次仅可按
`reports/stage1/S1-13-early-close-adr.md` 的用户批准例外，用明确标注为 partial 的
正式观测替代 long-run completion：

```yaml
capability:
  id: CAP-BASE-TRAINING
  status: active
  baseline_9n_run_id: <run>
```

因此 `S2-01` 对 `S1-13` 的依赖同时也是 Stage 1 closure gate；不得先把 capability
标为 active、再补 long run。从此 Stage 2–4 每个实现 loop 都必须执行新的 9 节点门禁。

## Stage 关闭条件

- 所有 Stage 1 acceptance 通过；
- Profile A 单次 L2 error ≤1e-6，50 updates 无漂移放大；
- 真实 M=4 1B-token run loss 合理；
- 首次 8+1/50×10 loss/runtime/topology gate 通过；
- 9 节点 runtime baseline 与兼容 key 已冻结；
- independent-jobs 8+1 运行至少一次，或若队列暂不支持，有明确 Stage 3 前补跑 blocker；
- Checker 允许关闭。
- 若使用一次性 early-close 例外，checkpoint 必须列出全部原 long-run 未通过项，
  且不得声称完成 1B token。
- Stage 1 checkpoint 报告 attempts、queue/active time、Checker cycles、返工根因及
  context/token 改进是否生效。
