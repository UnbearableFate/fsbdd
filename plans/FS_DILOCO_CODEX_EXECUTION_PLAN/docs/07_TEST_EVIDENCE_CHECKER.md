# 07. 测试、证据与 Checker

## 7.1 测试金字塔

1. 数学 oracle：标量/小向量、精确 reference transitions；
2. unit：schema、identity、partition、eligibility、weights；
3. property：覆盖、连续性、归一化、单调性、bounded state；
4. concurrency：partial write、latest reorder、duplicate polling；
5. integration：单 learner/syncer、真实 storage backend；
6. system：1/2/9 nodes、真实 model/data；
7. failure/performance：kill、slow FS、backpressure、goodput；
8. research：长跑、多种子、matched-token。

## 7.2 Requirement-to-evidence

每个 requirement/acceptance 至少映射一个：

- 自动测试；
- 可重复命令；
- 原始 trace；
- 数值对照；
- 性能报告；
- 手工检查只用于无法自动化的站点事实，且需截图/命令输出。

`plans/ACCEPTANCE_TRACEABILITY.csv` 是 Stage 0–4 acceptance 的索引。项目实现后可扩展到全部 INV/DISC/... requirement。

每个 loop 在 RED 前另建完整 matrix，至少包含：requirement/acceptance ID、assertion、
counterexample、目标 API、authoritative observation、共同 measurement interval、
aggregation formula、raw evidence 和 verdict。completeness checker 的 required set 必须
从 loop/source 权威输入派生，不能与输出表共用一个手工列表。

## 7.3 证据不可变性

- 每次 run 使用唯一 `run_id`；
- `run_id` 至少组合 UTC timestamp、唯一 execution identity（coallocated 使用 PBS job
  identity，independent submit 使用 submission nonce）与 config digest；相同 resolved
  config 的重跑也必须得到新 ID；
- manifest 必须声明 run ID 中 UTC timestamp 的 `time_source` 和对应原始值。只有
  `time_source=pbs_qtime` 时才要求它与 scheduler qtime 精确相等；若使用 submission UTC，
  则另存 scheduler qtime/job identity 并验证先后关系，不把二者伪装为相同事件；
- `RUN_ROOT` 与 evidence directory 都由该唯一 ID 派生，并以 fail-if-exists 方式保留；
- 原始 stdout/stderr/metrics 不修改；
- 分析产物引用原始文件 SHA；
- 重跑生成新目录；
- evidence root 可 GC 临时 cache，但不可删 stage closure 证据；
- 大 payload 不必永久归档，至少保存 identities、metadata、必要小样本与 storage tree/size snapshot。

raw evidence 保留在 resolved `EVIDENCE_ROOT`，不提交 Git。Git 只跟踪
`$REPO_ROOT/evidence/indexes/<loop-id>.md` 及其中引用的 hashes/locations。

正式 run 在提交前和 finalization 后都要运行 package validator。final validator 至少
检查无 placeholder、manifest schema、clean checked commit、source/skill/config/
scheduler/role/path identities、fail-if-exists、checksums，以及 checksum 列表与实际保留
文件一一对应。自然语言 Checker 不再重复承担这些机械检查。

性能/并发 rate 必须有单一可解释的观察区间：优先 coordinator 在全 rank barrier 后
start、收齐 all-done 后 stop。若不能同步，必须报告每个局部区间并给出有证明的 union/
intersection estimator；禁止求和非同步 local rates 或用 `sum(count)/max(local elapsed)`
伪造共同吞吐。

## 7.4 环境证据

至少：

- branch/commit、dirty state；
- source spec hashes；
- Miyabi skill URL/commit；
- hostname、PBS job ID、nodefile、queue/group；
- module list；
- Python/Torch/CUDA/HF versions；
- GPU model/UUID；
- filesystem/path/device information；
- env allowlist（不泄露 secrets）；
- resolved config and asset hashes。

## 7.5 Checker 独立性

Maker 完成后启动新的 Codex checker session。Checker 不读取 Maker 聊天推理，只读取落盘材料。Checker 不先接受 Maker 结论；从规范重建：

- 当前 loop 的单一目标；
- 受影响不变量；
- acceptance；
- 可能的反例；
- 门禁条件。

Checker 可以运行额外测试，但不应顺手修代码。发现问题输出 BLOCKED 或 follow-up；修复进入新的 Maker iteration。

独立不等于复制完整 Maker 上下文。Checker subagent 应无聊天继承启动，只读取本节列出的
落盘输入。一个 loop 默认复用同一 Checker 做昂贵 run 前的 Phase A 与 run 后的 Phase B；
不为 blocked 修复重复创建多个 fresh final Checker。Stage-level Checker 仍是独立的阶段
关闭职责。

## 7.6 Checker 最低清单

- 规格与 source hashes 一致；
- diff 只在 loop 范围；
- RED 有效；
- GREEN 未绕过 test；
- boundary/concurrency 覆盖；
- no hidden network data plane；
- storage discovery 有界；
- stale 用真实 base；
- consumed frontier 与 publication 同提交；
- 9 节点 role/node 真；
- loss/runtime thresholds pre-registered；
- logs 与 analysis 可复算；
- acceptance traceability 无空洞；
- no claims beyond evidence。
- bounded-state inventory 覆盖所有 mutable containers，而不只公开 counters；
- resumability 声明有真实 interruption/restart，且 retained rows 绑定 code/config/
  generator/formula identity；
- measurement interval/aggregation 与报告数值一致；
- package validator 通过且 timestamp source 没有虚假 exact-qtime claim。

## 7.7 阶段关闭

每个 Stage checkpoint 包含：

- 通过 loop；
- 通过 acceptance IDs；
- 未关闭 follow-ups；
- 决策/ADR；
- 关键配置和基准；
- Stage 级长跑/多种子/故障证据；
- 风险变化；
- 下一 Stage 进入条件；
- 独立 Checker verdict。

Stage 只在全部 required IDs 有自动化/可重复证据且 Checker 允许关闭时结束。

## 7.8 证据最小保留期

至少保留到论文/artifact 完成：

- 所有 stage closure runs；
- 首次/最近 9N baseline；
- 每个算法语义 acceptance；
- 所有负面结果；
- kill/atomicity matrix；
- matched-token raw metrics；
- 影响设计决策的失败 run。

中间重复的纯环境失败可归档压缩，但索引和哈希保留。
