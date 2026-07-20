# 06. STAGE0-4_SPEC.md 修订点与理由（配套 RESEARCH_PLAN v1.4 草案）

配套文件：[07-STAGE0-4_SPEC-v2.2-draft.md](07-STAGE0-4_SPEC-v2.2-draft.md) 是应用了
全部修订的完整版本。本文件只列"改什么、为什么、证据在哪"。

驱动来源：[04-research-plan-revision-notes.md](04-research-plan-revision-notes.md)
的 M1–M12（下称 M#）与 [02-code-review.md](02-code-review.md) §1（Stage 0-C
golden case 覆盖缺口）。每条注明驱动项。

**采纳耦合**：SP1/SP2/SP10 引用 RESEARCH_PLAN v1.4 草案新定义的 H3a/H3b/EQ3 与
S0A-03——本 Spec 草案必须与 [05-RESEARCH_PLAN-v1.4-draft.md](05-RESEARCH_PLAN-v1.4-draft.md)
**同批采纳**，否则条款引用悬空。二者未采纳前，source 下的 v1.3 / v2.1 仍是唯一
权威版本。

按影响排序：

---

## SP1. Stage 4 验收锚点与质量 gate 改用 H3a/H3b（驱动：M2）

- **位置**：§2 SIM-05、§8 A-STALE-07、A-ALG-02、"质量 gate 与回退"段。
- **证据**：Stage 0-A 预注册锚点预测挽回 **−0.0446pp**。RESEARCH_PLAN 原验收
  指标 3"提升 ≥ 仿真预测的 50%"在预测为负时为空条件；Spec 侧 A-STALE-07 只覆盖
  接受率偏差 ≤2×，挽回幅度的一致性无任何 Spec 级约束——即 Stage 4 可以在挽回
  与预测严重不符的情况下通过全部 Spec 验收。
- **问题**：Spec 与 RESEARCH_PLAN 对同一验收的口径开始漂移；"H3"这一名字在
  plan v1.4 中已不存在（拆为 H3a/H3b + 探索性 EQ3）。
- **修改**：A-STALE-07 增加挽回等价带（|实测−预测| ≤ max(2×|预测|, 1.0pp)）并
  标注 H3a；A-ALG-02 证据栏引用 H3a/H3b 并要求质量臂 from-scratch；质量 gate 段
  的回退触发从"H3"改为"H3b"；SIM-05 的"偏差 ≤ 2×"扩为完整等价带表述。

## SP2. 新增 SIM-06 / A-SIM-04：全矩阵归因是 Stage 4 的 Spec 级前置（驱动：M3）

- **位置**：§2 新增 SIM-06 与验收 A-SIM-04；§8 前言进入条件；§8 消融配置要求；
  §10 REPORT-01；§11 验收矩阵。
- **证据**：S0A 全矩阵方差极大（accepted-token 效率 8.9%–98.4%、token 加权丢弃
  率最高 88.8%），锚点处效率 33.5% 但挽回 ≈0——"丢弃很多"与"S_max=1 能挽回"
  是两个未归因的独立事实；2,592 个既有聚合足以回答，零新算力。
- **问题**：Spec §8 的消融矩阵是无条件全轴（3×2×4 = 24 臂 × 多种子），在挽回
  预测为负的证据下，这一预算没有决策依据；且"检查调度参数是否掩盖收益"
  （RESEARCH_PLAN 篇幅表第三行的要求）在两份文档中都没有可验收的落点。
- **修改**：SIM-06 规定三项产出（挽回全分布与 ≥3pp 有效区、丢弃按原因分解、
  预注册决策规则执行）；允许在既有输出缺少按原因计数时用冻结代码同种子补跑
  （分钟级），不得改 SIM-03 矩阵；A-SIM-04 为 Stage 4 进入条件；§8 消融矩阵
  改为由 A-SIM-04 决策记录两分支（保留全轴 / 裁剪为最小对照）；REPORT-01 增加
  丢弃按原因分解为必报量。

## SP3. 新增 DISC-04/DISC-05：测量真实性与阈值/单位纪律（驱动：M7）

- **位置**：§1.6；§3 A-BENCH-04 注记；§6 A-PERF-04 证据栏。
- **证据**：(a) S1-13 以 `minimum_step_seconds` 人为放慢 learner 才让容量门
  通过，真实速度下的 latest-wins/quorum/syncer 余量从未被测过；(b) formal run
  预算余量仅 0.1–0.5%、smoke 的 59/60 目标结构性不可达，各浪费一次多节点
  attempt；(c) A-BENCH-04 评估时把 `H=50`（local steps）按 50 秒折算预算，
  按实测 H_wall ≈ 9.5s 重算后结论侥幸仍成立（0.324s ≤ 0.95s）。
- **问题**：三类错误都发生在 Spec 管辖的验收执行中，但 Spec 没有任何条款禁止
  它们；Loop 纪律（§9）只约束测试形态，不约束测量口径。
- **修改**：DISC-04（吞吐/goodput/容量类验收 run 禁止人为 pacing；goodput 分母
  与分子同设置；门禁通过不得依赖放慢任一角色）；DISC-05（频率/预算类阈值冻结
  前必须有微基准或实测依据、余量 ≥10%；"延迟 ÷ 同步周期"类比值一律用实测
  wall-clock，H 的 steps/秒单位必须显式标注）；A-BENCH-04 附单位口径更正注记；
  A-PERF-04 证据栏显式引用 DISC-04。

## SP4. Stage 2 延迟分解必须含 syncer duty cycle（驱动：M11、M4）

- **位置**：§6 A-PERF-06 证据栏；§10 TEL-02。
- **证据**：S1 实测单 syncer 1.06s/update（160MB fragment）、duty cycle ~61%；
  update interval 增长 → adoption lag → 有效 staleness 的因果链是新风险表头号
  系统风险的观测哨。
- **问题**：TEL-02 列了 merge/publication 分项时间与 update interval，但没有
  duty cycle 这一容量余量的直接指标；A-PERF-06 的分解报告可以不含它而通过。
- **修改**：TEL-02 增加 syncer duty cycle（update 执行时间 ÷ 观察区间）；
  A-PERF-06 证据栏要求分解报告必含 duty cycle 与 per-fragment update interval。

## SP5. 新增 REC-06：断电级持久性范围必须在 Stage 3 ORIENT 裁决（驱动：M11）

- **位置**：§7。
- **证据**：当前发布路径无 fsync——对进程级 kill -9 完备（rename 原子性 +
  fail-closed 读侧），对节点断电/内核崩溃不完备（可见性记录可能先于 payload
  持久化，安全性保住但可用性需人工恢复）。C3 是论文主张，措辞范围必须与实现
  一致。
- **问题**：Spec §7 说"范围为 kill-restart 级别"但没有定义 kill 的边界在进程
  还是节点；论文措辞、durable 模式、回退链三个选项都有依据，无人裁决。
- **修改**：REC-06 规定 Stage 3 ORIENT 必须以 ADR 三选一：(a) C3 限定进程级
  crash；(b) 可配置 durable-publish 模式并量化代价；(c) 实现 FS-06 可选
  "上一完整 current 记录"自动回退链。选 (b)/(c) 时扩展 A-REC-03。

## SP6. BENCH-05 未运行事实与义务转移入文（驱动：M8）

- **位置**：§3 BENCH-05 与状态行。
- **证据**：`reports/stage0/storage_decision.md` 记录 BENCH-05（对象存储）未
  执行；§1.1 次级主张（对象存储可实例化）目前零数据支撑。
- **修改**：BENCH-05 条目注明"Stage 0-B 未执行、义务移至 RESEARCH_PLAN Stage 5
  必做廉价项，本条保留为基准定义"；状态行同步记录。

## SP7. 已关闭 Stage 的状态行与关键实测（驱动：M10）

- **位置**：§2、§3、§4、§5 各节首、§11 验收矩阵。
- **证据**：Stage 0 三线与 Stage 1 已全部关闭（PROGRESS.yaml；Stage 1 为 v2.1
  记录的 early-close 例外）。
- **问题**：Spec 无状态标注，后续 session 需交叉查 PROGRESS/报告才能确认哪些
  验收已闭合。
- **修改**：各节加一行状态（关闭日期 + 关键实测数字 + 证据指针）；§11 矩阵行
  注明已关闭/新增未完成项。不改任何已关闭验收的定义。

## SP8. Phase 2 边界加初步信号注记（驱动：M12）

- **位置**：§1.4。
- **修改**：加一句事实注记（1.06s/update@160MB、duty ~61%，PERF-06 证据预计由
  Stage 2 分解与 Stage 5 容量曲线产出）；承诺不变。

## SP9. STALE-04 golden case 必须覆盖生产 merge 后端（驱动：02 §1）

- **位置**：§4 状态行、§8 STALE-04、A-STALE-02 证据栏。
- **证据**：Stage 0-C 的 s=1 golden cases 当前绑定 torch 参考 merge 路径
  （`base_source` 仅 torch 版实现），而生产 syncer 走 numpy streaming 路径
  （close.py 确认 `merge_backend="numpy"`）。照原文执行，Stage 4 验收可以在
  从未测试生产代码路径的情况下通过。
- **修改**：STALE-04 增加"启用为运行时测试时必须覆盖生产 merge 后端，不得仅
  覆盖参考实现路径"；A-STALE-02 证据栏同步；§4 状态行预警此缺口。

## SP10. 消融质量臂 from-scratch 初始化入 Spec（驱动：M6）

- **位置**：§8 消融配置要求、A-ALG-02。
- **证据**：S1 实测 pretrained 续训的 loss 信号微弱（9,217 步 ratio 0.9940 vs
  门槛 0.99），以其做质量比较信噪比不足且带预训练先验。
- **修改**：消融配置要求注明质量口径 matched-token 且质量臂 from-scratch
  （引用 RESEARCH_PLAN §2.1）；A-ALG-02 证据栏同步。

---

## 审查过但未修改的项

- INV-01..10、STOR-01、DISC-01..03、§5 全部协议条款（MODEL/FRAG/LEARN/PROP/
  STALE/SYNC/OPT/GLOBAL/FS/PROG）、§9 Loop 协议、§12 冻结表主体：现有证据均
  支持，语义不动。
- DISC-01 在代码中的 6 处违反（02 §1）：这是实现债，Spec 条款本身正确且已
  禁止——修代码，不改 Spec。
- v2.1 的 S1-13 例外文本（§5.11）：原样保留。
- A-STALE-07 的"接受率 > 5%"下限：与锚点预测 5.31% 匹配的预注册值，保留；
  若 A-SIM-04 决策迁移锚点，随 ADR 一并更新。
