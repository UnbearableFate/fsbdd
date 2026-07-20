---
title: FS-Based Decoupled DiLoCo Stage 0–4 需求与设计规格
version: 2.2-draft-claude
date: 2026-07-16
status: Proposal Draft（位于 reflection/，非权威 source）
governs: plans/01/RESEARCH_PLAN.md 的 Stage 0 至 Stage 4
supersedes: plans/01/FS_BASED_DECOUPLED_DILOCO_V1_REQUIREMENTS_AND_DESIGN_SPEC.md
based_on: source/STAGE0-4_SPEC.md v2.1（sha256 7a038a40…）
---

> **采纳说明**：本文件是 Claude 基于 Stage 0/1 证据提出的 Spec 修订完成稿，
> 修改点与理由见 [06-spec-revision-notes.md](06-spec-revision-notes.md)
> （SP1–SP10）。本稿引用 RESEARCH_PLAN v1.4 草案新定义的 H3a/H3b/EQ3 与
> S0A-03，必须与 [05-RESEARCH_PLAN-v1.4-draft.md](05-RESEARCH_PLAN-v1.4-draft.md)
> 同批采纳，否则条款引用悬空。采纳须走 §0 修订纪律 / §13 变更控制：由用户
> 批准、写 ADR、替换 `source/STAGE0-4_SPEC.md` 并更新 `SOURCE_HASHES.txt` 与
> `PROGRESS.yaml` 哈希。未采纳前，v2.1 仍是唯一权威版本。

# FS-Based Decoupled DiLoCo：Stage 0–4 需求与设计规格

## 0. 文档定位

本文是 [RESEARCH_PLAN.md](RESEARCH_PLAN.md) 中 Stage 0 至 Stage 4 的**规范性实现契约**。RESEARCH_PLAN 决定研究排程、指标与投稿策略；本文决定实现语义与验收标准。二者引用关系：RESEARCH_PLAN 引用本文条款 ID（INV/DISC/SIM/BENCH/ORACLE/MODEL/FRAG/LEARN/PROP/STALE/SYNC/OPT/GLOBAL/FS/PROG/PERF/REC/TEL/REPORT-xx 与验收 A-xx）。

本文不规定：仓库目录、模块/类/函数命名、命令行接口、具体文件路径、序列化库、线程/进程/异步框架。

术语：

- **必须**：对应 Stage 验收不可缺少；
- **应当**：除非有实测证据支持偏离，否则必须遵守；
- **可以**：允许的实现或实验选择；
- **不得**：违反即不满足本规格。

修订纪律：实现过程中发现本文无法满足时，必须先记录设计变更及其证据，再修改规格，并在 §13 变更记录中注明由哪个 Stage 的证据驱动；不得通过隐式行为改变算法语义。

Stage 5–6 与 Phase 2 不在本文范围内，其实验设计见 RESEARCH_PLAN；本文 §1.4 保证 Stage 0–4 的设计不阻碍它们。

---

## 1. 全局设计（所有 Stage 共用）

### 1.1 系统拓扑

- 固定数量的 learner，每个 learner 暂定 1 GPU、1 node，持有完整本地模型、独立 inner optimizer（AdamW 类）、独立数据 shard；
- 一个逻辑 CPU-only syncer（暂定 1 CPU node），负责全部 fragments；运行环境必须避免同时启动两个可写 syncer，本阶段不通过分布式选主解决误启动；
- 所有节点访问同一共享存储；
- 固定 membership，不支持运行中动态加入、退出或重新编号。

**存储原语抽象（STOR-01）**：协议正确性只允许依赖两个存储原语：

1. **原子可见性发布**——一个小型可见性记录可以被原子地替换，reader 永远只能看到完整的旧记录或完整的新记录（POSIX FS 上以原子 rename 实例化）；
2. **最终可读**——已发布的对象在有限延迟内对所有节点可读。

存储访问必须封装在仅使用上述两个原语的接口之后，不得散布依赖锁、追加原子性、目录事务或其他强语义的调用，以便后续以对象存储（原子 PUT / 条件写）实例化而不改上层协议。

### 1.2 符号与核心概念

| 符号 | 含义 |
|---|---|
| `M` | learner 总数 |
| `L` | 非空逻辑层数量 |
| `F` | fragment 数量，`0 <= f < F` |
| `H_f` | learner 对 fragment `f` 的计划同步间隔（Stage 0–4 统一为 `H`，带 per-fragment offset） |
| `v_f` | fragment `f` 当前 global version |
| `G_f^v` | fragment `f` 在 global version `v` 的参数 |
| `O_f^v` | fragment `f` 在 version `v` 的 outer optimizer state |
| `L_{i,f}` | learner `i` 发布的 fragment `f` 本地参数快照 |
| `b_{i,f}` | 该 proposal 声明的 base global fragment version |
| `s_{i,f}` | proposal staleness，`s = v_f - b_{i,f}` |
| `tokens_i` | learner `i` 自采用 base fragment 后处理的 token 数 |
| `Q` | 一次 fragment outer update 所需最小 distinct learner 数 |
| `Q_fresh` | 一次 update 所需最小 fresh（`s=0`）learner 数 |
| `S_max` | 可接受的最大 fragment-version staleness（配置值） |
| `lambda_s` | staleness 衰减系数，权重 `tokens/(1+lambda_s*s)` |

"stale fragment proposal" 严格指：proposal 声明的 base global fragment version 小于 syncer 当前 global fragment version。它不是损坏文件，也不是"当前 global fragment 不够新"。

### 1.3 系统级不变量

所有实现和测试必须证明以下不变量。

- **INV-01 learner 独立前进**：learner 的 inner training 不得等待其他 learner、quorum、grace window、syncer outer step 或完整 global model 的统一版本。
- **INV-02 fragment 单写者**：每个 fragment 的 global state 只能由唯一逻辑 syncer 推进。
- **INV-03 fragment 原子可见**：reader 只能看到旧的完整 global fragment state 或新的完整 state；不得看到新参数与旧 outer optimizer state 的组合或写入一半的 payload。
- **INV-04 proposal 完整可见**：syncer 不得读取尚未完整发布的 learner proposal。
- **INV-05 一次性消费**：同一 proposal 最多参与一次成功 outer update；一次 update 中同一 learner 最多贡献一个 proposal。
- **INV-06 有界 staleness**：只有 `0 <= s <= S_max` 的 proposal 才能进入候选集；未来版本、缺少 base、超限或 base identity 不匹配的 proposal 必须被拒绝。
- **INV-07 stale update 基于真实 base**：stale proposal 的 outer displacement 必须由它实际基于的 `G_f^b` 与本地 `L_{i,f}` 计算；不得用当前 `G_f^v` 减 stale local fragment 伪造 fresh update。
- **INV-08 状态增长有界**：权威状态、proposal discovery 成本和单次 update 成本不得随历史 outer update 数线性增长。
- **INV-09 mixed-version model 合法**：learner 与当前 global model 都可以由不同版本的 fragments 组成，不需要统一 version。
- **INV-10 telemetry 不参与正确性**：metrics、日志、可视化不得决定 proposal 是否已提交或 global state 是否有效。

### 1.4 范围外但不得被阻碍的扩展

- **Phase 2 多 syncer**：fragments 静态划分为互不重叠集合，每集合一个 syncer；任一 fragment 任一时刻只有一个逻辑 writer；相同输入下与单 syncer 数值等价；无 failover、无 active-active。per-fragment 独立权威状态（FS-04，无 global head）即为此保留。
  **初步信号注记（2026-07-16，SP8）**：Stage 1 实测单 syncer 串行处理约
  1.06s/update（160MB fragment）、duty cycle 约 61%；PERF-06 所需的正式瓶颈
  证据预计由 Stage 2 延迟分解（A-PERF-06）与 RESEARCH_PLAN Stage 5 容量曲线
  自然产出。本阶段承诺不变：不引入多 syncer。
- **对象存储实例化**：由 STOR-01 保证。
- **per-fragment 非均匀 `H_f`**：调度语义必须允许（LEARN-03、PROG-06），但不属于 Stage 0–4 验收。

### 1.5 明确不做（Stage 0–4）

- overlapping syncer ownership、选主、lease、fencing、自动 failover；
- dynamic membership；
- event sourcing、deterministic replay、完整提交历史、prefix replay、完整审计链；
- exact learner recovery（恢复语义见 §7，允许 inner optimizer 重建）；
- 分布式数据库或日志服务；无界 proposal 队列；
- sub-tensor fragmentation；MoE、encoder-decoder、多模态模型；TP/PP/FSDP/ZeRO 组合；
- 通信量化、稀疏化、压缩；动态 layer importance 调度；
- RDA merge 对照（属 RESEARCH_PLAN Stage 5；本文只要求 merge 接口可插拔，见 OPT-02）。

共享存储上"先完成 payload、后宣布可见"属于正常并发读写的最低要求，不视为复杂容错功能。

### 1.6 实现纪律：接口通用、行为先收窄

- **DISC-01**：`S_max` 必须是配置值。`S_max = 0` 不得成为独立代码路径；禁止把"base 必须等于当前版本"硬编码进数据模型或聚合逻辑。
- **DISC-02**：以下五个要素自 Stage 1 即落地：
  1. proposal 元数据携带 base version + base content identity + tokens/steps（LEARN-07/PROP-01）；
  2. 资格检查参数化为 `0 <= s <= S_max`（`S_max=0` 时退化为 `b == v`）；
  3. 权重函数实现为 `tokens/(1+lambda_s*s)`（`s=0` 时退化为纯 token 加权）；
  4. 消费状态记录 last consumed proposal sequence + last consumed base version（PROP-06）；
  5. base 保留窗口实现为大小 `S_max+1`（`S_max=0` 时即只保留当前版本）。
- **DISC-03**：`S_max`、`lambda_s`、`Q`、`Q_fresh`、grace window、`H` 与 offsets 自 Stage 1 起进入冻结 run 配置与实验记录。消融两臂必须共享同一二进制，只差配置值。
- **DISC-04 测量真实性（新增，SP3）**：任何吞吐、goodput、容量类验收 run 不得
  包含人为 step pacing 或等效放慢机制；为可观测性引入的节奏控制在测量 run 中
  必须关闭。goodput 分母（关闭通信的本地吞吐）必须使用与分子完全相同的确定性、
  精度与 telemetry 设置。节奏/预算类门禁的通过不得依赖放慢任一角色。
- **DISC-05 阈值与单位纪律（新增，SP3）**：一切频率/预算类验收阈值（runtime
  预算、cycle 目标、可见性/写入预算）冻结前必须有单节点微基准或既有实测依据，
  预算余量 ≥ 10% 预测中位数。所有"延迟 ÷ 同步周期"或"时间 ≤ x% 同步周期"类
  比值一律以**实测 wall-clock 同步周期**计算并注明步时来源；`H` 以 local steps
  或秒表述时必须显式标注单位。（历史教训：A-BENCH-04 曾把 `H=50` steps 按 50
  秒折算预算；S1-13 曾以人为最小步时保门禁节奏——两类错误均不得复发。）

---

## 2. Stage 0-A：离散事件仿真规格

**状态（2026-07-16）**：SIM-01..05 / A-SIM-01..03 已关闭（评估 run
`20260714T145700Z-green3-resume-n7b4a-14126dde2f00`：7,776 行全矩阵、3 种子、
2,592 个 profile 聚合）。预注册 Profile B 锚点预测挽回 **−0.0446pp**、stale
接受率 5.31%、accepted-token 效率 33.48%——RESEARCH_PLAN 篇幅表第三行已触发
（记录见 RESEARCH_PLAN §0.1 决定 1）。SIM-06 / A-SIM-04 为本修订新增，未完成。

- **SIM-01 仿真对象**：M 个 learner（速度取恒定异构比与 lognormal 抖动两种分布）、F 个 fragment 的交错发布调度（`H`、offset）、上传与可见性延迟、quorum `Q`、grace window、`S_max ∈ {0,1,2}` 资格规则、one-per-learner、same-base 一次消费。不模拟训练质量，只模拟事件与计数。
- **SIM-02 忠实度**：仿真的接受/拒绝/消费决策逻辑必须与 §5.5 和 §8 的语义一致，并与 Stage 0-C oracle 共享同一组决策测试向量。
- **SIM-03 扫描矩阵**：`M ∈ {4,8,16}`、`Q/M ∈ {0.5,0.75,1}`、异构比 `∈ {1.0,1.5,2,3}`、grace `∈ {0, 0.1H, 0.25H}`、可见性延迟 `∈ {0, 1s, 5s, 30s}`。
- **SIM-04 输出**：token 加权 discard rate、accepted-token 效率、fragment update interval 分布、stale 接受率、`S_max=1` 相对 `S_max=0` 的挽回幅度预测。
- **SIM-05 用途边界**：输出用于默认参数选型、Stage 4 验收锚点（H3a 等价带：接受率偏差 ≤ 2×、挽回差 ≤ max(2×\|预测\|, 1.0pp)，见 §8 A-STALE-07）和 Stage 5 延迟扫描测点；不得作为运行时正确性依据（INV-10 同理适用）。
- **SIM-06 全矩阵归因分析（S0A-03，新增，SP2）**：消费 SIM-03 扫描的既有输出
  （profile 聚合与种子级明细），产出三项：
  1. **挽回全分布**：`S_max=1` 相对 `S_max=0` 的 accepted-token 效率提升在全
     矩阵上的分布；报告是否存在挽回 ≥ 3pp 的参数区及其位置（`Q/M`、异构比、
     grace、可见性延迟的哪个组合）；
  2. **丢弃归因分解**：token 加权丢弃按原因分解——latest-wins 覆盖、base 过期
     （too-stale）、quorum 已满未选、grace 截止——并归因锚点效率（33.5%）与
     全矩阵效率方差（8.9%–98.4%）的主导轴；
  3. **预注册决策规则执行**：存在有效区 → Profile B 锚点经 ADR 迁移至该区、
     Stage 4 消融矩阵保留全轴；不存在 → Stage 4 消融矩阵按 §8 裁剪。
  既有输出缺少按原因分解的计数时，允许以冻结的同版本仿真代码与相同种子补跑
  （分钟级成本），不得改动 SIM-03 矩阵或接受规则。零新 GPU 算力。

**验收**：

| ID | 必须证明的事实 | 最小证据 |
|---|---|---|
| A-SIM-01 | 仿真决策与 oracle 决策表一致 | 共享测试向量全部通过 |
| A-SIM-02 | 扫描覆盖 SIM-03 全矩阵 | 参数扫描报告 |
| A-SIM-03 | 产出默认 profile 建议与挽回幅度预测 | 选型报告（含 RESEARCH_PLAN Stage 0-A 篇幅表所需数据） |
| A-SIM-04 | 全矩阵归因完成且决策规则已执行（新增，SP2） | 归因报告 + 决策记录；Stage 4 进入条件（RESEARCH_PLAN S0A-03） |

---

## 3. Stage 0-B：目标存储微基准规格

**状态（2026-07-16）**：BENCH-01..04 / A-BENCH-01..04 已关闭，决策 `PASS`
（`reports/stage0/storage_decision.md`）：Miyabi Lustre 跨节点可见性 p99
7.95ms；10^5 次原子可见性替换 0 违例（unsafe 对照 5,208 次 detector 命中）；
持续元数据吞吐下界 39,403 ops/s（需求 77×）；16 并发流写一轮 fragments 最坏
0.324s。**BENCH-05 未执行**，旁证义务移至 RESEARCH_PLAN Stage 5（必做廉价项）。

- **BENCH-01 可见性延迟**：节点 A 完成发布 → 节点 B 首次可读的延迟分布（p50/p99），空载与负载两组。
- **BENCH-02 原子性**：并发 writer/reader 下 ≥ 10^5 次可见性记录替换，reader 每次读到的必须是完整旧版或完整新版。
- **BENCH-03 元数据吞吐**：模拟 `M × F` latest 引用的轮询模式，测持续 stat/readdir 速率与延迟，报告对共享元数据服务的影响。
- **BENCH-04 带宽**：50–500MB payload 顺序写/读吞吐，单流与 M 并发流。
- **BENCH-05（可选）对象存储**：同套基准在 S3 类存储上执行，原子 PUT / 条件写替代 rename，作为 STOR-01 可实例化的旁证。（Stage 0-B 未执行；执行义务已移至 RESEARCH_PLAN Stage 5 必做廉价项，本条保留为基准定义，SP6。）

**验收阈值**（不达标 → 记录缓解方案：换存储 / 调大 `H` / 显式屏障，之后才能进入 Stage 1）：

| ID | 阈值 |
|---|---|
| A-BENCH-01 | 可见性 p99 ≤ min(5s, 5% × 计划 fragment 同步周期) |
| A-BENCH-02 | 原子性 0 违例 |
| A-BENCH-03 | 元数据吞吐 ≥ 10 × 稳态轮询需求 |
| A-BENCH-04 | M 并发写一轮 fragments 的时间 ≤ 10% 同步周期（同步周期以实测 wall-clock 计，单位口径见 DISC-05） |

**单位口径更正（2026-07-16，SP3）**：Stage 0-B 评估 A-BENCH-04 时曾把
`H=50`（local steps）按 50 秒折算预算；按 Stage 1 实测步时（160M 下
H_wall ≈ 9.5s）重算，结论仍成立（0.324s ≤ 0.95s），A-BENCH-04 的 PASS 维持。
该类单位混用此后由 DISC-05 禁止。

---

## 4. Stage 0-C：算法数学 oracle 规格

**状态（2026-07-16）**：A-ALG-00 已关闭：`G^v − L_stale` 错误实现被拒，全部
golden cases 通过并可在 CI 重复。**注意（SP9）**：s=1 golden cases 当前绑定
torch 参考 merge 路径；Stage 4 启用为运行时测试时必须同时覆盖生产 merge 后端
（见 §8 STALE-04 / A-STALE-02）。

- **ORACLE-01 golden cases**：fresh pseudo-gradient（标量 + 小向量）；stale `s=1` 的 base-relative displacement；必须包含"错误实现 `G^v − L_stale`"的反例并证明其被拒绝。
- **ORACLE-02 weighting**：`r_i = tokens_i/(1+lambda_s*s_i)` 与归一化 `w_i = r_i/Σr_j` 的性质测试——非负、和为 1、staleness 增大权重单调不增、fresh 无折扣。
- **ORACLE-03 消费语义测试向量**：one-per-learner、consumed sequence、consumed base、same-base 拒绝；与 SIM-02 共享。
- **ORACLE-04 reference transitions**：direct weighted averaging merge 与带 momentum/Nesterov 的 outer optimizer 的最小参考迁移（含 momentum 状态演化）；同时保留无 momentum SGD 与 lr=1 direct-averaging 等价控制。
- **ORACLE-05 requirement-to-evidence map**：建立条款 ID → 测试/证据的映射表初版，此后每个 Stage 持续维护。

**验收（A-ALG-00）**：错误使用 `G^v − L_stale` 的实现被测试明确拒绝；全部 golden cases 通过且可在 CI 中重复。

---

## 5. Stage 1：fresh-only 端到端系统规格

**状态（2026-07-16）**：已按 S1-13 一次性用户批准例外关闭（见 §5.11 例外条目）；
原 ≥1B token run 义务保留为未完成（后续处理建议见 RESEARCH_PLAN Stage 1/Stage 2）。

运行配置：`S_max = 0`（经由 DISC-02 的通用资格检查实现），验收配置为 Profile A（§5.11）。

### 5.1 模型范围与逻辑层

- **MODEL-01 模型类型**：dense decoder-only Transformer causal LM——输入 embedding、有序 Transformer blocks、可选 block 外 normalization 或少量辅助可训练参数、`lm_head`。
- **MODEL-02 逻辑层定义**：按前向顺序为 input embedding 一层、每个完整 Transformer block 各一层、`lm_head` 一层。block 内的 attention/MLP projections、LayerNorm/RMSNorm、bias 及其他 block 独占可训练参数属于同一逻辑层；参数名形如 `layers.<index>.*` 时逻辑层边界是 `layers.<index>`。
- **MODEL-03 零散参数归属**：不属于上述三类的少量可训练参数（如 final norm）必须归入相邻逻辑层，规则确定性：归入同步字节数较小的一侧；相同则归前一侧；仅一侧相邻则归该侧；整个 run 中不得改变。
- **MODEL-04 共享参数**：每个底层参数只能有一个同步 owner。tied embedding/lm_head：共享权重默认归 input embedding 逻辑层；lm_head 仅拥有独有参数；不得产生重复 payload 或重复 outer optimizer state；映射必须记录 tied identity。
- **MODEL-05 参数覆盖**：所有需同步的可训练参数恰好属于一个逻辑层、恰好一个 fragment，不遗漏、不重复。可由配置重建的静态 buffer、mask、rotary cache 不进 payload；影响训练语义的可变 buffer 须在支持该模型前单独规定归属。
- **MODEL-06 歧义处理**：无法可靠识别结构时不得静默采用字符串猜测的 fragment map，必须要求显式、可检查的逻辑层映射并应用相同规则。

### 5.2 Layer-aligned fragment 划分

- **FRAG-01 完整层边界**：fragment 包含一个或多个完整逻辑层；逻辑层不得拆分。
- **FRAG-02 连续性**：fragment 由前向顺序连续的逻辑层组成，划分为 `F` 个连续非空区间。
- **FRAG-03 数量约束**：生产模式 `1 < F < L`；开发和数值 oracle 可用 `F = 1`，但不构成 fragment-mode 验收。
- **FRAG-04 均衡度量**：以同步 dtype 下的**同步字节数**为准，不按层数。
- **FRAG-05 划分目标**：约束内按优先级：最小化最大 fragment 字节数 → 最小化对平均值的总偏差 → 固定靠前切分规则；结果在所有节点确定性一致。
- **FRAG-06 超大逻辑层**：大于理想平均值的层可单独成 fragment；不为均衡拆层。
- **FRAG-07 静态映射**：fragment map 在 run 启动前冻结，learner、syncer、evaluation、导出全程一致；运行中不得重分片。
- **FRAG-08 映射报告**：run 启动前产出可检查摘要——逻辑层顺序、每层参数量与字节数、零散参数归属、tied ownership、每 fragment 层区间与字节数、最大/最小/平均及最大/平均比。报告用于验证，不是 runtime authority。

### 5.3 Learner 需求

- **LEARN-01 本地状态**：完整本地模型；独立 inner optimizer；独立数据流；每 fragment 当前采用的 global version；每 fragment 自采用后的 local steps 与 processed tokens；每 learner/fragment 单调 proposal sequence。
- **LEARN-02 连续 inner training**：FS 写入、syncer 聚合、其他 learner 的速度不得成为 barrier（INV-01）。
- **LEARN-03 发布调度**：统一间隔 `H`，per-fragment offsets 应当把发布均匀铺在 `H` 个 local steps 内，并考虑 fragment 字节数避免突发；不同 learner 可用确定性微小偏移防惊群，但不得破坏 quorum 合理形成。语义必须允许后续 per-fragment `H_f`（非本阶段验收）。
- **LEARN-04 快照边界**：只能在完整 inner optimizer step 之后取 fragment snapshot；不得捕获 forward/backward/optimizer mutation 中间态。
- **LEARN-05 GPU→CPU→FS 流程**：快照 → 有限 CPU staging → GPU 继续训练 → CPU 后台发布。训练阻塞只限 snapshot/copy 安全边界（性能量化验收在 Stage 2）。
- **LEARN-06 有界 backpressure**：每 learner/fragment 待发布状态有界；同时最多一个正式 publication in flight；未开始发布的旧 snapshot 可被新 snapshot 替代（latest-wins）；已开始发布的 payload 保持不可变直到完成或放弃；可以跳过计划发布；inner training 优先。
- **LEARN-07 proposal 内容**：发布本地 fragment 参数值（非 delta、无压缩）；必须声明 base global fragment identity（version + content identity），使 syncer 能在 fresh 与 stale 两种情况下重建正确 displacement。
- **LEARN-08 global fragment 采用**：发现新版本后：后台取得最新完整版本（可跳过中间版本）；在完整 inner optimizer step 后的安全边界整体覆盖该 fragment；更新 base version；重置该 fragment 的 local-step/token counters；继续训练，不等待其他 fragment。
- **LEARN-09 inner optimizer state**：采用 global fragment 时默认保留对应参数的 inner optimizer moments，只覆盖参数并重置 per-fragment counters；"采用时重置 moments"仅作独立 ablation，不得与默认结果混合报告。
- **LEARN-10 mixed-version local model**：learner 可同时持有不同 global versions 的 fragments；任何逻辑不得要求版本对齐后才能训练（INV-09）。
- **LEARN-11 in-flight proposal 与采用并存**：旧-base proposal 写入中允许采用更新 global fragment；已开始发布的 proposal 保持原 base identity，由 syncer 按资格规则接受或拒绝（`S_max=0` 下表现为拒绝，Stage 4 起可被接受），不得就地改写成新 base。

### 5.4 Proposal 语义

- **PROP-01 最小逻辑身份**：run identity、model/fragment-map identity、learner identity、fragment identity、单调 proposal sequence、base global fragment version、base content identity、local steps、processed tokens、snapshot 对应 local step、payload dtype/shape/字节数/完整性信息、payload identity。不规定单文件或多文件表示。
- **PROP-02 完整发布**：payload 先完成，再由小型可原子发布的可见性记录宣布完整；syncer 只通过完整可见性记录发现 proposal（INV-04、STOR-01）。
- **PROP-03 latest-wins discovery**：每 learner/fragment 只需暴露一个最新完整 proposal；发现空间 `O(M × F)`，不得扫描历史。
- **PROP-04 不可变性与单调 latest**：已发布 proposal 内容与身份不得修改；latest 引用只能向更大 sequence 前进；延迟完成的旧 publication 不得使 latest 倒退。
- **PROP-05 一人一票**：一次 update 中同一 learner 最多一个 proposal 被选择；quorum 按 distinct learner 数计算。
- **PROP-06 固定大小消费状态**：每 learner/fragment 固定大小记录最近已消费 proposal sequence 与最近已消费 base version，不保存无界 consumed 历史。同一 learner 基于同一 base 的累计 trajectory 最多成功消费一次；更新 sequence 但仍基于已消费 base 的 proposal 必须被拒绝。
- **PROP-07 基本资格**：进入候选集需满足：run/model/fragment-map/fragment identity 匹配；shape/dtype/完整性有效；sequence 大于最近已消费 sequence；base version 大于最近已消费 base version；local steps 与 tokens 为正；local progress 不超过配置上限；base version 不在未来；base content identity 与 syncer 保留的对应版本一致；`0 <= s <= S_max`。

### 5.5 Staleness 通用机制（本阶段以 S_max=0 运行）

- **STALE-01 定义**：选择时 fragment 当前版本 `v`、proposal base `b`，`s = v - b`。`s = 0` fresh；`1 <= s <= S_max` eligible stale；`s < 0` future/invalid；`s > S_max` reject。
- **STALE-05 权重**：未归一化 `r_i = tokens_i / (1 + lambda_s × s_i)`，归一化 `w_i = r_i / Σ_j r_j`。要求：fresh 不受折扣；staleness 增大权重单调不增；最终权重非负且和为 1；权重、staleness、tokens 进入 telemetry。默认 `lambda_s = 1.0`。替代 policy 不得在缺少本基线对照时替换它。

资格与消费全部经由 DISC-02 的通用实现；`S_max` 值属于 run 配置。

### 5.6 Syncer 需求

- **SYNC-01 单一逻辑 syncer**：负责全部 fragments（§1.1）。
- **SYNC-02 per-fragment 独立状态**：current parameters、current version、current outer optimizer state、outer update count、bounded base-parameter history（窗口 `S_max+1`）、每 learner 最近已消费 sequence 与 base version、readiness/grace 状态。不同 fragment 不共享需原子推进的全局版本。
- **SYNC-03 readiness 驱动**：不按 CPU 循环数机械聚合；由 fragment 当前版本、eligible distinct proposals、`Q` 与 `Q_fresh`、grace window、公平调度共同驱动。
- **SYNC-04 quorum 与 grace window**：必须支持 `Q = M`、grace = 0 的 reference profile 与 `Q < M` 的 decoupled profile；grace 固定有限；grace 结束后使用当时全部 eligible、未消费且满足选择策略的 distinct learner proposals。adaptive grace 非本阶段必需。
- **SYNC-05 fragment 公平性**：多 fragment 同时 ready 时用确定性无饥饿调度；高频 ready 的 fragment 不得无限阻止其他 fragment。
- **SYNC-06 有界内存聚合**：峰值工作内存与当前 fragment、其 outer state、少量 streaming accumulator 相关，不与 `M × fragment size` 或完整模型成正比；应逐个读取 proposal 做 streaming reduction。
- **SYNC-07 不物化完整模型**：稳态 outer update 不得读写完整模型；只有初始化、evaluation snapshot、最终导出可以组装。
- **SYNC-08 成功 update 的逻辑顺序**：固定 current version 与完整 current state → 读取验证候选 → 固定 selected set、base identities、staleness、weights → 重建各 pseudo-gradient → merge → 以 current state 执行 outer optimizer → 完整发布 new parameters + new outer state + next version → 发布成功的同时 selected sequences 及 base versions 成为已消费（publication 未完成则仍未消费）→ 旧 proposal 经 sequence/consumed-base/staleness 规则自然失效。
- **SYNC-09 选择稳定性**：update 开始执行后，selected set 与 weights 不因目录顺序、后到 proposal 或 wall-clock 改变；后到 proposal 参与下一次 update。

### 5.7 Merge 与 outer optimization

- **OPT-01 统一 pseudo-gradient**：每个 contribution 为 `g_i = G_f^{b_i} - L_{i,f}`；fresh 是 `b = v` 的特例；区别仅在 base version 与权重。
- **OPT-02 direct weighted averaging 基线与可插拔 merge**：必须提供 `g_merge = Σ_i(w_i × g_i)` 作为数值 oracle 与默认 merge。merge policy 必须实现为可插拔接口（后续 RDA 对照属 RESEARCH_PLAN Stage 5），任何替代 policy 不得在缺少 direct averaging 对照时成为默认。
- **OPT-04 outer optimizer**：主线为带 momentum/Nesterov 的 SGD 类，per-fragment 独立状态；同时保留无 momentum SGD 与 lr=1 direct-averaging 等价控制。
- **OPT-05 应用点**：merged pseudo-gradient 一律作用于 current `G_f^v` 与 current `O_f^v`，生成 `v+1`；不得回滚到旧 base 再执行 outer step。
- **OPT-06 数值身份**：一次 update 的数值身份由 current version 与 content identity、selected proposal identities、各 base identities、tokens/staleness/weights、merge policy、outer optimizer policy 与超参数、accumulation dtype、fragment map identity 决定；用于复现与 Checker，不形成无限历史 authority。

### 5.8 Global fragment 发布与 learner 采用

- **GLOBAL-01 global model 定义**：`{(G_f^{v_f}, O_f^{v_f}, v_f)}` 的集合，即版本向量；不要求每步物化完整 checkpoint。
- **GLOBAL-02 发布单位**：一次 publication 将 new parameters、new outer state、new version、selected sequence frontier 与 consumed base frontier、policy identity 与完整性信息作为一个逻辑整体宣布可见（INV-03）。
- **GLOBAL-03 版本单调**：每次成功 publication 使该 fragment version 恰好 +1；不得跳号、倒退或覆盖同一 version 的不同内容。
- **GLOBAL-04 latest-only adoption**：learner 可从 `v` 直接采用 `v+k`，不依次应用中间版本。
- **GLOBAL-05 evaluation snapshot**：evaluation 先冻结 fragment-version vector 与 content identities，再加载该固定组合；不得追逐变化中的 latest。
- **GLOBAL-06 bootstrap**：新 run 初始化产生完整冻结 fragment map 与每 fragment 的 version 0 参数、outer state、identity；learner 开始训练前取得完整 global version vector。

### 5.9 共享存储契约

- **FS-01 角色**：大张量 payload 交换、latest proposal discovery、per-fragment current global state、bounded base history、少量 run/fragment-map metadata、best-effort telemetry。不是事件日志或数据库。
- **FS-02 payload-first / visibility-last**：大 payload 在不可见状态完成写入与完整性验证，再经小型原子可见性记录暴露（STOR-01）。
- **FS-03 小型 current 引用**：每 fragment 一个小型 current 引用，指向当前完整 parameters、outer state 与 metadata；参数与 outer state 不得通过彼此独立的"最新文件"推断。
- **FS-04 per-fragment authority**：per-fragment current state 即 fragment authority；不存在要求全 fragments 共同提交的 global head。
- **FS-05 固定 discovery 面**：正常发现成本与 `M × F` 成正比，不与历史 proposal 或 outer update 数成正比。
- **FS-06 bounded retention**：稳态保留上限——每 fragment 当前 global state、前 `S_max` 个 base parameter versions、每 learner/fragment 最新完整 proposal、少量写入中临时对象、可选上一完整 current record（人工恢复辅助）、有限 telemetry。不保存全部旧版本、旧 proposals 或 loser attempts。
- **FS-07 完整性**：reader 至少验证 run identity、fragment-map identity、fragment identity、version/sequence、dtype/shape/字节数、payload 完整性标识、base content identity。
- **FS-08 能力预检**：由 Stage 0-B（§3）承担；训练前必须已通过或已记录缓解方案。
- **FS-09 GC 非关键路径**：临时文件、被覆盖 proposal、超窗 base payload 异步回收；GC 失败不影响正确性，只产生空间增长与告警。
- **FS-10 无历史扫描**：正常启动与稳态运行不得从头扫描或 replay 历史。

### 5.10 调度、进度与停止

- **PROG-01**：每 fragment 独立维护 outer update count。
- **PROG-02**：`global_cycle = min_f(outer_update_count_f)`。
- **PROG-03 停止条件**：以目标 global cycles、wall-clock/compute budget 或预注册 token/FLOP 预算之一冻结；不得只以最快 learner 的 local step 为准。
- **PROG-04 进度报告**：分别报告各 learner local steps/tokens、各 fragment outer update count、global cycle、各 fragment accepted tokens、fresh/stale accepted 数；不得混称为同一"global step"。
- **PROG-05 learning-rate 语义**：inner LR schedule 明确按 learner local steps 或 processed tokens 推进；outer LR schedule 默认按 per-fragment outer update count 推进；staleness attenuation 只改变 merge weight，不得隐式改变 outer LR。
- **PROG-06 频率扩展边界**：本阶段统一 `H`；后续 per-fragment `H_f` 实验必须同时报告 `Σ_f(fragment_bytes_f / H_f)` 以区分预算分配与总量增加。

### 5.11 Profile A 与 Stage 1 验收

**Profile A（Fresh Reference）**：固定 `M`；`Q = M`；`Q_fresh = M`；`S_max = 0`；grace = 0；direct weighted averaging；确定性 fragment schedule。全部超参数 run 前冻结入实验记录。

**Stage 1 验收**（同 RESEARCH_PLAN Stage 1 指标）：

| ID | 必须证明的事实 | 最小证据 |
|---|---|---|
| A-FRAG-01 | 每个参数恰好属于一个 fragment | 参数 identity 全覆盖检查 |
| A-FRAG-02 | fragments 连续且非空 | 映射 checker |
| A-FRAG-03 | 最大 fragment 在约束下最优或达可证明目标 | exhaustive 小模型 + 大模型摘要 |
| A-PROP-01 | 部分 proposal 不可见 | 中途读取反例 |
| A-PROP-02 | 同 learner 一次最多一个 contribution | duplicate proposal trace |
| A-PROP-03 | 同 proposal 不重复消费 | repeated polling trace |
| A-GLOBAL-01 | params 与 outer state 共同可见 | publication interruption test |
| A-GLOBAL-02 | per-fragment version 单调 | transition trace |
| A-LEARN-01 | learner 不等待其他 learner/quorum | speed-heterogeneity timing trace（单 learner ×0.5 减速，其余吞吐变化 ≤ 2%） |
| A-LEARN-02 | adoption 只覆盖目标 fragment | parameter identity check |
| A-LEARN-03 | mixed-version model 可持续训练 | end-to-end trace |
| A-PERF-01 | 稳态不传完整模型 | byte accounting |
| A-ALG-01 | direct merge 与 oracle 一致 | 单次 update 相对 L2 误差 ≤ 1e-6（fp32 累加），连续 50 updates 无漂移放大 |
| A-EVAL-01 | evaluation 使用冻结 version vector | snapshot manifest + report |

默认另须完成：160M 模型、M=4、真实共享 FS 上 ≥ 1B token 的端到端 run，loss
趋势合理（不要求 matched 对照）。

**S1-13 一次性用户批准例外（2026-07-16）**：用户明确要求以 job
`2394870.opbs` 的当前部分观测关闭 Stage 1。该 run 不得表述为完成 ≥1B token，原
token/cycle/ratio/scheduler/package gate 均保留为未通过；只允许结合已通过的 9N
baseline、完整五节点同 profile smoke，以及落盘的有限正 loss、连续 step、负稳健
slope 和预算内 projected runtime，支持“真实训练已发生且当前趋势/时间合理”这一
较窄结论。例外仅适用于本次 S1-13 closure，不得复用于后续 stage 或声称原 1B
evidence 已完成。权威记录为 `reports/stage1/S1-13-early-close-adr.md` 与
`reports/stage1/S1-13-partial-formal-observation.json`。

---

## 6. Stage 2：异步管线与性能规格

- **PERF-01 无完整模型稳态传输**：bootstrap、evaluation、导出之外，正常数据面不得反复读写完整模型。
- **PERF-02 传输重叠**：GPU→CPU、CPU→FS、FS→CPU 应尽可能与 inner training 重叠；CPU→GPU 采用只在安全 step boundary 产生有限暂停。
- **PERF-03 有界 staging**：learner 与 syncer 的 staging 内存由少量 fragment 大小决定，不由完整模型或全部 learner fragments 决定。
- **PERF-04 单次成本不随历史增长**：固定 `M`、`F`、fragment size、`S_max`、quorum 下，第 10000 次 update 的发现、选择、读取、提交复杂度不因历史而系统性增加（长跑证据在 RESEARCH_PLAN Stage 5 收集）。
- **PERF-05 队列不发散**：pending uploads、ready fragments、可见旧 proposals、FS 临时 payload、GC backlog 不得持续单调增长。
- **PERF-06 profile-first 门槛**：任何"需要多 syncer、CHFS、压缩或更复杂调度"的主张必须先证明单 syncer 的具体瓶颈位于 CPU merge / FS 吞吐 / metadata discovery / publication / adoption / 资源干扰之一。
- **PERF-07 延迟注入能力**：存储访问层必须支持可配置的人为可见性延迟注入（作用于发布→可见路径），供本阶段慢 FS 压力测试与 RESEARCH_PLAN Stage 5 延迟扫描使用；注入不得改变正确性语义。

**验收**：

| ID | 必须证明的事实 | 最小证据 |
|---|---|---|
| A-PERF-02 | pending queue 有界 | 慢 FS（限速至 1/10）注入 ≥ 2h，队列与临时对象有界，learner 吞吐不受影响 |
| A-PERF-04 | GPU goodput ≥ 95%（对照关闭通信的本地吞吐；160M 与 0.5–1B 各一次） | 吞吐对照报告（测量 run 无人为 pacing、分母与分子同设置——DISC-04，SP3） |
| A-PERF-05 | 快照+采用暂停合计 ≤ 2% 步时间 | timing trace |
| A-PERF-06 | 端到端延迟可分解 | 快照/写/发现/读/采用分解报告，**必须含 syncer duty cycle 与 per-fragment update interval**（单 syncer 容量哨兵，SP4） |

---

## 7. Stage 3：崩溃一致性与恢复规格

本阶段验证 FS 承载权威状态带来的 crash-consistent restart。范围为 kill-restart 级别；进程重启由运行环境/调度器负责，本文只规定重启后的行为。

- **REC-01 learner 恢复**：重启后从存储取得完整当前 global model 与版本向量；inner optimizer 允许重新初始化（低频本地保存 inner state 为可选优化，非验收项）；per-fragment counters 归零；proposal sequence 必须恢复单调性（持久化或以安全跳变重建，不得复用旧 sequence）；继续训练。
- **REC-02 syncer 恢复**：重启后读取各 fragment current 引用即恢复全部权威状态（参数、outer state、version、consumed frontier）；忽略 base 不匹配的 proposals；继续等待当前版本 quorum。不得要求 replay 或历史扫描（FS-10）。
- **REC-03 发布中断原子性**：publication 任意点崩溃后，系统只能处于两种状态之一——旧 fragment state 仍为 current，或新 parameters + 新 outer state + 新 version + 新 consumed frontier 整体成为 current（GLOBAL-02/INV-03 的崩溃扩展）；残留临时对象由 GC 回收（FS-09）。
- **REC-04 可用性边界**：存活且合格的 learner 数少于 `Q` 时，该 fragment 停止前进——这是算法可用性边界，不是错误；其他 fragments 不受影响；learner 恢复后无需人工干预自动继续。
- **REC-05 范围限制**：不做自动进程重启/failover、不做 exact replay、不承诺 inner optimizer moments 跨重启保留（其影响属训练质量 ablation）。
- **REC-06 持久性范围决定（新增，SP5；Stage 3 ORIENT 必答）**：当前发布路径对
  进程级 kill -9 完备；对节点断电/内核崩溃不完备（可见性记录可能先于 payload
  持久化——读侧 fail-closed 保证不消费损坏状态，安全性成立，但可用性恢复需
  人工介入）。Stage 3 ORIENT 必须以 ADR 三选一并使 C3 论文措辞与实现一致：
  (a) C3 主张明确限定进程级 crash（论文 limitations 同步记载断电边界）；
  (b) 增加可配置 durable-publish 模式（payload 与可见性记录的 fsync 链），并
  量化其发布延迟代价作为论文数据点；
  (c) 实现 FS-06 可选"上一完整 current 记录"的自动回退链并纳入注入矩阵。
  选 (b) 或 (c) 时相应扩展 A-REC-03 的注入矩阵。

**验收**：

| ID | 必须证明的事实 | 最小证据 |
|---|---|---|
| A-REC-01 | syncer kill-restart 无损继续 | kill -9 → 重启 → 下一次成功 update ≤ 2× 正常间隔；10 次不同时机注入全过 |
| A-REC-02 | learner kill-restart 无损继续 | 重启 ≤ 5 分钟恢复训练；loss 曲线除该 learner 贡献暂缺外无跳变 |
| A-REC-03 | 无半提交状态 | 发布中断注入矩阵（payload 写入中 / 可见性替换前 / 替换后）全部只出现两种合法结果 |
| A-REC-04 | quorum 不足自动停、恢复自动续 | 场景 trace |

---

## 8. Stage 4：stale-aware 规格

运行配置：默认 `S_max = 1`、`lambda_s = 1.0`、`Q_fresh >= 1`。得益于 DISC-02，本阶段新增实现收敛为：base 保留与 GC、旧 base displacement 重建、fresh anchor 与选择规则、拒绝原因分类。

**进入条件（新增，SP2）**：除 RESEARCH_PLAN Stage 3 通过外，A-SIM-04（S0A-03
全矩阵归因）必须完成；其决策记录决定本阶段消融矩阵规模（见下）。

研究定位：Decoupled DiLoCo 原论文避免将旧参数版本上的 stale gradient 直接应用于新版本；本阶段的有界 stale 接受是本项目的算法扩展，必须始终保留 fresh-only（`S_max=0`）对照并单独报告。

- **STALE-02 支持的 S_max 值**：必须支持 `S_max = 0`（fresh-only reference）与 `S_max = 1`（stale-aware target，默认）；`S_max > 1` 在完成 `S_max = 1` 验收前不属必需路径。
- **STALE-03 base 保留**：syncer 可用的 global fragment parameter history 至少覆盖当前版本与前 `S_max` 个版本（固定窗口，DISC-02 要素 5）；超窗 proposal 即使 metadata 完整也必须拒绝；超窗 payload 由 GC 异步回收（FS-09）。
- **STALE-04 正确 displacement**：`g_{i,f} = G_f^b - L_{i,f}`；不得使用 `G_f^v - L_{i,f}`（INV-07；Stage 0-C 的 s=1 golden cases 在此启用为运行时测试）。**启用为运行时测试时必须覆盖生产 merge 后端，不得仅覆盖参考实现路径（SP9）。**
- **STALE-06 fresh anchor**：默认 profile 要求 `Q_fresh >= 1`——每次 update 至少一个 fresh contributor；stale 可计入总 quorum 但不得单独推动版本连续前进。`Q_fresh = 0` 只允许作为明确标记的算法实验。
- **STALE-07 候选选择顺序**：达到 quorum 并结束 grace 后，若候选超过最大参与数，按确定性优先级：更低 staleness → 更大有效 token 数 → 更新 sequence → 固定 learner identity 序。同一 learner 最多一次。
- **STALE-08 未选择 proposal**：未选择不算已消费；只要 sequence 未消费、base history 仍在、staleness 未超限、payload 仍是当前可见 proposal，可在后续 update 中被选择。一旦某 proposal 被成功提交，该 learner 基于相同 base 的所有后续 proposal 失去资格（PROP-06）；须先采用更新的 base 才能再次贡献。
- **STALE-09 readiness**：同时满足 eligible distinct learners ≥ `Q` 且 fresh distinct learners ≥ `Q_fresh`；quorum 不按文件数、sequence 数或 token 数计算。
- **STALE-10 local-progress 上限**：必须支持配置 local-progress 上限，拒绝基于旧 base 训练过长、displacement 异常大的 proposal；每次拒绝记录明确 rejection reason 并计入 telemetry（分类至少含：too-stale、missing-base、future-base、base-identity-mismatch、consumed-base、over-progress-cap）。

**消融配置要求（SP2/SP10 修订）**：所有消融两臂共享同一二进制（DISC-03），
质量口径 matched-token 且质量臂 **from-scratch 初始化**（RESEARCH_PLAN §2.1）。
矩阵规模由 A-SIM-04 决策记录决定：

| 情形 | 矩阵 |
|---|---|
| 归因判定存在挽回 ≥3pp 有效区 | 全轴保留：`S_max ∈ {0,1}`、`lambda_s ∈ {0.5,1,2}`、`Q_fresh ∈ {1,0}`、异构比 `∈ {1.0,1.5,2,3}`；Profile B 锚点经 ADR 迁移至有效区 |
| 不存在（当前预期情形） | 裁剪为：`S_max ∈ {0,1}` 主对照 + `lambda_s = 1` + 异构比 `∈ {1.5, 3}`；`Q_fresh` 轴与其余 `lambda_s` 档取消；多种子预算全部用于 H3b |

**Profile B（Decoupled Stale-Aware）**：固定 `M`；`2 <= Q < M`；`Q_fresh >= 1`；`S_max = 1`；有限固定 grace window；token × inverse-staleness weighting；direct averaging merge；注入 learner 速度差异使 stale proposal 实际出现并被接受。全部超参数 run 前冻结。

**验收**：

| ID | 必须证明的事实 | 最小证据 |
|---|---|---|
| A-STALE-01 | `s=0` 等价 fresh reference | golden numeric trace |
| A-STALE-02 | `s=1` 使用 old base displacement | 反例向量 + 运行时 trace（**必须覆盖生产 merge 后端**，SP9） |
| A-STALE-03 | `s > S_max` 被拒绝 | boundary matrix（0/1/S_max/S_max+1） |
| A-STALE-04 | missing/wrong base identity 被拒绝 | content mismatch test |
| A-STALE-05 | weights 符合 `tokens/(1+λs)` 并归一化 | property test |
| A-STALE-06 | 默认 update 至少一个 fresh contributor | quorum trace |
| A-STALE-07 | stale 接受率与挽回幅度与仿真预测一致（H3a，SP1） | 接受率 > 5% 且与 Stage 0-A 预测偏差 ≤ 2×；实测挽回与预测差 ≤ max(2×\|预测\|, 1.0pp)（等价带；原"≥ 预测的 50%"在预测为负时为空条件，废止） |
| A-PROP-04 | 同 learner 同 base trajectory 不被更新 sequence 重复消费 | same-base replacement trace |
| A-PROP-05 | 延迟完成的旧 publication 不使 latest 倒退 | reordered completion trace |
| A-ALG-02 | stale-aware 与 fresh-only 有独立 matched 结果 | matched-token 实验报告（H3a/H3b 检验见 RESEARCH_PLAN Stage 4；质量臂 from-scratch，SP10） |

**质量 gate 与回退（SP1 修订）**：H3b matched-token 质量无损检验（阈值见
RESEARCH_PLAN Stage 4）失败时，默认运行配置回退 `S_max = 0`；机制与测试保留，
结果作为负面结果报告。无论 EQ3 结果如何，stale 结论报告必须包含 A-SIM-04 的
丢弃归因分解（REPORT-01）。

---

## 9. Loop Engineering 执行协议

每个最小工作单元遵循 `ORIENT → SPECIFY/RED → IMPLEMENT/GREEN → HARDEN → CHECK → PERSIST`。

- **ORIENT**：确认当前规格版本、当前 Stage 与未通过 acceptance IDs、最近实验结果与 blockers、当前环境/模型/数据/存储能力、未决算法变更。
- **SPECIFY/RED**：实现前先建立能在缺失或错误实现上失败的证据（golden transition、fragment-map 反例、stale-base 错误重建反例、partial publication 可见性测试、bounded-storage 断言、learner 非阻塞 timing trace、单 learner 重复计票反例等）。测试必须证明需求，不以行覆盖率替代。
- **IMPLEMENT/GREEN**：每次只解决当前最小失败差距。同一 loop 不得同时引入新 fragment 语义、新 stale policy、新 merge、新 outer optimizer、新发布模型、新拓扑；算法变化与系统优化必须可分别开关比较。
- **HARDEN**：Stage 1–4 合计至少覆盖：写入中 proposal/global fragment 不可见；duplicate polling 不重复消费；latest 覆盖；future base；missing base；staleness 边界 0/1/S_max/S_max+1；mixed fresh/stale quorum；base identity mismatch；tied weights；极大 embedding；慢 FS backpressure；采用时存在 in-flight stale proposal；kill-restart 注入矩阵；长运行存储与 latency 有界。
- **CHECK**：Maker 与 Checker 使用不同检查上下文。Checker 至少：从不变量反推遗漏；提供 Maker 未列出的反例；比较 reference math 与实际结果；检查存储中是否存在无界历史依赖；检查 stale update 是否使用真实 base；检查性能 claim 是否匹配拓扑与预算；输出 `PASS` / `PASS_WITH_FOLLOWUPS` / `BLOCKED`。
- **PERSIST**：持久记录已满足 acceptance IDs、失败命令与原始输出、关键 run 配置、证据位置、设计决策与 blockers、下一个最小 failing gap。持久化的是工程进度与研究证据，不是 runtime 提交历史。

---

## 10. Telemetry 与研究报告要求

telemetry 为 best-effort（INV-10），但验收必须能收集以下指标。

- **TEL-01 learner 指标**：inner step latency/throughput；processed tokens；每 fragment GPU→CPU 时间；CPU→FS 写入时间；FS→CPU 读取时间；CPU→GPU 采用时间；snapshot skip/replacement 次数；pending upload 数；fragment version lag；采用前额外 local steps。
- **TEL-02 syncer 指标**：proposal discovery latency；quorum wait；grace wait；selected learner count；fresh/stale count；staleness histogram；normalized weights；rejection reason counts（按 STALE-10 分类）；payload read bytes/time；merge time；outer optimizer time；publication time；per-fragment update interval；**syncer duty cycle（update 执行时间 ÷ 观察区间，SP4）**；ready queue age。
- **TEL-03 storage 指标**：live payload bytes；proposal slots；保留 base versions 数；metadata operations；orphan/temp bytes；GC reclaimed bytes；可见性延迟。
- **TEL-04 算法指标**：training loss；displacement norm（fresh 与各 staleness bucket 分列）；各 merge policy norm（启用对照时）；fragment-specific update norm；evaluation snapshot version vector；matched-token/compute/communication 对照标识。
- **REPORT-01 stale 结论必报量**：`S_max`、`lambda_s`、`Q`、`Q_fresh`、grace window、accepted staleness distribution、fresh/stale normalized weight mass、rejected-too-stale 比例、**丢弃按原因分解（含 latest-wins 覆盖，A-SIM-04 口径，SP2）**、learner adoption lag、fragment update frequency、total bytes/token、tokens/FLOPs/wall-clock 预算、fresh-only 对照、direct averaging 对照、loss 与 evaluation、GPU goodput 与 syncer/存储成本。
- **REPORT-02 收益不得混淆**：不得把"stale 提高 quorum/资源利用率"与"更高总通信量或更多 outer updates 带来的质量变化"混为一谈；效率比较必须 matched-compute 或 matched-communication，质量比较必须 matched-token。

---

## 11. Stage 0–4 验收矩阵（汇总）

| Stage | 验收 IDs |
|---|---|
| 0-A | A-SIM-01..03（已关闭）；A-SIM-04（本修订新增，未完成——Stage 4 进入条件） |
| 0-B | A-BENCH-01..04（已关闭；BENCH-05 未执行，义务移至 RESEARCH_PLAN Stage 5） |
| 0-C | A-ALG-00（已关闭） |
| 1 | A-FRAG-01..03、A-PROP-01..03、A-GLOBAL-01..02、A-LEARN-01..03、A-PERF-01、A-ALG-01、A-EVAL-01（已按 §5.11 例外关闭；≥1B run 义务未清偿） |
| 2 | A-PERF-02、A-PERF-04..06 |
| 3 | A-REC-01..04（REC-06 的 ADR 为 ORIENT 交付物，不新增验收 ID；选 (b)/(c) 时扩展 A-REC-03） |
| 4 | A-STALE-01..07、A-PROP-04..05、A-ALG-02 |

跨 Stage 设计性质：A-PERF-03（第 N 次 update 不依赖历史扫描；由 PERF-04/FS-05/FS-10 保证，1000+ updates 长跑证据在 RESEARCH_PLAN Stage 5 收集）。

每个 Stage 只有其全部验收 IDs 具备自动化或可重复证据、且 Checker 出具 `PASS` 或不影响主结论的 `PASS_WITH_FOLLOWUPS` 时才算关闭。

---

## 12. 设计决策冻结表

| 项目 | 决定 |
|---|---|
| 模型范围 | dense decoder-only Transformer causal LM |
| 逻辑层 | embedding、完整 Transformer blocks、lm_head |
| 零散参数 | 并入相邻较小逻辑层，平局归前侧 |
| fragment | 相邻完整层、连续、非空、按同步字节均衡（layer-aligned；balanced-tensor 为 Stage 5 消融） |
| fragment frequency | 统一间隔 `H`，错开 offsets |
| learner payload | 本地完整 fragment 参数快照 |
| global state | per-fragment parameters + outer state + version |
| proposal discovery | 每 learner/fragment latest-wins |
| quorum | distinct learners |
| stale 支持 | 必选特性；接口 Stage 1 落地（S_max=0 运行），Stage 4 启用 S_max=1 |
| stale displacement | `G_base - L_local` |
| stale weight | `tokens/(1+lambda_s×s)`，默认 `lambda_s=1` |
| fresh anchor | 默认 `Q_fresh>=1` |
| baseline merge | weighted direct averaging（接口可插拔，RDA 对照属 Stage 5） |
| outer optimizer | fragment-wise momentum/Nesterov SGD 主线 |
| learner adoption | latest-only、整 fragment 覆盖、保留 inner moments |
| 存储状态 | bounded current/base/latest，不保存完整历史 |
| 存储介质 | 仅依赖"原子可见性发布 + 最终可读"两个原语（STOR-01） |
| syncer | 一个逻辑 CPU syncer（Phase 2 扩展边界见 §1.4） |
| 故障恢复 | kill-restart 级别纳入 Stage 3 验收；断电级范围由 REC-06 ADR 裁决；不做 exact replay 与自动 failover |
| telemetry | best-effort，不是 authority |
| 测量纪律 | 验收测量不得含人为 pacing；阈值须有微基准依据、余量 ≥10%；wall-clock 单位口径（DISC-04/05） |

---

## 13. 变更记录

- **v2.2-draft-claude（2026-07-16）**：基于 Stage 0/1 证据的修订草案（修订点
  SP1–SP10 见 reflection/claude/06；与 RESEARCH_PLAN v1.4 草案配套采纳）。
  新增 DISC-04/05（测量真实性、阈值与单位纪律）；新增 SIM-06 / A-SIM-04
  （S0A-03 全矩阵归因，Stage 4 进入条件），§8 消融矩阵改为其决策两分支；
  A-STALE-07 增加挽回等价带并标注 H3a，质量 gate 改引 H3b，消融质量臂
  from-scratch；STALE-04 / A-STALE-02 要求 s=1 golden cases 覆盖生产 merge
  后端；新增 REC-06（断电级持久性范围 ADR，Stage 3 ORIENT 必答）；TEL-02 与
  A-PERF-06 增加 syncer duty cycle；A-BENCH-04 附单位口径更正注记；BENCH-05
  未执行事实与义务转移入文；已关闭 Stage 增加状态行；§1.4 Phase 2 增加初步
  信号注记；REPORT-01 增加丢弃按原因分解。协议不变量（INV-01..10、STOR-01）、
  DISC-01..03、§5 全部协议条款与 S1-13 例外文本均未改变。
- **v2.1（2026-07-16）**：按用户显式指令加入一次性 S1-13 closure 例外。保留
  job `2394870.opbs` 原 formal contract 的未通过事实，不声称完成 1B token；允许以
  已通过的 9N/五节点 smoke 与该 job 的部分 loss/runtime 观测支持较窄的 Stage 1
  “训练已发生”结论。算法、协议不变量、Stage 1 acceptance IDs 与后续 stage gate
  均未改变。
- **v2.0（2026-07-14）**：取代 `FS_BASED_DECOUPLED_DILOCO_V1_REQUIREMENTS_AND_DESIGN_SPEC.md`，按 RESEARCH_PLAN v1.1 重组为 Stage 0–4 契约。主要变化：新增 Stage 0-A 仿真规格（§2）、Stage 0-B 存储微基准规格（§3，原 FS-08 升级并给出阈值）、Stage 3 崩溃一致性与恢复规格（§7，原"故障恢复不属于验收"决定被推翻）；新增实现纪律 DISC-01..03（接口通用、行为先收窄）与存储原语抽象 STOR-01；stale 支持改为必选特性（Stage 1 落地接口、Stage 4 启用）；RDA 对照（原 OPT-03）与长跑/多 syncer 内容移出本文范围（RESEARCH_PLAN Stage 5 / Phase 2）；新增 PERF-07 可见性延迟注入能力。
