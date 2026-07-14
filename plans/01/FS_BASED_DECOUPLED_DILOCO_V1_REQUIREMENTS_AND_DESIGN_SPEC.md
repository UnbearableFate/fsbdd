---
title: FS-Based Decoupled DiLoCo v1 Requirements and Design Specification
version: 0.1-draft
date: 2026-07-13
status: Review Draft
repository: https://github.com/UnbearableFate/fs_based_decoupled_diloco
scope: Phase 1 single logical syncer with bounded stale-fragment support; Phase 2 boundary for distributed non-overlapping syncers
---

# FS-Based Decoupled DiLoCo v1：需求与设计规格

## 0. 文档定位

本文定义全新、从零实现的 `fs_based_decoupled_diloco` 的第一版需求、算法语义、共享文件系统交互语义、性能边界、验收证据和阶段切换条件。

本文是**规范性文档**，不规定：

- 仓库目录；
- 模块、类或函数名称；
- 命令行接口形式；
- 具体文件路径；
- 具体序列化库；
- 具体线程、进程或异步框架。

后续实现计划必须服从本文。若实现过程中发现本文无法满足，必须先记录设计变更及其证据，再修改规范；不得通过隐式行为改变算法语义。

本文中的术语：

- **必须**：第一阶段验收不可缺少；
- **应当**：除非有实测证据支持偏离，否则必须遵守；
- **可以**：允许的实现或实验选择；
- **不得**：违反后即不满足本规格。

---

## 1. 项目愿景

本项目要验证一个尽量简单的研究命题：

> 多个独立 GPU learner 能否只通过共享文件系统异步交换分层模型片段，由轻量 CPU syncer 执行 fragment-wise outer optimization，在不建立分布式数据库、事务日志、完整审计链或复杂故障协调协议的情况下，实现有效且高效的 Decoupled DiLoCo 训练。

第一版优先级按以下顺序排列：

1. 算法语义明确且可验证；
2. learner 不因其他 learner 或 syncer 的进度而阻塞；
3. 稳态只传输 fragment，不反复传输或物化完整模型；
4. 文件系统状态和发现成本有界；
5. 支持有界 stale fragment proposal；
6. 能通过实验解释训练质量、staleness 与系统性能之间的关系；
7. 为第二阶段的无重叠分布式 syncer 保留自然扩展边界。

第一版不是通用分布式数据库，也不是精确重放系统。

---

## 2. 研究边界与阶段划分

### 2.1 第一阶段范围

第一阶段拓扑为：

- 固定数量的 learner；
- 每个 learner 暂定为 1 GPU、1 node；
- 一个逻辑 CPU-only syncer，暂定为 1 CPU node；
- 所有节点访问同一个共享 POSIX 文件系统；
- 每个 learner 持有完整本地模型和独立 inner optimizer；
- syncer 按 fragment 聚合 learner proposal，并维护每个 fragment 的 global parameters、outer optimizer state 和版本；
- 第一阶段必须支持 fresh proposal 和有界 stale proposal；
- 第一阶段使用固定 membership，不支持运行中动态加入、退出或重新编号。

### 2.2 第二阶段范围

第二阶段增加多个 syncer，但必须满足：

- fragment ownership 静态或显式配置；
- 不同 syncer 负责互不重叠的 fragment 集合；
- 任一 fragment 在任一时刻只有一个逻辑 writer；
- 不做 active-active、backup、hedging 或 RAID 式重复计算；
- 不做 syncer failover；
- 多 syncer 结果必须与单 syncer 在相同输入选择和数值策略下等价。

第二阶段不属于本文第一阶段验收，但第一阶段的 global state、proposal 和 fragment 独立性不得阻碍该扩展。

### 2.3 第一阶段明确不做

第一阶段不包含：

- overlapping syncer ownership；
- syncer 选主、lease、fencing 或自动 failover；
- dynamic membership；
- complete event sourcing；
- deterministic replay；
- 完整提交历史或 prefix replay；
- distributed checkpoint；
- learner 精确恢复；
- active-active global writer；
- 分布式数据库或日志服务；
- 无界 proposal 队列；
- sub-tensor fragmentation；
- MoE、encoder-decoder、多模态模型的通用支持；
- tensor parallel、pipeline parallel、FSDP 或 ZeRO 与本协议的组合；
- 通信量化、稀疏化或压缩；
- 动态 layer importance 调度。

共享文件系统上的“先完成 payload、后宣布可见”属于正常并发读写的最低要求，不视为复杂容错功能。

---

## 3. 符号与核心概念

| 符号 | 含义 |
|---|---|
| `M` | learner 总数 |
| `L` | 非空逻辑层数量 |
| `F` | fragment 数量 |
| `f` | fragment 标识，`0 <= f < F` |
| `H_f` | learner 对 fragment `f` 的计划同步间隔 |
| `v_f` | fragment `f` 当前 global version |
| `G_f^v` | fragment `f` 在 global version `v` 的参数 |
| `O_f^v` | fragment `f` 在 version `v` 时对应的 outer optimizer state |
| `L_{i,f}` | learner `i` 发布的 fragment `f` 本地参数快照 |
| `b_{i,f}` | 该 proposal 所基于的 global fragment version |
| `s_{i,f}` | proposal staleness，定义为 `v_f - b_{i,f}` |
| `Q` | 一次 fragment outer update 所需的最小 distinct learner 数 |
| `Q_fresh` | 一次 update 所需的最小 fresh learner 数 |
| `S_max` | 可接受的最大 fragment-version staleness |

本文中“stale fragment”严格指：

> learner 发布的某个 fragment proposal 所声明的 base global fragment version 小于 syncer 当前 global fragment version。

它不是损坏文件，也不是“当前 global fragment 不够新”。

---

## 4. 系统级不变量

第一阶段所有实现和测试必须证明以下不变量。

### INV-01：learner 独立前进

一个 learner 的 inner training 不得等待：

- 其他 learner；
- quorum；
- grace window；
- syncer 完成 outer step；
- 完整 global model 的统一版本。

### INV-02：fragment 单写者

每个 fragment 的 global state 在第一阶段只能由唯一逻辑 syncer推进。

### INV-03：fragment 原子可见

对某个 fragment version，reader 只能看到：

- 旧的完整 global fragment state；或
- 新的完整 global fragment state。

不得看到新 parameters 与旧 outer optimizer state 的组合，也不得看到写入一半的 payload。

### INV-04：proposal 完整可见

syncer 不得读取尚未完整发布的 learner proposal。

### INV-05：一次性消费

同一个 proposal 最多参与一次成功的 outer update；一次 update 中同一个 learner 最多贡献一个 proposal。

### INV-06：有界 staleness

只有满足 `0 <= s <= S_max` 的 proposal 才能进入候选集。未来版本、缺少 base、超出 staleness 上限或 base identity 不匹配的 proposal 必须被拒绝。

### INV-07：stale update 基于其真实 base

stale proposal 的 outer displacement 必须由它实际基于的 `G_f^b` 与本地 `L_{i,f}` 计算。不得直接用当前 `G_f^v` 减去 stale local fragment 来伪造 fresh update。

### INV-08：状态增长有界

正常训练中的权威状态、proposal discovery 成本和单次 update 成本不得随历史 outer update 数线性增长。

### INV-09：mixed-version model 合法

learner 和当前 global model 都可以由不同版本的 fragments 组成。不同 fragment 不需要共享统一 version。

### INV-10：telemetry 不参与正确性

metrics、JSONL、日志、可视化和实验报告不得决定某个 proposal 是否已提交或某个 global state 是否有效。

---

## 5. 支持的模型范围

### MODEL-01：第一阶段模型类型

第一阶段面向可识别以下结构的 dense decoder-only Transformer causal LM：

1. 输入 embedding；
2. 有序 Transformer blocks；
3. 可选的 block 外 normalization 或少量辅助可训练参数；
4. `lm_head`。

### MODEL-02：逻辑层定义

逻辑层必须按模型前向顺序定义为：

1. input embedding 逻辑层；
2. 每个完整 Transformer block 各一个逻辑层；
3. `lm_head` 逻辑层。

一个完整 Transformer block 内的以下内容默认属于同一逻辑层：

- attention projections，例如 `q_proj`、`k_proj`、`v_proj`、`o_proj`；
- MLP projections；
- block 内 LayerNorm 或 RMSNorm；
- block 内 bias；
- 其他由该 block 独占的可训练参数。

参数名形如 `layers.<index>.*` 时，逻辑层边界是 `layers.<index>`，不是内部 projection。

### MODEL-03：零散可训练参数归属

不属于 embedding、任一 Transformer block 或 lm_head 的少量可训练参数，例如 final norm，必须归入相邻逻辑层。

归属规则必须确定性：

1. 若参数位于两个逻辑层之间，比较左右逻辑层当前唯一参数的同步字节数，归入较小的一侧；
2. 若大小相同，归入前一侧；
3. 若仅有一个相邻逻辑层，归入该层；
4. 完成归属后，整个 run 中不得改变。

### MODEL-04：共享参数

每个底层参数只能有一个同步 owner。

对于 input embedding 与 lm_head tied weights：

- 共享权重默认由 input embedding 逻辑层拥有；
- lm_head 仅拥有其独有参数；
- lm_head 若没有独有参数，不得因此产生重复 payload 或重复 outer optimizer state；
- 逻辑映射必须明确记录 tied identity。

### MODEL-05：参数覆盖

所有需要跨 learner 同步的可训练参数必须：

- 恰好属于一个逻辑层；
- 恰好属于一个 fragment；
- 不遗漏；
- 不重复。

可由配置重建的静态 buffer、mask、rotary cache 等不进入 fragment payload。若模型存在会影响训练语义的可变 buffer，必须在支持该模型前单独规定其归属。

### MODEL-06：歧义处理

当模型结构无法可靠识别 embedding、ordered Transformer blocks 或 lm_head 时，不得静默采用基于字符串猜测的 fragment map。必须要求一个显式、可检查的逻辑层映射，并应用相同的完整覆盖、共享参数和连续分片规则。

---

## 6. Layer-aligned fragment 划分

### FRAG-01：完整层边界

一个 fragment 包含一个或多个完整逻辑层。一个逻辑层不得拆分到多个 fragment。

### FRAG-02：连续性

fragment 必须由前向顺序上连续的逻辑层组成。

若逻辑层序列为：

`U_0, U_1, ..., U_(L-1)`，

则划分必须形成 `F` 个连续、非空区间。

### FRAG-03：数量约束

生产 fragment 模式要求：

`1 < F < L`。

开发和数值 oracle 可以使用 `F = 1`，但这不构成 fragment-mode 验收。

### FRAG-04：均衡度量

均衡目标使用**同步字节数**，而不是层数量。逻辑层大小定义为其唯一拥有参数在同步 dtype 下的总字节数。

### FRAG-05：划分目标

在“完整层、连续、非空”约束下，划分必须按以下优先级确定：

1. 最小化最大 fragment 同步字节数；
2. 在最大值相同的方案中，最小化各 fragment 相对平均大小的总偏差；
3. 仍并列时，使用固定的靠前切分规则。

结果必须在所有节点上确定性一致。

### FRAG-06：超大逻辑层

若 embedding、lm_head 或某个 Transformer block 大于理想平均 fragment 大小，可以单独形成 fragment。第一阶段不为追求均衡而拆分该层。

### FRAG-07：静态映射

fragment map 在一次 run 启动前冻结，并在以下角色间一致：

- 所有 learner；
- syncer；
- evaluation；
- 最终模型导出。

训练运行中不得重分片。

### FRAG-08：映射报告

每个 run 启动前必须产生可检查的映射摘要，至少包含：

- 逻辑层顺序；
- 每层唯一参数量和同步字节数；
- 零散参数归属；
- tied weight ownership；
- 每个 fragment 的层区间；
- 每个 fragment 的总字节数；
- 最大、最小和平均 fragment 大小；
- 最大 fragment / 平均 fragment 比例。

该报告用于验证，不是 runtime authority。

---

## 7. Learner 需求

### LEARN-01：本地状态

每个 learner 必须独立持有：

- 完整本地模型；
- 独立 inner optimizer；
- 独立数据流或数据 shard；
- 每个 fragment 当前采用的 global version；
- 每个 fragment 自上次采用 global update 后的 local steps 和 processed tokens；
- 每个 learner/fragment 的单调 proposal sequence。

### LEARN-02：连续 inner training

learner 必须持续执行 inner optimization。文件系统写入、syncer 聚合和其他 learner 的速度不得成为全局 barrier。

### LEARN-03：fragment 发布调度

每个 fragment 具有同步间隔 `H_f` 和 offset。

第一阶段验收配置中：

- 所有 fragment 使用同一同步间隔 `H`；
- offsets 应当把不同 fragment 的发布尽量均匀铺在 `H` 个 local steps 内；
- 调度应同时考虑 fragment 字节数，避免多个大 fragment 在同一步形成突发；
- 不同 learner 可以使用确定性的微小 offset 偏移，避免共享 FS 惊群，但不能破坏 quorum 的合理形成。

设计语义必须允许后续给不同 fragment 配置不同 `H_f`，但不同步频率不属于第一阶段完成条件。

### LEARN-04：快照边界

learner 只能在一个完整 inner optimizer step 完成后取得 fragment snapshot。不得在 forward、backward 或 optimizer mutation 中间捕获半更新状态。

### LEARN-05：GPU→CPU→FS 流程

稳态发布应满足：

1. 从 GPU 取得一个 fragment 的一致快照；
2. 转移到有限的 CPU staging 内存；
3. GPU 继续后续 inner training；
4. CPU 后台完成 FS publication。

第一阶段性能目标是把训练阻塞限制在必要的 snapshot/copy 安全边界，而不是让 GPU 等待整个 FS 写入。

### LEARN-06：有界 backpressure

每个 learner/fragment 的待发布状态必须有界。

当生成速度超过 FS 写入速度时：

- 不得建立无界队列；
- 同一 learner/fragment 同时最多有一个正式 publication in flight；
- 尚未开始正式 publication 的旧 snapshot 可以被更新 snapshot 替代；
- 已开始 publication 的 payload 必须保持不可变直到完成或被放弃；
- learner 可以跳过某次计划发布；
- learner 的 inner training 优先于保存所有中间 proposal。

### LEARN-07：proposal 内容

第一阶段 learner 发布本地 fragment 参数值，而不是要求通信压缩或只发布 delta。

每个 proposal 必须声明其 base global fragment identity，使 syncer 能在 fresh 和 stale 两种情况下重建正确的 outer displacement。

### LEARN-08：global fragment 采用

learner 发现某个 fragment 存在更新版本时，应：

1. 在后台取得最新完整版本；
2. 可以跳过未采用的中间版本；
3. 在完整 inner optimizer step 结束后的安全边界整体覆盖该 fragment；
4. 更新该 fragment 的 base version；
5. 将该 fragment 的 local-step 和 token counters 重置；
6. 继续训练，不等待其他 fragment。

### LEARN-09：inner optimizer state

第一阶段默认在采用 global fragment 时保留相应参数的 inner optimizer moments，只覆盖参数并重置 per-fragment progress counters。

“采用时重置 moments”可作为独立 ablation，但不得与默认结果混合报告。

### LEARN-10：mixed-version local model

learner 可以同时持有不同 global versions 的 fragments。任何逻辑不得要求所有 fragment 版本对齐后才能训练。

### LEARN-11：stale proposal 与采用并存

learner 可以在一个旧-base proposal 正在后台写入时采用更新的 global fragment。已开始发布的 proposal 继续保持原 base identity；它随后由 syncer 按 staleness 规则接受或拒绝，不得被就地改写成新 base。

---

## 8. Proposal 语义

### PROP-01：最小逻辑身份

一个完整 proposal 至少包含以下逻辑信息：

- run identity；
- model / fragment-map identity；
- learner identity；
- fragment identity；
- learner/fragment 单调 proposal sequence；
- base global fragment version；
- base global fragment content identity；
- local steps since base adoption；
- processed tokens since base adoption；
- snapshot 对应的 learner local step；
- payload dtype、shape、字节数和完整性信息；
- 本地 fragment payload identity。

本文不规定这些信息采用单一文件还是多个文件表示。

### PROP-02：完整发布

proposal payload 必须先完成，再由一个小型、可原子发布的可见性记录宣布完整。syncer 只能通过完整可见性记录发现 proposal。

### PROP-03：latest-wins discovery

每个 learner/fragment 对外只需要暴露一个最新完整 proposal。正常发现空间目标为 `O(M × F)`，不得要求扫描全部历史 proposal。

### PROP-04：不可变性与单调 latest

已完整发布的 proposal 内容和身份不得修改。learner 后续 proposal 使用更大的 sequence 并替换 latest 可见引用。latest 引用只能向更大的 sequence 前进；延迟完成的旧 publication 不得把 latest 倒退到更小 sequence。

### PROP-05：一次更新一个 learner 一票

一次 fragment outer update 中，同一 learner 最多有一个 proposal 被选择。quorum 按 distinct learner 数计算。

### PROP-06：固定大小消费状态

syncer 必须能判断一个 learner/fragment proposal 是否已被成功使用。每 learner/fragment 至少需要固定大小地记录最近已消费 proposal sequence 和最近已消费 base fragment version，不保存无界 consumed-ID 历史。

同一 learner 基于同一个 base fragment version 形成的累计 local trajectory 最多只能被成功消费一次。若一个更新 sequence 的 proposal 仍基于已经消费过的 base，它必须被拒绝；否则累计 displacement 会重复包含先前已经吸收的本地训练。

### PROP-07：基本资格

proposal 至少满足以下条件才可进入候选集：

- run、model 和 fragment map identity 匹配；
- fragment identity 匹配；
- shape、dtype 和 payload 完整性有效；
- proposal sequence 大于该 learner/fragment 最近已消费 sequence；
- proposal base version 大于该 learner/fragment 最近已消费 base version；
- local steps 和 tokens 为正；
- local progress 不超过配置的安全上限；
- base version 不在未来；
- base content identity 与 syncer 保留的对应 global fragment 一致；
- staleness 不超过 `S_max`。

---

## 9. 第一阶段的 bounded stale-fragment 语义

### 9.1 研究定位

Decoupled DiLoCo 论文的主要算法避免把在旧参数版本上计算的 stale gradient 直接应用到新版本。本项目第一阶段有意加入**有界 stale-fragment proposal**，因此它是 paper-faithful baseline 之外的算法扩展，必须始终保留 fresh-only 对照并单独报告。

### STALE-01：staleness 定义

选择时，fragment `f` 当前版本为 `v`，proposal base version 为 `b`：

`staleness s = v - b`。

- `s = 0`：fresh；
- `1 <= s <= S_max`：eligible stale；
- `s < 0`：future / invalid；
- `s > S_max`：too stale / reject。

### STALE-02：第一阶段默认上限

第一阶段必须支持：

- `S_max = 0`，fresh-only reference；
- `S_max = 1`，stale-aware target。

stale-aware 默认值为 `S_max = 1`。在完成 `S_max = 1` 的稳定性、训练质量和长运行验收前，不将 `S_max > 1` 作为必需路径。

### STALE-03：base fragment 保留

为计算 stale displacement，syncer 可用的 global fragment parameter history 必须至少覆盖：

- 当前版本；
- 前 `S_max` 个版本。

保留是固定窗口，不是完整历史。超出窗口的 proposal 即使 metadata 完整也必须拒绝。

### STALE-04：正确 displacement

对 learner `i` 的 fragment proposal，base 为 `b`、本地参数为 `L_{i,f}`，定义 per-learner outer pseudo-gradient：

`g_{i,f} = G_f^b - L_{i,f}`。

fresh proposal 是 `b = v` 的特例。

stale proposal 必须使用 `G_f^b`，不得错误使用：

`G_f^v - L_{i,f}`。

后者会把 base 到 current 的其他 global updates 混入该 learner 的本地贡献，产生无意的回拉或重复更新。

### STALE-05：staleness attenuation

第一阶段 stale-aware 基线使用以下未归一化权重：

`r_i = tokens_i / (1 + lambda_s × s_i)`，

其中：

- `tokens_i` 是自 learner 采用 base fragment 后处理的 token 数；
- `s_i` 是选择时的 fragment-version staleness；
- `lambda_s >= 0`；
- 第一阶段默认 `lambda_s = 1.0`。

归一化权重：

`w_i = r_i / sum_j(r_j)`。

要求：

- fresh proposal 不受 staleness 折扣；
- staleness 增大时权重单调不增；
- 所有最终权重非负且和为 1；
- 具体权重、staleness 和 tokens 必须进入实验 telemetry。

后续可以研究指数衰减、学习率缩放或 norm-aware staleness policy，但不得替换此基线而不保留对照。

### STALE-06：fresh anchor

stale-aware 默认 profile 要求：

`Q_fresh >= 1`。

即每次 fragment outer update 至少包含一个基于当前 global fragment version 的 fresh proposal。stale proposal 可以计入总 quorum，但不得在默认 profile 中单独推动 global fragment 连续前进。

`Q_fresh = 0` 只允许作为明确标记的算法实验，不作为第一阶段默认验收路径。

### STALE-07：候选选择顺序

达到 quorum 后，在允许的 grace window 内收集更多 eligible proposals。若候选超过最大参与数，选择优先级必须确定性：

1. 更低 staleness；
2. 更大的有效 token 数；
3. 更新的 proposal sequence；
4. 固定 learner identity 顺序作为最终 tie-break。

同一 learner 仍然最多贡献一次。

### STALE-08：未选择 proposal

未被选择的 proposal不视为已消费。只要它之后仍满足：

- sequence 未消费；
- base history 仍存在；
- staleness 未超过上限；
- payload 仍是该 learner/fragment 的当前可见 proposal；

它可以在后续 fragment update 中被选择。否则自然失效或被更新 proposal 覆盖。

一旦某个 proposal 被成功选择并提交，该 learner 基于相同 base version 的所有后续 proposal 都失去资格，即使它们具有更大的 sequence。learner 必须先采用一个更新的 global fragment base，才能再次为该 fragment 贡献。

### STALE-09：stale 与 quorum

一次 update 的 readiness 同时满足：

- eligible distinct learners 数量不少于 `Q`；
- fresh distinct learners 数量不少于 `Q_fresh`。

quorum 不按文件数量、proposal sequence 数量或 token 数量计算。

### STALE-10：stale 安全上限

除版本 staleness 外，第一阶段必须允许配置 local-progress 上限，防止一个 learner 基于同一旧 base 进行过长本地训练后仍以异常大 displacement 参与聚合。

超限 proposal 被拒绝，并记录明确 rejection reason。

---

## 10. Syncer 需求

### SYNC-01：单一逻辑 syncer

第一阶段只有一个逻辑 syncer，负责全部 fragments。运行环境必须避免同时启动两个可写 syncer；第一阶段不通过分布式选主解决误启动问题。

### SYNC-02：per-fragment 独立状态

syncer 对每个 fragment 独立维护：

- current global parameters；
- current fragment version；
- current outer optimizer state；
- outer update count；
- bounded base-parameter history；
- 每 learner 最近已消费 proposal sequence；
- 每 learner 最近已消费的 base fragment version；
- 当前 readiness / grace-window 状态。

不同 fragment 不共享一个必须原子推进的全局版本。

### SYNC-03：readiness 驱动

syncer 不按“CPU 执行了 n 次循环”机械聚合。fragment update 由以下事实驱动：

- fragment 当前版本；
- eligible distinct proposals；
- `Q` 与 `Q_fresh`；
- grace window；
- fragment 调度和公平性。

### SYNC-04：quorum 与 grace window

第一阶段必须支持：

- `Q = M`、grace 为 0 的可控 reference profile；
- `Q < M` 的 decoupled profile；
- 固定、有限的 grace window；
- grace 结束后使用当时所有 eligible、未消费且满足选择策略的 distinct learner proposals。

adaptive grace window 可在后续阶段加入，不是第一阶段必需条件。

### SYNC-05：fragment 公平性

当多个 fragments 同时 ready 时，syncer 必须使用确定性且无饥饿的调度规则。一个持续高频 ready 的 fragment 不得无限阻止其他 fragment 前进。

### SYNC-06：有界内存聚合

syncer 对一次 fragment update 的峰值工作内存目标应与：

- 当前 fragment；
- 该 fragment 的 outer optimizer state；
- 少量 streaming accumulator；

相关，而不与 `M × fragment size` 或完整模型大小成正比。

syncer 应能逐个读取 proposal 并进行 streaming reduction，不要求同时驻留所有 learner fragment。

### SYNC-07：不物化完整模型

稳态 outer update 不得要求 syncer 读取或写出完整模型。只有初始化、evaluation snapshot 或最终导出可以组装完整模型。

### SYNC-08：成功 update 的逻辑顺序

一次 fragment update 的逻辑过程必须是：

1. 固定 current fragment version 和完整 current state；
2. 读取并验证候选；
3. 固定 selected learner set、base identities、staleness 和 weights；
4. 重建每个 proposal 的 pseudo-gradient；
5. merge；
6. 使用 current global fragment 和 current outer state 执行 outer optimizer；
7. 完整发布 new parameters、new outer state 和 next version；
8. new current state 与新版本同时使 selected proposal sequences 及其 base versions 成为已消费；若 publication 未完成，则它们仍未消费；
9. 让旧 proposal 通过 sequence、consumed-base 或 staleness 规则自然失效。

### SYNC-09：选择稳定性

一次 update 开始执行后，selected set 和 weights 不因目录顺序、后到 proposal 或 wall-clock 变化而改变。后到 proposal 参与下一次 update。

---

## 11. Merge 与 outer optimization

### OPT-01：fresh 和 stale 共用同一 pseudo-gradient 定义

每个 learner contribution 均表示为：

`g_i = base_global_fragment - learner_local_fragment`。

区别仅在于 base version 和 staleness weight。

### OPT-02：direct weighted averaging baseline

第一阶段必须提供可作为数值 oracle 的 direct merge：

`g_merge = sum_i(w_i × g_i)`。

该路径用于：

- fresh-only reference；
- stale-aware reference；
- 与更复杂 merge 的数值比较；
- 小规模端到端验收。

### OPT-03：RDA 评估

在第一阶段关闭前，必须对 weighted Radial-Directional Averaging 进行独立验证和实验比较：

- 非 embedding fragments 至少完成 direct averaging 与 RDA 对照；
- embedding fragment 保留 direct averaging 基线；
- 生产默认 merge 只能在 matched-token、matched-compute 或 matched-communication 实验后冻结；
- 不得把 RDA 与 stale 支持同时引入而缺少 direct stale-aware oracle。

### OPT-04：outer optimizer

第一阶段主 outer optimizer 为带 momentum/Nesterov 的 SGD 类 outer optimizer，并为每个 fragment 独立维护状态。

同时必须保留以下简单控制：

- 无 momentum 的 SGD；
- learning rate 为 1 的 direct-averaging 等价控制。

### OPT-05：应用点

即使 proposal stale，merged pseudo-gradient 也作用于 current `G_f^v` 和 current `O_f^v`，生成 `G_f^(v+1)` 与 `O_f^(v+1)`。

不得恢复或回滚到旧 base 再执行 outer step。

### OPT-06：数值身份

一次 update 的数值身份至少由以下因素决定：

- current fragment version 和 content identity；
- selected proposal identities；
- 各 proposal base identities；
- tokens、staleness 和 normalized weights；
- merge policy；
- outer optimizer policy 和超参数；
- accumulation / compute dtype；
- fragment map identity。

这些信息用于实验复现和 checker，不要求形成无限历史 authority。

---

## 12. Global fragment 发布与 learner 采用

### GLOBAL-01：global model 定义

当前 global model 是所有 fragment current states 的集合：

`{(G_f^(v_f), O_f^(v_f), v_f) | f in [0, F)}`。

它可以是版本向量，不要求存在每步物化的完整 checkpoint。

### GLOBAL-02：per-fragment 发布单位

一次 global fragment publication 必须把以下内容作为一个逻辑整体宣布可见：

- new fragment parameters；
- new outer optimizer state；
- new fragment version；
- selected proposal sequence frontier 与 consumed base frontier；
- 必要的 policy identity 和完整性信息。

### GLOBAL-03：版本单调性

每次成功 publication 只将该 fragment version 增加 1。不得跳号、倒退或覆盖同一 version 的不同内容。

### GLOBAL-04：learner latest-only adoption

learner 可以从本地 version `v` 直接采用 global version `v+k`，不要求依次应用中间版本。采用后，本地 base identity 直接变为所取得的最新完整版本。

### GLOBAL-05：evaluation snapshot

evaluation 必须先冻结一个 fragment-version vector 和对应 content identities，再加载该固定组合。不得在 evaluation 加载过程中不断追逐变化中的 latest fragments。

### GLOBAL-06：bootstrap

新 run 初始化时必须产生完整、固定 fragment map 和每个 fragment 的 version 0 global parameters、outer state 与 identity。learner 在开始训练前取得一个完整 global version vector。

---

## 13. 共享文件系统契约

本节定义逻辑语义，不规定具体路径。

### FS-01：文件系统角色

共享 FS 仅承担：

- 大张量 payload 交换；
- latest proposal discovery；
- per-fragment current global state；
- bounded stale-base history；
- 少量 run / fragment-map metadata；
- best-effort telemetry。

它不是完整事件日志或分布式数据库。

### FS-02：payload-first、visibility-last

大 payload 必须在不可见状态下完成写入和完整性验证，再通过小型可原子发布的可见性记录暴露给 reader。

### FS-03：小型 current 引用

每个 fragment 应有一个小型 current 引用，能够把 reader 导向该 fragment 当前完整的 parameters、outer state 和 metadata。

参数和 outer state 不得通过彼此独立的“最新文件”推断。

### FS-04：per-fragment authority

第一阶段和第二阶段都使用 per-fragment current state 作为 fragment authority。不存在要求所有 fragments 一次共同提交的 global head。

### FS-05：固定 discovery 面

proposal discovery 的正常成本应与 `M × F` 成正比，不与历史 proposal 数或 outer update 数成正比。

### FS-06：bounded retention

权威或必要数据的稳态保留上限：

- 每个 fragment 当前 global state；
- 每个 fragment 前 `S_max` 个 base parameter versions；
- 每个 learner/fragment 最新完整 proposal；
- 少量写入中的临时对象；
- 可选的上一完整 current record 作为人工恢复辅助；
- 有限保留的 telemetry。

不要求保存全部旧 global versions、旧 proposals 或 loser attempts。

### FS-07：完整性

reader 至少验证：

- run identity；
- fragment-map identity；
- fragment identity；
- version / proposal sequence；
- dtype、shape 和预期字节数；
- payload 完整性标识；
- base content identity。

### FS-08：能力预检

在目标共享 FS 上运行训练前，必须验证：

- 大 payload 写入后跨节点可见；
- 小型 visibility/current 引用的原子替换语义；
- reader 不会把临时 payload 误认作完成对象；
- metadata 可见延迟在调度参数可接受范围内。

### FS-09：GC 非关键路径

临时文件、被 latest 覆盖的 proposal 和超出 base window 的 global payload 可以异步回收。GC 失败不得阻止训练正确性，只能造成暂时空间增长并产生告警。

### FS-10：无历史扫描恢复依赖

正常启动和稳态运行不得要求从第一个 update 开始扫描或 replay 历史。

---

## 14. 调度、全局进度与停止条件

### PROG-01：per-fragment update count

每个 fragment 独立维护 outer update count。

### PROG-02：global cycle

第一阶段定义：

`global_cycle = min_f(outer_update_count_f)`。

一个 global cycle 完成表示每个 fragment 至少又完成了一次 outer update。

### PROG-03：停止条件

主要训练停止条件应以以下之一冻结：

- 目标 global cycles；
- 目标 wall-clock / compute budget；
- 预注册的 token / FLOP 预算。

不得只以最快 learner 的 local step 作为全局停止条件。

### PROG-04：进度报告

必须分别报告：

- 各 learner local steps 和 processed tokens；
- 各 fragment outer update count；
- global cycle；
- 各 fragment accepted tokens；
- fresh / stale accepted contribution 数量。

不得把不同口径混称为同一个“global step”。

### PROG-05：learning-rate 进度语义

inner learning-rate schedule 必须明确依据 learner local steps 或 learner processed tokens 推进，并在所有实验中报告。

outer learning-rate schedule 必须按每个 fragment 自己的 outer update count 或预注册的 global-cycle 规则推进。第一阶段默认按 per-fragment outer update count 推进，避免一个 fragment 因等待 quorum 而错误跳过 scheduler 状态。

staleness attenuation 只改变 proposal merge weight；除非进入明确的独立实验，它不得隐式改变 outer learning rate。

### PROG-06：fragment 频率扩展边界

第一阶段使用统一 `H`。后续 layer-importance 实验可以使用 per-fragment `H_f`，但必须同时报告：

`sum_f(fragment_bytes_f / H_f)`，

用于区分“更聪明的通信预算分配”和“仅增加了总通信量”。

---

## 15. 性能与 backpressure 要求

### PERF-01：无完整模型稳态传输

bootstrap、evaluation 和最终导出之外，learner 与 syncer 的正常数据面不得反复写出或读取完整模型。

### PERF-02：传输重叠

GPU→CPU、CPU→FS、FS→CPU 应尽可能与 learner inner training 重叠。CPU→GPU 参数采用只在安全 step boundary 产生有限暂停。

### PERF-03：有界 staging

learner 和 syncer 的 staging 内存必须由少量 fragment 大小决定，而不是由完整模型或所有 learner fragment 总大小决定。

### PERF-04：单次成本不随历史增长

在固定 `M`、`F`、fragment size、`S_max` 和 quorum 下，第 10000 次 fragment update 的发现、选择、读取和提交复杂度不得因前 9999 次历史而系统性增加。

### PERF-05：队列不发散

长运行中以下 backlog 不得持续单调增长：

- learner pending uploads；
- syncer ready fragments；
- stale-but-still-visible proposals；
- FS 临时 payload；
- GC backlog。

### PERF-06：profile-first 门槛

任何声称“需要多个 syncer、CHFS、通信压缩或更复杂调度”的变更，必须先证明单 syncer 的具体瓶颈位于：

- CPU merge；
- FS read/write throughput；
- metadata discovery；
- fragment publication；
- learner adoption；
- 或资源干扰。

不得仅凭直觉扩张系统复杂度。

---

## 16. 最小 telemetry 与实验可解释性

telemetry 是 best-effort，但第一阶段验收必须能够收集以下指标。

### TEL-01：learner 指标

- inner step latency / throughput；
- processed tokens；
- 每 fragment GPU→CPU 时间；
- CPU→FS 写入时间；
- FS→CPU 读取时间；
- CPU→GPU 采用时间；
- snapshot skip / replacement 次数；
- pending upload 数；
- fragment version lag；
- 采用 global update 前额外执行的 local steps。

### TEL-02：syncer 指标

- proposal discovery latency；
- quorum wait；
- grace wait；
- selected learner count；
- fresh / stale count；
- staleness histogram；
- normalized weights；
- rejection reason counts；
- payload read bytes/time；
- merge time；
- outer optimizer time；
- publication time；
- per-fragment update interval；
- ready queue age。

### TEL-03：storage 指标

- 当前 live payload bytes；
- proposal slots 数；
- stale base versions 数；
- metadata operations；
- orphan / temp bytes；
- GC reclaimed bytes；
- FS 可见延迟。

### TEL-04：算法指标

- training loss；
- gradient / displacement norm；
- fresh 与各 staleness bucket 的 norm；
- direct averaging 与 RDA 的 merge norm；
- fragment-specific update norm；
- evaluation snapshot version vector；
- matched-token、matched-compute 和 matched-communication 对照标识。

---

## 17. 第一阶段验收 profiles

至少必须有以下两个正式 profile。

### Profile A：Fresh Reference

- 固定 `M`；
- `Q = M`；
- `Q_fresh = M`；
- `S_max = 0`；
- grace window 为 0；
- direct weighted averaging；
- 确定性 fragment schedule；
- 用于数值 oracle、同步基线和回归。

### Profile B：Decoupled Stale-Aware

- 固定 `M`；
- `2 <= Q < M`；
- `Q_fresh >= 1`；
- `S_max = 1`；
- 有限固定 grace window；
- token × inverse-staleness weighting；
- direct averaging 基线，并完成 RDA 对照；
- 人为注入 learner 速度差异，使 stale proposal 在正常运行中实际出现并被接受；
- 用于第一阶段主验收。

每个 profile 的所有超参数必须在 run 启动前冻结并进入实验记录。

---

## 18. Loop Engineering 执行协议

每个最小工作单元遵循：

`ORIENT → SPECIFY/RED → IMPLEMENT/GREEN → HARDEN → CHECK → PERSIST`

### 18.1 ORIENT

每次进入任务前必须确认：

- 当前规范版本；
- 当前完成阶段和未通过 acceptance IDs；
- 最近实验结果与 blockers；
- 当前环境、模型、数据和 FS 能力；
- 是否存在未决算法变更。

### 18.2 SPECIFY/RED

实现前先建立能在缺失或错误实现上失败的证据，例如：

- 标量或小向量 golden transition；
- fragment-map counterexample；
- stale-base 错误重建反例；
- partial publication 可见性测试；
- bounded-storage 长运行断言；
- learner 不应阻塞的 timing trace；
- 单 learner 重复计入 quorum 的反例。

测试必须证明需求，不以覆盖代码行为替代需求证据。

### 18.3 IMPLEMENT/GREEN

每次只解决当前最小失败差距。不得在同一 loop 中同时引入：

- 新 fragment 语义；
- 新 stale policy；
- 新 merge；
- 新 outer optimizer；
- 新 FS publication 模型；
- 新分布式拓扑。

算法变化与系统优化必须可以分别开关和比较。

### 18.4 HARDEN

第一阶段至少覆盖：

- 写入中的 proposal 不可见；
- 写入中的 global fragment 不可见；
- duplicate polling 不重复消费；
- latest proposal 覆盖；
- future base；
- missing base；
- staleness 边界 `0/1/S_max/S_max+1`；
- mixed fresh/stale quorum；
- base identity mismatch；
- tied weights；
- 极大 embedding；
- FS 慢导致的 backpressure；
- learner 采用时存在 in-flight stale proposal；
- 长运行存储和 latency 是否有界。

### 18.5 CHECK

Maker 与 Checker 必须使用不同检查上下文。Checker 至少：

1. 从本文不变量反推遗漏；
2. 提供一个 Maker 未列出的反例；
3. 比较 reference math 与实际结果；
4. 检查 FS 中是否存在无界历史依赖；
5. 检查 stale update 是否使用真实 base；
6. 检查性能 claim 是否采用匹配拓扑和预算；
7. 输出 `PASS`、`PASS_WITH_FOLLOWUPS` 或 `BLOCKED`。

### 18.6 PERSIST

每个 loop 必须持久记录：

- 已满足的 acceptance IDs；
- 失败命令和原始输出；
- 关键 run 配置；
- 证据位置；
- 设计决策和 blockers；
- 下一项最小 failing gap。

这里持久化的是工程进度和研究证据，不是 runtime 的完整提交历史。

---

## 19. 第一阶段渐进子阶段

### S1-00：研究契约与数学 oracle

目标：在真实 FS 和 GPU 流程之前冻结算法语义。

必须完成：

- fresh pseudo-gradient 的标量和向量 golden cases；
- stale `s=1` 的 base-relative displacement cases；
- inverse-staleness weighting；
- one-per-learner 与 consumed sequence；
- direct averaging；
- outer optimizer 的最小 reference transitions；
- requirement-to-evidence map。

退出条件：错误使用 `G^v - L_stale` 的实现必须被测试明确拒绝。

### S1-01：逻辑层识别与连续均衡分片

必须完成：

- embedding、Transformer blocks、lm_head 识别；
- block 内参数完整归属；
- final norm 等零散参数的邻侧小者规则；
- tied embedding/lm_head；
- 连续、非空、完整层划分；
- 最小化最大 fragment 字节数；
- 确定性映射摘要；
- 至少两种常见 decoder-only naming family 的验证。

退出条件：所有 trainable parameters 恰好一次覆盖，所有节点得到相同映射。

### S1-02：共享 FS 最小 publication 契约

必须完成：

- proposal payload-first / visibility-last；
- global fragment parameters + outer state 的共同可见性；
- per learner/fragment latest proposal；
- per-fragment current global state；
- bounded base history；
- 目标 FS 跨节点可见性和小型原子发布能力验证；
- 长运行不扫描历史。

退出条件：reader 永远不会把部分 payload 当成完整对象。

### S1-03：Fresh-only 单 syncer 端到端基线

必须完成：

- 多 learner 独立 inner training；
- 错开的 layer-fragment publication；
- distinct-learner quorum；
- direct weighted merge；
- fragment-wise outer optimizer；
- learner latest-only adoption；
- mixed-version local model；
- `Profile A` 通过。

退出条件：数值结果与 reference oracle 一致，且稳态不传完整模型。

### S1-04：Bounded stale-fragment 主线

该子阶段是第一阶段必需项，不是可选扩展。

必须完成：

- `S_max = 1`；
- current 与前一 base parameter retention；
- fresh + stale mixed quorum；
- `Q_fresh >= 1`；
- token / inverse-staleness weighting；
- stale proposal 一次性消费；
- too-stale、missing-base、future-base rejection；
- learner 速度差异下 stale proposal 实际被接受；
- fresh-only 与 stale-aware matched-budget 对照；
- `Profile B` 的数值和稳定性 gate。

退出条件：stale 支持不会退化为 raw stale-weight averaging，且能明确量化其质量和 goodput 影响。

### S1-05：异步传输与 backpressure

必须完成：

- fragment GPU→CPU→FS pipeline；
- global fragment FS→CPU→GPU pipeline；
- 安全 optimizer-step snapshot/adoption boundary；
- bounded staging；
- latest-wins pending snapshot；
- 慢 FS 条件下 learner queue 不发散；
- in-flight stale proposal 与新 global adoption 并存；
- learner GPU 不等待完整 FS round-trip。

退出条件：传输能与 inner training 有效重叠，并具有可解释的延迟分解。

### S1-06：Merge 与算法对照

必须完成：

- direct averaging 作为固定 oracle；
- weighted RDA 对照；
- embedding 与非 embedding 的 merge policy 比较；
- Nesterov outer optimizer 与简单控制；
- fresh-only、stale-aware 分开报告；
- matched-token / compute / communication 的至少一种正式对照。

退出条件：默认 merge 与 outer optimizer 由证据冻结，而不是由实现便利决定。

### S1-07：多节点长运行与第一阶段关闭

必须完成：

- 目标共享 FS 上的多节点运行；
- 至少一个能稳定产生 fresh/stale 混合的异构速度场景；
- 至少 1000 次 fragment outer updates 的 bounded-growth 验证；
- proposal discovery latency 不随历史增长；
- GPU goodput、FS bandwidth、syncer utilization 和 adoption lag 报告；
- 固定 evaluation snapshot；
- fresh-only 与 stale-aware loss / evaluation 对照；
- 独立 Checker 结论。

第一阶段只有在 S1-00 至 S1-07 全部通过后关闭。

---

## 20. 第一阶段 acceptance matrix

| ID | 必须证明的事实 | 最小证据 |
|---|---|---|
| A-FRAG-01 | 每个参数恰好属于一个 fragment | 参数 identity 全覆盖检查 |
| A-FRAG-02 | fragments 连续且非空 | 映射 checker |
| A-FRAG-03 | 最大 fragment 在约束下达到最优或可证明目标 | exhaustive 小模型 + 大模型结果摘要 |
| A-PROP-01 | 部分 proposal 不可见 | 中途读取反例 |
| A-PROP-02 | 同 learner 一次最多一个 contribution | duplicate proposal trace |
| A-PROP-03 | 同 proposal 不重复消费 | repeated polling trace |
| A-PROP-04 | 同 learner 的同一 base trajectory 不被以更新 sequence 重复消费 | same-base replacement trace |
| A-PROP-05 | 延迟完成的旧 publication 不能使 latest sequence 倒退 | reordered completion trace |
| A-STALE-01 | `s=0` 等价 fresh reference | golden numeric trace |
| A-STALE-02 | `s=1` 使用 old base displacement | 反例向量 |
| A-STALE-03 | `s>S_max` 被拒绝 | boundary matrix |
| A-STALE-04 | missing/wrong base identity 被拒绝 | content mismatch test |
| A-STALE-05 | weights 符合 tokens/(1+lambda×s) 并归一化 | property test |
| A-STALE-06 | 默认 update 至少一个 fresh contributor | quorum trace |
| A-GLOBAL-01 | params 与 outer state 共同可见 | publication interruption test |
| A-GLOBAL-02 | per-fragment version 单调 | transition trace |
| A-LEARN-01 | learner 不等待其他 learner/quorum | speed-heterogeneity timing trace |
| A-LEARN-02 | adoption 只覆盖目标 fragment | parameter identity check |
| A-LEARN-03 | mixed-version model 可持续训练 | end-to-end trace |
| A-PERF-01 | 稳态不传完整模型 | byte accounting |
| A-PERF-02 | pending queue 有界 | slow-FS stress |
| A-PERF-03 | 第 N 次 update 不依赖历史扫描 | 1000+ transition latency/storage report |
| A-ALG-01 | direct merge 与 oracle 一致 | numeric digest/tolerance report |
| A-ALG-02 | stale-aware 与 fresh-only 有独立结果 | matched experiment report |
| A-EVAL-01 | evaluation 使用冻结 version vector | snapshot manifest + report |

---

## 21. 第二阶段：分布式无重叠 syncer 的设计边界

第二阶段开始前，第一阶段必须已通过，且不得修改第一阶段的 proposal、staleness、merge 或 global-fragment 数学语义来掩盖分布式实现差异。

### PH2-01：无重叠 ownership

所有 fragments 被划分为互斥集合，每个集合由一个 syncer 负责。任何 fragment 只能属于一个 active syncer。

### PH2-02：fragment 独立提交

不同 syncer 可以并行推进不同 fragment。不存在要求所有 syncer 同时提交的 global transaction。

### PH2-03：相同输入等价

固定：

- current fragment state；
- proposals；
- selected set；
- weights；
- merge policy；
- outer optimizer；
- compute dtype；

多 syncer 与单 syncer 对同一 fragment 必须生成相同或在预注册浮点容差内一致的结果。

### PH2-04：全局 snapshot

evaluation 或导出仍通过冻结所有 per-fragment current identities 构成 global snapshot，不引入全局训练 barrier。

### PH2-05：暂不容错

第二阶段仍不包含：

- syncer overlap；
- backup owner；
- automatic reassignment；
- failover fencing；
- speculative duplicate execution。

### PH2-06：扩展理由

第二阶段性能结论必须比较：

- single syncer；
- multiple non-overlapping syncers；
- 相同 learner 数、模型、fragment map、FS、训练 tokens 和 merge policy。

必须报告 syncer CPU、FS 并发、learner GPU goodput 和总资源成本。

---

## 22. 设计决策冻结表

| 项目 | 第一阶段决定 |
|---|---|
| 模型范围 | dense decoder-only Transformer causal LM |
| 逻辑层 | embedding、完整 Transformer blocks、lm_head |
| 零散参数 | 并入相邻较小逻辑层，平局归前侧 |
| fragment | 相邻完整层、连续、非空、按同步字节均衡 |
| fragment frequency | 第一阶段统一间隔，错开 offsets |
| learner payload | 本地完整 fragment 参数快照 |
| global state | per-fragment parameters + outer state + version |
| proposal discovery | 每 learner/fragment latest-wins |
| quorum | distinct learners |
| stale 支持 | 第一阶段必需，默认 `S_max=1` |
| stale displacement | `G_base - L_local` |
| stale weight | `tokens/(1+lambda_s×staleness)`，默认 `lambda_s=1` |
| fresh anchor | 默认 `Q_fresh>=1` |
| baseline merge | weighted direct averaging |
| RDA | 第一阶段必须完成对照，默认由证据决定 |
| outer optimizer | fragment-wise momentum/Nesterov SGD 主线 |
| learner adoption | latest-only、整 fragment 覆盖、保留 inner moments |
| FS 状态 | bounded current/base/latest，不保存完整历史 |
| 第一阶段 syncer | 一个逻辑 CPU syncer |
| 第二阶段 syncer | 多个、互不重叠 fragment ownership |
| 故障恢复 | 不属于第一阶段/第二阶段当前验收 |
| telemetry | best-effort，不是 authority |

---

## 23. 研究报告要求

任何 stale-aware 结论必须至少同时报告：

- `S_max`；
- `lambda_s`；
- `Q`、`Q_fresh` 和 grace window；
- accepted proposals 的 staleness distribution；
- fresh/stale normalized weight mass；
- rejected-too-stale 比例；
- learner adoption lag；
- fragment update frequency；
- total bytes/token；
- training tokens、FLOPs 或 wall-clock budget；
- fresh-only 对照；
- direct averaging 对照；
- loss 和 evaluation；
- GPU goodput 与 syncer/FS 成本。

不得把以下两种收益混为一谈：

1. stale proposal 提高 quorum / 资源利用率；
2. 更高总通信量或更多 outer updates 带来的质量变化。

---

## 24. 开放问题，但不阻塞第一阶段启动

以下问题通过阶段性证据决定，不在开始前过度设计：

1. `Q < M` 的最佳默认值；
2. fixed grace window 的最佳长度；
3. weighted direct averaging 与 embedding-Avg / non-embedding-RDA 的最终默认组合；
4. learner 采用 global fragment 时是否重置对应 inner optimizer moments；
5. `S_max > 1` 是否有训练质量或 goodput 价值；
6. 是否需要 norm clipping 或 stale-specific outer learning-rate scaling；
7. 是否需要 per-fragment 非均匀同步频率；
8. 单 syncer 是否构成实际性能瓶颈；
9. 是否值得在 data payload 层引入 CHFS 或其他 cache；
10. 是否需要通信量化。

每个开放问题都必须保持一个简单 baseline，并通过独立实验解决。

---

## 25. 参考依据与差异声明

1. Decoupled DiLoCo for Resilient Distributed Pre-training：
   https://arxiv.org/abs/2604.21428
2. Streaming DiLoCo with overlapping fragment communication：
   https://arxiv.org/abs/2501.18512
3. 当前仓库及历史 DuraLoCo Loop-Engineering 计划仅作为经验来源，不作为新实现的架构依赖：
   https://github.com/UnbearableFate/fs_based_decoupled_diloco

差异声明：

- 本规格采用相邻完整层的连续均衡分片，不采用论文主要实验中的 balanced-tensor fragmentation；
- 本规格第一阶段显式允许 bounded stale proposal，而 Decoupled DiLoCo 论文主要算法强调不把旧参数版本上的 stale gradient 直接应用到新参数；
- 因此 stale-aware 结果必须标记为本项目的算法扩展，并始终保留 fresh-only reference；
- 本规格删除完整 event tape、prefix replay、分布式 checkpoint、fencing、冗余执行和完整审计要求；
- 本规格的 runtime correctness 依赖最小 per-fragment publication，不依赖历史事务链。

---

## 26. 第一阶段完成定义

第一阶段只有同时满足以下条件才算完成：

1. S1-00 至 S1-07 全部通过；
2. 所有系统级不变量有自动化或可重复证据；
3. layer-aligned fragment map 在目标模型上确定性且完整；
4. fresh-only reference 与数学 oracle 一致；
5. `S_max=1` stale-aware profile 在真实多 learner 速度差异下运行；
6. stale proposal 使用真实 old base 重建 displacement；
7. 默认每次 stale-aware update 至少一个 fresh contributor；
8. steady-state 不读写完整模型；
9. GPU→CPU→FS 与 FS→CPU→GPU 流程具有有界 backpressure；
10. 1000+ fragment updates 后 discovery、storage 和 latency 不随历史增长；
11. fresh-only 与 stale-aware 具有 matched-budget 质量和性能报告；
12. 独立 Checker 给出 `PASS` 或经明确 follow-up 不影响主结论的 `PASS_WITH_FOLLOWUPS`；
13. 第二阶段可以在不改变第一阶段数学语义的情况下，将 fragments 静态分给多个无重叠 syncer。

