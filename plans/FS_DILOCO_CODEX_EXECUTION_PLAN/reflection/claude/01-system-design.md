# 01. 训练系统设计审查（运行效率与准确率）

证据基线：
- 2393139.opbs（capacity smoke，修复前）：189 次 update，均值 1.739s、p90 1.834s；
  update 执行占首末 update 区间的 60.9%；shared root 653 文件 / 67.5GB；最大复合
  successor payload 340,239,068 字节。
- 2393536.opbs（publish 修复后）：232 次 update 均值 1.0597s；四 learner 全部按时
  完成，失败仅因终局 quorum 边界（58/59）。
- 2394870.opbs（formal long，用户早停）：9217 个公共 step，loss ratio 0.99399、
  robust slope −4.26e-6；预测全程 21,501–21,573s vs 冻结预算 21,600s。
- Stage 0-B：跨节点可见性 p99 7.95ms；元数据 39,403 ops/s；16-stream 写 0.324s。

---

## §1 复合 global payload 造成的读写放大（最高优先）

### 现状

`FragmentGlobalState` 的编码（`global_state.py:382 encode_global_state`）把
**parameters + outer_state（envelope 头 + momentum 全量）+ 历史 base** 拼成单一二进制
blob，一次 `publish` 发布。momentum buffer 与参数等大，因此：

- syncer 每次 outer update 写 ≈ 2× fragment 字节（实测 340MB vs 165MB fragment）；
- **每个 learner 的每次 adoption 都读整个复合 payload**（`adoption.py:390
  load_bound_fragment_record`），其中 momentum 与 envelope 对 learner 完全无用——
  M=8 时每个 fragment 版本的 adoption 读流量是必要量的 2 倍；
- Stage 4 打开 S_max=1 后，前一版 base 参数也进同一 blob，learner adoption 读放大
  升至 **3×+**，且每次 update 都把没有变化的旧 base 重新写一遍（旧 base 内容不变
  但作为新 payload 的一部分重新落盘、重新哈希）。

### 影响

- A-PERF-01（稳态不传完整模型）在字面上满足，但 bytes/token 这个论文 C5/REPORT-01
  指标被无谓放大 2–3 倍；H1 的"FS 传输开销占比 <10%"直接吃亏。
- Stage 4 的 base 保留窗口（STALE-03）在当前编码下是"每次 update 重写整个历史窗口"，
  与 FS-06 的 bounded retention 精神相悖（有界但常数放大）。

### 建议

把"一个可见性记录 = 一个 payload"放宽为"一个可见性记录命名多个不可变 section
payload"（记录里已有 per-section 的 sha256/bytes 结构，`_semantic_header` 的
`sections` 数组几乎就是现成 schema）：

1. parameters、outer_state、每个历史 base 各自独立 payload 文件，内容寻址
   （sha256 命名），可见性记录原子替换时同时命名全部 section——原子性仍由单一小
   记录保证，INV-03/GLOBAL-02 不受影响；
2. learner adoption 只取 parameters section；
3. 未变化的旧 base 不重写（新记录引用旧 payload 路径）；GC 以"当前记录引用集合"
   为存活根，与现有 `reclaim_unreferenced_payloads` 兼容；
4. 该改动同时消除 Stage 4 的历史 base 重写，属于"打开 S_max=1 之前必须完成"的
   基础工作，建议作为 Stage 2 的一个 loop 而不是留到 S4-01。

## §2 syncer 串行执行是外推到 0.5–1B 的第一瓶颈

### 现状

`run_syncer` 主循环（`close.py:1028`）严格串行：poll → 逐 fragment
`execute_next`（读 M 个 proposal → NumPy merge → 编码 → publish 复合 payload）。
所有哈希、I/O、向量运算在一个线程一个核上；GH200 节点 72 CPU 只用 1 个。

每次 update 的字节量级（M=4、fragment 165MB）：读 4×165MB（含 SHA-256 校验）+
写 330MB（含 SHA-256）≈ 1.3GB 触碰/哈希，与实测 1.06s 一致（哈希+页缓存 I/O
~1.3GB/s 单核）。

### 外推

H=50、实测 step ~0.176s → 每 fragment 发布周期 ≈ 8.8s；F=4 → syncer 预算
2.2s/update，当前余量约 2×。到 Stage 2/5 的 0.5–1B 模型：fragment 1–2GB、
update 成本 ≈ 6–13s；即使 step 时间同步变大，余量也会跌破 1，syncer 成为
全局时钟——update interval 变长 → adoption lag 变长 → learner 实际基于更旧的
base 训练 → 有效 staleness 上升，**这在 fresh-only 配置下直接表现为 discard 或
更新频率下降，损害收敛质量**，而不只是吞吐。

### 建议（保持 INV-02 单逻辑写者，不需要 Phase 2 多 syncer）

1. **并行读+哈希**：selection 冻结后用线程池并发读/校验 M 个 proposal（I/O 与
   hashlib 都释放 GIL），单线程 merge 不变；
2. **fragment 流水线**：fragment f 的 publish（写+哈希）与 fragment f+1 的读/merge
   重叠；单写者语义按 fragment 保持；
3. **减少哈希遍数**（见 02 §3）：同一 payload 从 learner 产出到 syncer 消费当前
   被完整哈希 4–6 遍，其中至少 2 遍可由"不可变边界一次验证"替代；
4. 在 spec 增加 syncer 侧验收（现在完全缺失，见 §5）：如
   `A-SYNC-PERF: update_interval_p90 ≤ 0.5 × H_wall`，在 160M 与 0.5–1B 各测一次。
   S1-13 的两次容量失败都是撞 harness 预算才暴露的，spec 里没有任何条款约束
   syncer 节奏。

## §3 syncer 常驻内存 ∝ M×F×fragment（SYNC-06 冲突）

`Proposal`（`proposal.py:104`）自带完整 `parameters: bytes`，readiness 缓存的是
整个 Proposal 对象（为修复"每次 poll 重哈希"而改为缓存已验证对象）。因此 syncer
稳态内存 = 所有 latest proposal payload 之和 = M×F×fragment_bytes：

- 当前（M=4、F=4、165MB）≈ 2.6GB，可接受；
- M=8 + 1B 模型（fragment ~2GB）≈ 64GB，超出 CPU 节点合理配置，且与 SYNC-06
  "峰值工作内存不与 M × fragment size 成正比"矛盾。merge 侧的
  `maximum_active_local_payloads=1` 记帐只覆盖 merge 函数视角，掩盖了真实驻留。

建议：readiness 缓存降级为"已验证的 PublicationRecord + 元数据"（识别 sequence
变化用 `payload_identity` 即可，无需字节）；payload 在 selection 冻结后经
`read_bound_record` 流式拉取（基础设施已存在），配合 §2 的并行读。这样把驻留从
M×F×fragment 降到 max_contributors×fragment（且可进一步流式化到 1×fragment）。

## §4 `minimum_step_seconds` 掩盖了真实容量边界

`close.py:763` 按冻结契约给每个 optimizer step 设置下限 sleep，声明目的是
"preserve H50 fixed-slot observability below the 800 step ceiling" 和"证明 formal
long 调度能守住 cycle 余量"。问题：

1. 它把"syncer 跟不上"翻译成"把 learner 放慢到 syncer 跟得上"，与 INV-01 的精神
   相反（learner 独立前进是协议卖点）；
2. 真实速度下的 latest-wins 覆盖率、quorum 形成模式、proposal 生产/消费比全都
   没有被任何 run 观察过；Stage 2 的 goodput 定义（对照关闭通信的本地吞吐）如果
   带着 pacing 测就是自证；
3. 它已经制造了一次纯边界误报（2393536：58/59 终局 quorum 不可能形成，浪费一次
   五节点 attempt + 一轮台账/审查）。

建议：Stage 2 起 gate 契约不再允许 pacing；smoke 的终止条件改为
"到达 cycle 目标即停"而非"固定 step 预算内到达 cycle 目标"（终局 quorum 不可形成
的问题随之消失）；把"learner 全速时 syncer 消费率/覆盖率"本身列为 telemetry 观察量
（proposal superseded-before-consumption 比率已可从 sequence 差推出）。

## §5 崩溃一致性：kill-restart 之外的空洞要在 Stage 3 之前写清楚

`PosixStorageBackend.publish` 全程无 fsync：payload 写完（页缓存）→ 记录 temp 写完
→ `os.replace`。对 **进程级 kill -9**（Stage 3 的声明范围）这是正确且完备的——页
缓存由 OS 持有。但对节点崩溃/断电，可见性记录可能已持久而 payload 字节丢失：
reader 侧 sha256 fail-closed 保证不读坏数据（安全性成立），但该 fragment 的
current 记录将永久指向不可读 payload，**可用性**死锁，只能人工回滚。FS-06 里
"可选保留上一完整 current record（人工恢复辅助）"正是为此准备的，但当前实现
没有落（`_mark_retiring` 只留时间戳标记，不保证旧 payload 在新记录持久前存活）。

建议（三选一，写进 Stage 3 的 ORIENT）：
1. 论文与 spec 把 C3 明确限定为进程级 crash（最低成本，措辞改动）；
2. publish 增加可配置 `durable` 模式：fsync(payload)+fsync(record tmp)+rename+
   fsync(dir)，在 Stage 3 量化其代价（Lustre 上很可能可接受，且是论文的好数据点）；
3. 至少实现 FS-06 的"上一完整记录"回退链，把断电从"人工修复"降级为"自动回退一版"。

另外 Stage 3 的 REC-01"proposal sequence 必须恢复单调性"目前依赖的持久化点建议
提前在 Stage 2 telemetry 里核对（learner 重启后 sequence 从何恢复——从自己 slot 的
latest 记录读回是现成方案，但要有测试证明 crash 时 in-flight 覆盖不会回退）。

## §6 研究准确率：stale 假设的预测证据已经为负

`reports/stage0/simulation_recommendation.md` 的预注册 Profile B 锚点
（M=8、Q=4、Q_fresh=1、S_max=1、λ=1、grace 0.1H、异构 2×、可见性 1s）：

- 预测挽回：**−0.0446pp**（95% CI 半宽 ~0）；
- stale accepted-token 率仅 5.31%；
- accepted-token 效率 33.48%；
- 论文分级已触发 "<3pp → ablation_or_discussion"。

两个层面的含义：

1. **H3（≥10pp）在该锚点下预测不成立**。RESEARCH_PLAN 的篇幅表处理了叙事降级，
   但 Stage 4 仍是 6 个 loop、每个（或 gate 单元）一次 9N、外加 S4-06 多种子
   matched-token —— 为一个预测为负的效应支付全价。建议在 S4-01 之前增加一个
   廉价仿真 loop：扫描锚点邻域之外的区域（Q/M→0.75/1、异构 3×、可见性 5–30s、
   grace=0、S_max=2），回答"是否存在 stale 有实质收益的参数区，如果有，把 Profile B
   锚点移过去并写 ADR；如果没有，把 S4-06 砍到最小负结果对照（1 seed pair +
   置信区间，而不是 ≥3 seeds 全矩阵）"。
2. **33.48% 的 accepted-token 效率本身是个未解释的大数**：Q=4/8 意味着每次 update
   丢近三分之二的已处理 token，而 stale-1 只捡回 5%。这既可能是仿真调度/offset
   的伪影，也可能是 fresh-anchored bounded-stale 机制的真实上限——两种情况对论文
   叙事（C4 甚至 C2 的 matched-communication 对照）影响完全不同，值得在 Stage 4
   之前用仿真归因（按丢弃原因分解：sequence 覆盖 / base 过期 / quorum 已满）。

## §7 门禁阈值与 workload 的校准问题

1. **预训练 checkpoint 上的 loss ratio 门**：9N 门用 pretrained GPT-2 微调
   WikiText-2（快速下降，门槛宽松），formal long 用 pretrained Pythia-160M 续训
   FineWeb-Edu——后者 9217 步实测 ratio 0.9940，与 0.99 门槛只差 0.004。从
   pretrained 出发的 loss 下降幅度天然微小，阈值又是拍的，导致门禁信噪比低。
   建议：门槛按 workload 分开校准并写明依据；Stage 5 的质量线（本就要求从头训练）
   建立后，"loss 在下降"类门禁改用 scratch-init 短跑，其斜率信号强一个量级。
2. **runtime 预算余量 0.1–0.5%**：projected 21,501–21,573s vs 21,600s 预算。任何
   队列外的扰动（页缓存冷、邻居干扰）都会把一次 6 小时 run 变成"预算超支失败"。
   冻结预算是对的，但预算=预测中位数×(1+10–15%) 这类规则应写进 gate 契约模板，
   不要再出现个位秒余量。
3. **fresh-only Q=M 的终局边界**是结构性的（最后一个 H 窗口永远无法形成满 quorum），
   任何"固定 step 预算 + cycle 目标"的 smoke 都会踩到。§4 的终止条件修改可以根治；
   同时建议 Stage 2 起把 9N 回归 profile 换成 Q<M 的 decoupled 配置——它既是论文
   目标配置，又天然没有终局死锁。

## §8 数据面 dtype：fp32 载荷值得一个 ADR

当前同步 dtype 为 fp32（payload `<f4`，merge fp32 累加符合 spec）。bf16 线格式
可把所有 payload 字节、哈希、带宽减半，且这是 FRAG-04 意义上的"同步 dtype"选择，
不属于 §1.5 排除的"量化/压缩"（需要 ADR 明确解释这一边界）。代价是
pseudo-gradient 在 bf16 量化下的精度损失需要一组 matched 对照。建议列为 Stage 5
可选消融（优先级低于 §1/§2，因为那两项不改变数值语义）。

## §9 learner 侧较小但值得记录的点

1. `LearnerRuntime` 的 update-norm 采样对整组 fragment 参数做 `detach().clone()`
   （GPU 上全模型副本）再算 delta 范数（`runtime.py:798-805,862-874`）。采样间隔
   50/1000 时代价可接受，但默认值 1 是个陷阱（每步全模型 clone）；建议默认改为 0
   （禁用）或强制显式配置，且 Stage 2 goodput 对照两臂必须固定同一采样率（INV-10）。
2. 每步 `torch.cuda.synchronize`（`runtime.py:860`）是计时需要，但 Stage 2 追 95%
   goodput 时应改为事件计时或降频采样。
3. 数据加载完全同步在训练线程内（`next(iterator)` + 同步 H2D，`non_blocking=False`
   无 pinned memory），160M/512-seq 下可能无所谓，1B 模型建议加一层预取。
4. `_batches`/`PackedTokenShard` 每 epoch 确定性 permutation、shard 固定归属，
   与 matched-token 方法学一致，好。
