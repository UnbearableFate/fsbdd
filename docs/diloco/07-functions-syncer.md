# syncer 函数级代码参考

本篇对应 `src/fsbdd/diloco/syncer`。syncer 的核心边界是：readiness 只冻结事实，merge 只计算 successor，atomic commit 才推进可见 authority；三者不能合并成“读 latest 后直接覆盖”。

## 1. `syncer/readiness.py`

源码：[`readiness.py`](../../src/fsbdd/diloco/syncer/readiness.py)

### 1.1 配置、authority 和状态值

| 符号 | 代码内容 |
|---|---|
| `ReadinessError` | proposal 资格、状态机、调度或 authority 违反契约。 |
| `_require_integer` / `_require_nonnegative` / `_require_positive` | 严格整数 validator，显式拒绝 bool。 |
| `ReadinessConfig.__post_init__` | 要求 `0≤q_fresh≤q≤max_contributors`、非负 grace/lambda；当前 Stage 1 显式只允许 `s_max=0`。 |
| `FragmentReadinessAuthority.__post_init__` | 把一片 `FragmentGlobalState` 与对应 `ConsumptionFrontiers` 绑定，identity/descriptor 必须相同。 |
| `FragmentReadinessAuthority.identity` | digest 覆盖 state content/version 和按 learner 顺序的消费 frontier；selection 因而绑定参数与消费历史。 |
| `ReadinessPhase` | `WAITING → GRACE → FROZEN → ACTIVE`；零 grace 可直接 `WAITING → FROZEN`。 |
| `FrozenSelection.__post_init__` | 验 syncer/fragment/version/authority/timing；proposal、weight 等长非空；learner/proposal 唯一；weight facts 与 proposal 一致且和≈1；重算 selection identity。 |
| `FrozenSelection.learner_ids` | 保持冻结 proposal 顺序返回 learner tuple。 |
| `FrozenSelection.to_dict` | 输出 authority、generation、时间、ordered proposal facts/weights 和 selection identity。 |
| `UpdateLease` | 把唯一 active claim ordinal 与 frozen selection 绑定。 |
| `FragmentReadinessView` / `PollReport` / `ReadinessSnapshot` | 分片、单轮 polling、全状态机的只读 evidence。 |
| `_FragmentRound` | 每片 mutable 状态：authority、phase、资格计数、grace、selection、generation、claims。 |

### 1.2 `SyncerReadinessMachine`

| 方法 | 代码内容 |
|---|---|
| `__init__` | 校验 store/config/topology 和每片 authority；建立 RLock、round-robin cursor、唯一 active lease、每 fixed slot proposal/record/payload identity cache 及 transition counters。cache 上界是 `M×F`。 |
| `_install_proposal_observation` | 强制 fixed-slot observation 单调：sequence/base 不回退；同 sequence record 和 proposal 不得变化。只有 record 完全相同才算 cache hit；任何 visibility 改变都必须连 payload 重新验证，避免“相同 payload SHA、metadata 被改写”绕过校验。 |
| `_validate_authority` | authority 必须与 proposal store frozen identities、descriptor position 和 learner membership 相同。 |
| `_now` | 取注入 clock 或显式 observed ns，并禁止 observation time 回退。 |
| `_policy` | 从当前 authority 生成 `EligibilityPolicy`。正式模式用配置门槛；permissive 模式用 `q=1/q_fresh=0/max=M` 先得到全部有序合格候选。保留 base 只有 current，`s_max=0`。 |
| `_set_phase` | 变更 phase 并更新已定义 transition counter。 |
| `_freeze` | 从 deterministic eligible 顺序截到 max，要求至少 q；计算 float32 weights；generation 加一；digest authority、proposal content identities 和 weights，构造 immutable `FrozenSelection`，进入 FROZEN。 |
| `_normalize_candidates` | 按 fragment 和 proposal id 去重；同 id 不同 content 失败，同内容重复只增加 duplicate count。 |
| `_view` | 把 `_FragmentRound` 投影为不可变 view。 |
| `observe` | 对 WAITING/GRACE 片运行 selection；更新 eligible/fresh/rejected/duplicate。达到 quorum：零 grace 立即 freeze，否则开始 grace；grace 期间 quorum 消失则回 WAITING，保留至 deadline 则 freeze。FROZEN/ACTIVE 不被新 latest proposal 改写。 |
| `poll_store` | 对每 learner×fragment 固定 slot 做 metadata probe。支持 bound read 时，只并行读取发生变化的 immutable payload，然后按确定顺序装 cache；不支持时用 `load_latest_if_changed` fallback。missing 和 incomplete 分开计数，最后以完整 cache 调 `observe`。每轮固定 metadata 工作量为 `M×F`。 |
| `claim_next` | 若已有 active 返回 None；否则从 cursor 循环找第一个 FROZEN，生成 lease，cursor 移到下一片并进入 ACTIVE。确保单 syncer 同时只 merge 一片且片间公平。 |
| `_require_active` | lease 必须对象相等于当前唯一 active，并返回目标 round。 |
| `release` | merge/commit 失败时清 active，将同一 selection 从 ACTIVE 退回 FROZEN，可重试而不重新选择。 |
| `_validate_authority_progress` | 新 authority version/content 不得回退或同版本改内容；每 learner consumed sequence/base frontier 都不得回退。 |
| `_install_authority_unlocked` | 验新 authority；相同 identity 为 no-op。推进时清该片 readiness/grace/selection，回 WAITING，并记录 active/authority transition。 |
| `install_authority` | 外部安装非 active 片的新 authority；active 目标必须走 `complete`。 |
| `complete` | 要求 active update 确实推进 authority，清 lease，并原子安装 successor。 |
| `selection_for` | 返回某片当前 frozen selection 或 None。 |
| `active_lease` | RLock 下读取唯一 active lease。 |
| `snapshot` | 汇总 cursor、active、poll/claim、cache hit/miss、resident proposal 上界、transition 和每片 view。 |

### 1.3 readiness 的关键确定性

候选排序来自 `protocol.proposal.select_candidates`：staleness 小优先、effective tokens 大优先、sequence 大优先、最后按 learner/proposal id。`observe` 在 grace 结束时重新对当前 cache 排序后冻结；一旦 FROZEN，后到 proposal 不会修改本轮 selection。round-robin 只决定哪个已经冻结的片先执行，不参与候选排序。

## 2. `syncer/merge.py`

源码：[`merge.py`](../../src/fsbdd/diloco/syncer/merge.py)

### 2.1 输入、policy 与 source

| 函数/方法 | 代码内容 |
|---|---|
| `MergeError` | merge 输入、流式资源或数值更新违反 frozen contract。 |
| `_require_text` / `_require_hex` / `_require_nonnegative` / `_require_positive` | identity 和计数 validator。 |
| `_f32` | 用 network-endian struct pack/unpack 规范为 IEEE float32，拒绝 overflow/NaN/Inf/bool。 |
| `_require_fp32_payload` | 要求非空 immutable bytes 且长度为 4 的倍数；数值有限性在 decode 后检查。 |
| `ContributionFact.__post_init__` | 验 proposal/learner/base/tokens/staleness/weight/parameter SHA/payload bytes。 |
| `ContributionFact.f32_weight` / `identity_facts` | 返回规范 float32 weight；输出完整贡献 provenance。 |
| `FragmentOuterState.__post_init__` | update count 非负；可选 momentum 是 fp32 bytes。 |
| `FragmentOuterState.identity_facts` | 输出 update count 和 momentum SHA，不嵌入大 bytes。 |
| `OuterSGDPolicy.__post_init__` | 规范 fp32 learning rate/momentum；要求 lr>0、`0≤momentum<1`、Nesterov 必须有正 momentum。 |
| `f32_learning_rate` / `f32_momentum` / `identity_facts` | 暴露真正用于计算和 identity 的 float32 超参数。 |
| `FragmentMergeRequest.__post_init__` | descriptor 必须 flat fp32 且 shape 匹配 current bytes；outer update count=current version；contributions 非空且 proposal/learner 唯一；每 payload 同片大小；`staleness=current-base`；逐项 float32 累加的权重和≈1；momentum 同 shape。 |
| `StreamingContributionSource.open_payload` | context-manager protocol：一次租用一个 local payload。 |
| `ResolvedBase.__post_init__` | 验 base version/content、fp32 bytes 和 parameters SHA。 |
| `StreamingBaseSource.open_base` | noncurrent contribution 的 retained base context-manager protocol。 |
| `PayloadLocation.__post_init__` | identity 非空；path 必须相对 frozen root 且不能含 `..`。 |
| `SourceMetrics` | opens/read bytes/current active/maximum active 四项计数。 |
| `ImmutableFileContributionSource.__init__` | resolve frozen root，按 identity 建唯一相对路径表，初始化租用计数。 |
| `ImmutableFileContributionSource.open_payload` | 整文件读取指定 immutable path，更新 active metrics，并在 finally 释放；checksum 由 merge request facts 验证。 |
| `ImmutableFileContributionSource.metrics` | 返回 source counters。 |
| `ImmutableFileBaseSource.__init__` | 复用 immutable file source，并保存 base content identity→version/content/parameter SHA metadata。 |
| `ImmutableFileBaseSource.open_base` | 按 contribution 声明的 base identity 定位文件，构造并验证 `ResolvedBase`。 |
| `ImmutableFileBaseSource.metrics` | 代理内部 file source。 |
| `MergePolicy.identity_facts` / `accumulate` | 抽象策略 identity 与就地累加接口。 |
| `DirectWeightedAverage.identity_facts` | 冻结名称与公式 `sum(weight * (declared_base - local))`。 |
| `DirectWeightedAverage.accumulate` | 复用当前 local tensor，就地执行 `local = weight*(base-local)`，再加到 accumulator，避免额外 fragment tensor。 |

### 2.2 结果与 accounting

| 符号 | 代码内容 |
|---|---|
| `ByteAccounting.to_dict` | 导出 current/local/base/successor 的精确 bytes 和 read count；`full_model_operations` 默认为 0。 |
| `MemoryAccounting.maximum_tensor_fragment_multiples` | `maximum_live_tensor_bytes / fragment_bytes`。 |
| `MemoryAccounting.to_dict` | 导出逻辑 tensor 峰值和可选 RSS evidence。 |
| `FragmentUpdateResult.__post_init__` | successor/merged gradient 必须 fp32 payload，update identity 必须 SHA；其余对象由生成路径构造。 |
| `_TensorLedger.__init__` | 逻辑 live/maximum 从零开始。 |
| `_TensorLedger.add` / `remove` | 跟踪 live tensor bytes 和峰值；underflow 表示实现错误。 |
| `_tensor_from_payload` | `<f4` NumPy copy→CPU torch tensor，检查有限性。 |
| `_payload_from_tensor` | 要求 CPU float32 finite tensor，输出 contiguous little-endian bytes。 |
| `build_update_facts` | 生成 update identity 的完整 semantic：current、outer state、selection、ordered proposal/base/token/staleness/weights/local SHA、merge policy、outer hyperparameters、fragment identities。 |

### 2.3 `execute_streaming_fragment_update`（Torch reference path）

此函数验证 request/policy 后按如下顺序工作：

1. 解码 current，分配一个同 shape accumulator；可选 `observe_rss` 闭包从 procfs 采样 RSS。
2. 按冻结贡献顺序逐个 `open_payload`。校验长度、SHA、有限性；同 current base 直接引用 current，旧 base 则只在本贡献期间 `open_base` 并严格核对 version/content/shape。
3. 调 merge policy 累加 `weight*(base-local)`；每个 local/base lease 在下一个贡献前释放。得到
   `g = Σ_i w_i (G_base(i) - L_i)`。
4. 无 momentum 时 `direction=g`；有 momentum 时首次 `v=g`，否则 `v=momentum*v+g`；Nesterov 使用 `direction=g+momentum*v`，普通 momentum 使用 `direction=v`。
5. successor 为 `G_next = G_current - learning_rate*direction`。编码 successor、merged gradient 和 next momentum；构造 update facts/digest、byte/memory accounting。

local 闭包 `observe_rss` 只在 `measure_process_rss=True` 时调用 `linux_process_memory_bytes`，不改变数值路径。函数同时检查 momentum-free request 不得携带 buffer、所有租用均已释放、所有结果有限。

### 2.4 `execute_numpy_streaming_fragment_update`（当前 CPU Profile A path）

NumPy 路径保持与 reference path 明确一致的 float32 运算顺序，但当前仅接受 fresh/current-base contribution：

1. local 闭包 `array` 将 fp32 bytes copy 为 finite `<f4` array。
2. 常驻 current、accumulator 和一个 reusable `local_scratch`；逐贡献把 `current-local`、乘 weight、加 accumulator 全部用 `out=` 就地执行。
3. 若 source 声明 `prevalidated_payload_integrity=True`，可跳过第二次 local SHA；否则逐 payload 校验。任何 noncurrent base 直接失败。
4. momentum/Nesterov/successor 同样用 `np.float32` 和 `out=` 明确顺序；构造同一种 `DirectWeightedAverage` update facts 和 result。

该实现不 import Torch/CUDA，是正式 CPU syncer 使用的路径；`local_scratch` 的内嵌 `array` helper 和 Torch 路径的 `observe_rss` 都属于父函数实现细节，不是公共 API。

| 其余函数 | 代码内容 |
|---|---|
| `linux_process_memory_bytes` | 读取 `/proc/<pid>/status` 的 `VmRSS`/`VmHWM`，单位 kB 转 bytes；缺字段或单位异常失败。 |

## 3. `syncer/profile_a.py`

源码：[`profile_a.py`](../../src/fsbdd/diloco/syncer/profile_a.py)

### 3.1 配置、计数和停止策略

| 函数/方法 | 代码内容 |
|---|---|
| `ProfileAError` | fresh-reference profile 或集成 transition 错误。 |
| `_text` / `_ordinal` / `_positive` / `_finite_nonnegative` | Profile A 基础 validator。 |
| `profile_a_policy_identity` | digest `DirectWeightedAverage + outer SGD float32 hyperparameters + accumulation dtype`，绑定 commit store 与 executor。 |
| `ProfileAConfig.__post_init__` | 强制 `q=q_fresh=M`、`s_max=0`、grace=0、schedule=`round_robin`；因此每轮必须收齐所有 learner 的 current-base proposal。 |
| `ComparisonBasis.__post_init__` | 校验三个 matched flag；三者都 false 时只允许固定的“protocol/runtime acceptance only” claim。 |
| `ComparisonBasis.to_dict` | dataclass 序列化。 |
| `LearnerProgressRecord.__post_init__` / `to_dict` | 校验和导出每 learner 的 local step、input tokens、loss-bearing targets。 |
| `FragmentProgressRecord.__post_init__` / `to_dict` | 校验和导出每片 outer count、accepted tokens、fresh/stale contribution 累计。 |
| `ProfileAProgressReport.__post_init__` | learners/fragments 非空；fragment index 连续有序；`global_cycle=min(fragment outer counts)`。 |
| `total_compute_steps` / `total_accepted_tokens` | 对 learner steps、fragment accepted tokens 求和。 |
| `ProfileAProgressReport.to_dict` | 导出报告并附每项 counter 的语义，避免把 local step、fragment update、global cycle 混为一谈。 |
| `ProfileAProgressTracker.__init__` | 冻结唯一 learner 顺序和 fragment 初始 outer counts，累计量置零。 |
| `update_learner` | 更新指定 learner 的绝对计数，任何字段回退失败。 |
| `record_fragment_commit` | 目标 outer count 必须恰加一；accepted tokens/fresh 必须正，stale 可为零；在旧累计上相加。 |
| `report` | 按冻结 learner 顺序和 fragment 顺序生成 report。 |
| `StopDecision.to_dict` | 导出 stop/reason/policy identity 及观测预算。 |
| `FrozenStopPolicy.__post_init__` | 至少设置一个预算；cycle/walltime/steps/tokens/FLOPs 若设置必须为正。 |
| `FrozenStopPolicy.identity` | 对完整 frozen budget 计算 digest。 |
| `FrozenStopPolicy.evaluate` | 按固定优先级 cycle→walltime→compute steps→accepted tokens→FLOPs 判断首个 reason，并记录所有观测量；未提供 FLOPs 时 FLOPs budget 不触发。 |

### 3.2 payload source、结果与 executor

| 方法 | 代码内容 |
|---|---|
| `_SelectedProposalSource.__init__` | 从 frozen proposal tuple 建 `proposal_id→parameters` 内存表并要求唯一；标记 payload 已经 proposal decode 校验。 |
| `_SelectedProposalSource.open_payload` | 按 contribution id 租用 bytes，更新 open/bytes/active/maximum active，在 finally 释放。 |
| `_SelectedProposalSource.metrics` | 返回流式 source evidence。 |
| `ProfileAUpdate.__post_init__` | selection typed；successor 必须 previous+1；merge result parameters 必须等于 committed bytes；两段 latency 非负。 |
| `ProfileAFragmentExecutor.__init__` | 对齐 atomic/proposal store topology、identity、descriptor、learners 和 `s_max=0`；验 policy identity；载入 atomic snapshot，把 state+frontiers 转 readiness authorities；创建配置为 Profile A 的 readiness machine。backend 只允许 `torch` 或 `numpy`。 |
| `_contributions` | 按 frozen selection 顺序把 proposal+weight 转成 `ContributionFact`，保留 base、tokens、staleness、payload SHA/bytes。 |
| `poll` | 代理 readiness `poll_store`。 |
| `execute_next` | 可先 poll；round-robin claim。重新加载 authority 并与 selection 绑定点比较；构造 source、outer state 和 merge request；按 backend merge，再构造 `AtomicCommitRequest` 原子提交。异常 release 原 selection；成功 complete readiness、累计 fresh/stale/tokens，返回包含两段 latency 的 `ProfileAUpdate`。无 frozen 片返回 None。 |

`execute_next` 的提交边界很重要：只有 `AtomicGlobalCommitStore.commit` 成功后才调用 readiness `complete` 和 progress `record_fragment_commit`。merge 成功但 commit 失败不会虚增 version/frontier/counter；selection 留在 FROZEN 可按同一事实重试。

## 4. 空初始化文件

`syncer/__init__.py` 当前为空，不启动调度线程，也不选择 merge backend。
