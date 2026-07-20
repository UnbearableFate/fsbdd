# learner 函数级代码参考

本篇对应 `src/fsbdd/diloco/learner`。这里的“安全边界”专指一次 inner optimizer step 已完成、scheduler 已推进且梯度已清空之后的时刻；快照发布和全局参数采用都只能在这个边界进入训练线程。

## 1. `learner/runtime.py`

源码：[`runtime.py`](../../src/fsbdd/diloco/learner/runtime.py)

### 1.1 进度、事件与 scheduler

| 符号 | 代码内容 |
|---|---|
| `LearnerError` | learner 数据、RNG、训练或边界回调契约错误。 |
| `_nonnegative_int` | 严格拒绝 bool，并校验非负整数。 |
| `FragmentProgress.__post_init__` | 校验 `global_version/local_steps_since_adoption/processed_input_tokens_since_adoption/proposal_sequence` 四个单调计数器。 |
| `FragmentProgress.to_dict` | 用 dataclass 字段生成可序列化进度。 |
| `LearnerProgress.__post_init__` | 校验 learner id、三个总计数和非空 `FragmentProgress` 列表。 |
| `LearnerProgress.initialize` | 由初始 fragment version vector 构造全零本地进度。 |
| `LearnerProgress.from_dict` | 严格要求顶层和 fragment schema，恢复 checkpoint 进度；未知/缺失字段直接失败。 |
| `LearnerProgress.to_dict` | 序列化 learner 总计数及每片进度。 |
| `SafeBoundaryEvent.to_dict` | 深转 event，并把 tuple 向量改成 JSON list。 |
| `LearnerRunSummary.to_dict` | 序列化训练区间、吞吐、更新范数、进度、RNG evidence 和可选事件。 |
| `StepScheduler.step` | runtime 所依赖的最小 scheduler protocol；实现还必须暴露 `step_count`。 |
| `ConstantStepScheduler.__post_init__` | 校验恢复后的 step count。 |
| `ConstantStepScheduler.step` | 每个 optimizer step 后加一。 |
| `ConstantStepScheduler.state_dict` / `load_state_dict` | 以仅含 `step_count` 的严格 schema 保存/恢复。 |

### 1.2 learner 私有 RNG

| 函数/方法 | 代码内容 |
|---|---|
| `_rng_tensor_bytes` | 把 PyTorch RNG state tensor 复制为 CPU contiguous bytes。 |
| `_rng_state_tensor` | 由 bytes 经独立 NumPy copy 恢复 uint8 tensor。 |
| `LearnerRng.__post_init__` | 校验 learner/seed/device/state：CPU 不得有 device state；CUDA 必须有有效 device index 和 state。 |
| `LearnerRng.initialize` | 使用独立 CPU/CUDA `torch.Generator` 按 seed 生成初始状态，不污染进程全局 RNG。 |
| `LearnerRng.from_state_dict` | 严格解码 hex 状态和完整 schema。 |
| `LearnerRng.state_dict` | 导出设备绑定、激活次数和 CPU/device state hex。 |
| `LearnerRng.state_sha256` | 按“长度前缀+状态 bytes”对 CPU/device 状态计算 evidence digest。 |
| `LearnerRng.bind` | 一次性绑定 learner id 和实际 device；禁止同一 RNG 被第二个 runtime 复用。 |
| `LearnerRng.activate` | context manager：保存进程 RNG，安装 learner 私有状态，运行训练，捕获新私有状态，再确认进程 RNG 完整恢复；禁止未绑定或递归激活。 |

这使同进程中的多个串行 learner 仍拥有相互隔离、可 checkpoint 的随机流；它不是跨进程通信机制。

### 1.3 数据 shard

| 方法 | 代码内容 |
|---|---|
| `PackedTokenShard.__init__` | 调 `validate_materialized_shards` 验 manifest/文件；锁定 learner 对应 shard；以 `<u4` 只读 memmap 打开 `[blocks, sequence_length]`；恢复 epoch/position 并建立确定性 epoch permutation。 |
| `PackedTokenShard._epoch_order` | 使用 `default_rng(epoch_seed_base + learner_index + epoch)` 生成 block permutation。 |
| `PackedTokenShard.state_dict` | 导出 profile digest、learner index、epoch 和 position。 |
| `PackedTokenShard.next_batch` | 按 permutation 取够 batch；跨末尾自动推进 epoch；转成 int64 `input_ids`，复制为 labels，并生成全 1 attention mask。数据源因此可无限迭代。 |

### 1.4 batch helper

| 函数 | 代码内容 |
|---|---|
| `_distributed_initialized` | 同时检查 distributed 对象存在、available、initialized；learner 路径要求始终为 false。 |
| `_batch_counts` | 要求非空 rank-2 input/同形 labels；attention mask 只能是 0/1。`processed=mask.sum()`；loss-bearing target 是 shifted position 中 active 且 label≠-100 的数量。两者必须为正。 |
| `_extract_loss` | 从 attribute 或 mapping 取 `loss`；要求有限 scalar tensor。 |

### 1.5 `LearnerRuntime`

| 方法 | 代码内容 |
|---|---|
| `__init__` | 拒绝已初始化的 `torch.distributed`；要求 scheduler count 等于恢复进度；fragment groups 非空、不重叠并精确覆盖所有 trainable model parameters；optimizer parameters 也必须同集合且唯一；校验 accumulation、grad norm、fp32/bf16、observer 和 clock；CUDA bf16 不支持时禁止静默 fallback；最后独占绑定 RNG。 |
| `_clock` | 调注入的 monotonic ns clock 并校验非负。 |
| `_autocast` | fp32 返回 disabled autocast；bf16 在 CPU/CUDA 启用 `torch.autocast(dtype=bfloat16)`。 |
| `_move_batch` | 只接受五种模型输入字段，所有值须是 tensor；同步复制到 runtime device。 |
| `run` | 校验 stop/throttle 参数；记录 RNG 前态，在 `LearnerRng.activate` 中调用 `_run_owned`，返回时补入 RNG owner、前后 digest/activation 以及全局状态已恢复的 evidence。 |
| `_run_owned` | 完整 inner-training loop，细节见下。 |

`_run_owned` 每步严格执行：

1. 必要时克隆各 fragment 参数用于更新范数抽样；清零本步 loss/token 计数。
2. 取 `gradient_accumulation_steps` 个 batch，移动到设备并计数。若有 attention mask，把 padding labels 改成 `-100`，使模型 loss 的有效 target 与计数契约一致。
3. autocast forward；对每个 mean loss 反向传播 `loss * target_count`，累加精确 loss numerator。
4. 所有梯度统一除以本步总 target 数，再 `clip_grad_norm_(error_if_nonfinite=True)`；依次执行 optimizer、scheduler、`zero_grad(set_to_none=True)`。异常路径也先清梯度再抛出。
5. CUDA 同步；在抽样步按 fragment 计算参数差的 L2 norm。可选 `minimum_step_seconds` 只用于实验节流。
6. 增加 learner 总计数以及**所有 fragment** 的 since-adoption 步数/token；构造 `SafeBoundaryEvent`。此时梯度已清空。
7. 按配置顺序调用 safe-boundary observers。每个返回 mapping 或 None，metric key 不得碰撞；active key 会从 inactive placeholder 删除。之后记日志，再检查 stop policy。
8. 结束时要求至少一个边界、正观测区间、模型确有更新、scheduler 与 local step 一致、distributed 仍未初始化；生成聚合 summary。`retain_events=False` 时仅保留最后事件用于汇总而不驻留整个事件序列。

## 2. `learner/publication.py`

源码：[`publication.py`](../../src/fsbdd/diloco/learner/publication.py)

### 2.1 schedule 和 fragment 映射

| 函数/方法 | 代码内容 |
|---|---|
| `SnapshotPublishError` | 快照、调度或异步发布错误。 |
| `_positive_int` / `_nonnegative_int` / `_hex_identity` / `_positive_float` | publication 的严格基础 validator。 |
| `FragmentPublicationSchedule.__post_init__` | 要求 bytes/intervals/offsets 非空等长；bytes、interval 正，`0≤offset<interval`；算法名非空。 |
| `FragmentPublicationSchedule.unified` | 为统一周期 `H` 计算 byte-weighted circular midpoint；其局部 `objective` 计算候选 offset 到目标中点的环形距离，并以 offset 本身打破平局。在 `H≥F` 时用 nearest-free 保证同 learner 同步内不撞 due step，再施加 learner phase。 |
| `fragment_count` / `unified_h` | 返回片数；所有 interval 相同才返回统一 H，否则 None。 |
| `is_due` | 判定 `(local_step-offset) mod interval == 0`，step 必须从 1 开始。 |
| `due_fragments` | 按 fragment index 顺序返回本步所有 due 片。 |
| `frequency_budget` | 用 `Fraction` 精确计算 `Σ bytes_f/H_f`，并在统一 H 时给出每步 due bytes 和峰值。 |
| `to_dict` | 导出 schedule 加完整频率 evidence。 |
| `build_fragment_descriptors` | 重验 map digest；为每片按 map identity/index/parameter identities 派生 descriptor identity，dtype 固定 fp32、shape 为扁平参数个数。 |
| `build_fragment_parameter_groups` | 验 registry/map；按 registry 的 owner name 从 `named_parameters(remove_duplicate=False)` 取真实对象，保持 fragment 和 parameter identity 顺序。 |

### 2.2 snapshot 值对象

| 符号 | 代码内容 |
|---|---|
| `AdoptedFragmentBase.__post_init__` | 校验已采用 version 和 content SHA。 |
| `FragmentSnapshot.__post_init__` | 对已经 materialize 的 `Proposal` 加 staging 时间；验证 interval 和传输耗时。 |
| `FragmentSnapshot` 的 `identities/learner_id/descriptor/sequence/proposal_id/base_version/base_content_identity/local_steps/processed_tokens/snapshot_local_step/parameters` | 无复制地代理内部 proposal 字段。 |
| `StagedFragmentSnapshot.__post_init__` | 验 frozen identities、flat fp32 descriptor、sequence/base/progress、payload 精确字节数和 staging 时间顺序；此时尚未计算 proposal/content hash。 |
| `StagedFragmentSnapshot.proposal_id` | 由 learner、六位 fragment index、十二位 sequence 生成稳定可读 id。 |
| `StagedFragmentSnapshot.materialize_proposal` | 在后台调用 `Proposal.create`，计算参数 SHA 和 semantic content identity。 |
| `_PublishSlot` | 每片一个 `pending`、一个 `in_flight` 及对应 trace；`failed` 后终止该 worker。 |

### 2.3 `BoundedProposalPublisher`

| 方法 | 代码内容 |
|---|---|
| `__init__` | 冻结 store/learner/hooks；为每片创建一条 daemon worker 和一个 latest-wins slot；初始化有界 trace、计数、Condition 与 sink lock。resident snapshot 上界为每片 pending 1 + in-flight 1。 |
| `_new_trace` | 为 staged/materialized snapshot 建统一 trace；尚未后台 materialize 的 identity/hash 字段保留 None。 |
| `_materialize_proposal` | 已有 `FragmentSnapshot` 直接取 proposal；staged 值在 worker 中 materialize，并记录 hash 计算时间和结果。 |
| `_record_terminal_locked` | 在 Condition 下复制 terminal trace，更新每片 latest、outcome count；published 时累加次数、bytes 和耗时。 |
| `_emit_terminal` | 用独立 sink lock 串行调用 trace sink/logger；sink 异常进入共享 `_errors`，不会在 worker 中悄然丢失。 |
| `submit` | 验 snapshot 地址；sequence 不超过历史最高则终结为 `skipped_nonmonotonic`；若已有 pending，把旧 pending 终结为 `replaced_before_publish`；否则安装新 pending 并唤醒 worker。不会覆盖 in-flight。 |
| `_worker` | 每片串行取 pending→in-flight，后台 materialize、hook、store publish；成功记 `published`。任何异常记 `failed`，pending 记 `abandoned_after_failure`，保存首错并停止该 worker；terminal sink 发出完毕也纳入 drain 条件。 |
| `_raise_if_failed_locked` / `raise_if_failed` | 把首个后台异常包装为同步 `SnapshotPublishError`。 |
| `drain` | 等待所有 pending/in-flight 和 terminal emits 清空，期间传播首错，超时失败。 |
| `close` | 先 drain，再设置 closing、唤醒并 join 全部 worker；保留 drain 原始失败。 |
| `summary` | 返回每片占用、水位、skip/replacement/published、bytes、分阶段耗时、terminal outcome、错误和有界 trace evidence。 |

### 2.4 `FragmentSnapshotCoordinator`

| 方法 | 代码内容 |
|---|---|
| `__init__` | 对齐 identities、store descriptors、progress、parameter groups、adopted bases 和 schedule；要求 groups 非重叠且 numel/bytes 精确匹配 flat fp32 descriptor；可使用外部 publisher 或自行创建。 |
| `_validate_safe_boundary` | event 必须属于当前 learner/当前且递增的 step，所有进度向量与 mutable progress 相同，并确认所有 fragment 参数 `grad is None`。 |
| `on_safe_boundary` | 在 capture lock 内找 due 片；对每片**先预留并递增 proposal sequence**（失败可留 gap，但 checkpoint 后绝不复用），同步 GPU→CPU 序列化出 immutable fp32 bytes，创建 staged snapshot 并 submit；更新边界并返回 publication active metrics。 |
| `update_adopted_base` | 采用方未负责计数时使用：要求 version 严增，替换 base context，并同步更新 fragment version、把 since-adoption step/token 清零。 |
| `update_adopted_base_context` | 采用方已更新计数时使用：要求目标 version 已写入且两个 since-adoption 计数已为零，只替换内部 base identity。 |
| `drain` / `close` | 代理 publisher。 |
| `summary` | 汇总 identities、schedule、base vector、progress 和 publication。 |
| `fragment_payload_sha256` | 以 `serialize_fragment_parameters` 的规范 bytes 计算目标片精确 SHA，供 adoption audit。 |
| `serialize_fragment_parameters` | 按 parameter group 顺序 detach→CPU→fp32→contiguous，拒绝 NaN/Inf，统一转 little-endian `<f4` 并拼接；最终长度必须等于 descriptor 预算。 |

## 3. `learner/adoption.py`

源码：[`adoption.py`](../../src/fsbdd/diloco/learner/adoption.py)

### 3.1 audit helper 和待采用值

| 函数/方法 | 代码内容 |
|---|---|
| `AdoptionError` | polling、全局值验证或安全边界采用错误。 |
| `_nonnegative_int` / `_positive_float` / `_hex_identity` | adoption 基础 validator。 |
| `_tensor_bytes` | tensor detach 后复制为 CPU contiguous raw bytes。 |
| `_optimizer_value_identity` | 递归规范化 optimizer state：tensor 记 dtype/shape/bytes/SHA；mapping 按 key type+repr 排序；序列保序；支持有限 scalar/None，拒绝未知类型和非有限浮点。 |
| `optimizer_fragment_state_sha256` | 按 fragment parameter position 对每个 `optimizer.state.get(parameter,{})` 建 canonical digest。 |
| `optimizer_fragment_state_entry_count` | 统计该片已有非空 optimizer state 的参数数。 |
| `PendingFragmentAdoption.__post_init__` | state/record 必须一致绑定 run/map/fragment/version/shape/base；验证 discovery/fetch 时钟和耗时。 |
| `record_payload_identity` | 暴露 exact immutable global-state file SHA。 |
| `to_dict` | 输出 fragment/version/content/record/payload hash 及 discovery timing。 |

### 3.2 `LatestFragmentPoller`

| 方法 | 代码内容 |
|---|---|
| `__init__` | 冻结 store、初始 adopted version/content；每片保留一个 latest-wins pending、最高观察 record 和计数；支持后台线程或手工 poll。 |
| `start` | 幂等启动 daemon poller；closed 或已有后台错误时失败。 |
| `_record_observation` | fail-closed 检查 version 不回退、同 version content/record 不变；已采用或旧 pending 只计数；更高 version 替换该片 pending 并生成 discovery trace。 |
| `poll_once` | 每片先 metadata-only peek；未变化时不读大 payload。发现更高 version 时优先按 bound record 读取 exact file，再构造 pending；`PublicationNotReady` 是可重试瞬态，其他错误传播。发 trace/logger 后增加 poll cycle。 |
| `_worker` | 周期调用 `poll_once`；首个异常保存并停止整个 poller。 |
| `_raise_if_failed_locked` / `raise_if_failed` | 把后台首错同步传播。 |
| `take_pending` | 原子取走所有片的 pending 并把槽清空；之后若发现更高版本仍可重新填入。 |
| `mark_adopted` | version 必须严格增加且恰等于最高观察版本，content 必须匹配；推进 adopted vector，并丢弃不新于它的 pending。 |
| `summary` | 输出 adopted/highest/pending vectors、fixed-slot read/discovery/replacement/retry 等计数与线程状态。 |
| `close` | 幂等 stop+join，超时或保存的后台错都同步失败。 |

### 3.3 `FragmentAdoptionCoordinator`

| 方法 | 代码内容 |
|---|---|
| `__init__` | 对齐 store/progress/initial states/groups；groups 非空不重叠且 numel 匹配 descriptor；可启用 optimizer identity audit；创建 poller，并保存每片当前 content identity。 |
| `start` / `poll_once` | 代理 poller，便于后台或确定性手动驱动。 |
| `_validate_safe_boundary` | 与 snapshot coordinator 相同地检查 learner、递增 step、进度向量及 `grad is None`。 |
| `_decode_sources` | 验 global parameters 长度；按 `<f4` 零拷贝解析、检查有限性，再按本地 parameter shape 切分并 copy 成独立 CPU tensor。 |
| `_audit_hashes` | audit 未开启返回三个空列表；开启时为全部片计算参数 hash、optimizer-state hash 和初始化 entry count。 |
| `_apply_one` | 忽略已采用版本；解码 CPU sources；可选记录全片审计；在 `torch.no_grad` 下只把目标片同步 copy 到对应 device，并对涉及的 CUDA device synchronize；随后仅推进目标片 version、清零目标片 since-adoption counters，保留 proposal sequence 和 inner optimizer state；通知 snapshot base context 和 poller。审计确认非目标参数不变、fp32 目标等于全局 payload、所有 optimizer states 不变；生成版本跳跃和两段传输耗时 trace。 |
| `on_safe_boundary` | 在 coordinator lock 内传播 poller 错、验证边界、原子取 pending，逐片 `_apply_one`；更新 last boundary，计算 observed-adopted lag，返回 observer active metrics。混合版本向量在此自然形成。 |
| `summary` | 输出 progress/content vector、采用次数、跳过中间版本数、累计 FS→CPU/CPU→GPU、每片 latest trace 和 poller evidence。 |
| `close` | 代理 poller close。 |

## 4. 空初始化文件

`learner/__init__.py` 当前为空，不创建线程，也不 re-export 类。
