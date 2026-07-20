# protocol 函数级代码参考

## 1. `protocol/storage.py`

源码：[`storage.py`](../../src/fsbdd/diloco/protocol/storage.py)

### 1.1 异常、数据类型与 Protocol

| 符号 | 代码内容 |
|---|---|
| `PublicationError` | 所有 storage publication/validation 错误基类。 |
| `PublicationNotReady` | fixed slot 或其 payload 之后可能可读，当前 deadline 内未完整。 |
| `PublicationNotFound` | `NotReady` 子类，slot 在 deadline 前不存在。 |
| `PublicationInterrupted` | crash injection point 主动模拟中断。 |
| `PublicationSpec` | writer 声明的 run/map/fragment/version/sequence/dtype/shape/base identity。 |
| `ReadExpectation` | reader 必须匹配的 run/map/fragment，及可选 version/sequence/dtype/shape/base。 |
| `PublicationRecord.to_dict` | 将 complete visibility metadata 转 dict。 |
| `PublishedPayload` | 已验证 record 与 payload bytes 对。 |
| `StorageBackend.publish/read` | 最小 storage protocol。publish 接受 visibility hook/crash point；read 按 expectation 和 deadline 返回完整 payload。 |
| `RecordReadableStorageBackend.read_record` | metadata-only 扩展，不读大 payload。 |
| `BoundRecordStorageBackend.read_bound_record` | 按已经冻结的 exact record 读 immutable payload，避免重新追 latest。 |

### 1.2 模块 helper

| 函数 | 行 | 行为 |
|---|---:|---|
| `_require_text` | 141–144 | 非空 string。 |
| `_require_hex64` | 147–150 | regex 校验 64 位小写 hex。 |
| `_require_ordinal` | 153–156 | 非 bool 非负 int。 |
| `_require_nonnegative_finite_real` | 159–167 | timeout/interval 用有限非负 float。 |
| `_require_shape` | 170–179 | tuple/list 的非负 integer shape。 |
| `_validate_spec` | 182–190 | 逐字段验证 `PublicationSpec`。 |
| `_validate_slot` | 193–196 | slot 必须符合 `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`。 |
| `_canonical_record_bytes` | 199–202 | sorted compact JSON+换行。 |
| `_parse_record` | 205–262 | JSON/schema/complete marker/field 类型/path/canonical bytes 全量验证；要求 payload identity=SHA，且 path 严格为 `payloads/*.bin` 两段。 |
| `decode_publication_record` | 265–272 | public metadata decoder；输入必须 bytes 且≤64 KiB，再调 `_parse_record`。 |
| `_read_regular_file` | 275–299 | 禁 symlink；`O_NOFOLLOW|O_CLOEXEC`；`fstat` regular file/maximum size；按 declared size+1 有界读取。 |
| `_read_visibility_record` | 302–315 | 以 64 KiB 上限安全读取并 decode record；保留 `FileNotFoundError` 供 poll 语义。 |
| `_validate_read_expectation` | 318–342 | 验 expectation 本身。 |
| `_validate_expectation` | 345–362 | 精确比较必选和所有非 None 可选字段；任何 mismatch fail closed。 |

### 1.3 `PosixStorageBackend`

| 方法 | 行 | 行为/副作用 |
|---|---:|---|
| `__init__` | 366–380 | 校验 readback flag，创建 `payloads/visibility/.record-tmp/.retired` 四个目录。 |
| `root` | 383–384 | 返回 backend root。 |
| `publication_verification_mode` | 387–392 | 返回 `source_sha256_complete_write` 或诊断 `post_write_readback`。 |
| `read_record` | 394–425 | 轮询固定 visibility path；不存在则睡眠至 deadline 后抛 `PublicationNotFound`；存在立即完整解析并匹配 expectation。格式错误不会被当作暂时缺失。 |
| `inspect_inventory` | 427–479 | 只解析 records/目录 metadata，不读 tensor；统计 current records、physical/referenced/orphan payload bytes、temp 和 retirement marker。坏 record 被跳过用于 best-effort inventory。 |
| `_retirement_times` | 481–499 | best-effort 读取 `.retired/*.json`，返回 payload relative path→失去引用的 unix ns；坏 marker 忽略。 |
| `_mark_retiring` | 501–529 | 在替换 current record 前原子写旧 payload 的 retirement timestamp marker；失败阻止 publication。 |
| `reclaim_unreferenced_payloads` | 531–621 | 重算 current references；按 mtime/name 从新到旧排序 orphan；无条件保留 newest `retain_recent`，并保留年龄小于 safety age 的项；best-effort unlink payload/marker/temp，返回回收量+当前 inventory。绝不删 referenced payload。 |
| `publish` | 623–761 | 核心 payload-first/visibility-last。校验 crash point；用 UUID+`xb`+unbuffered loop 完整写 payload并验 `fstat` size；默认 hash source，诊断模式再 bytewise readback；构造 record并写 temp；执行调用方 `visibility_hook`；为旧 record 标 retirement；`os.replace` current visibility；支持四个 crash injection point。大 payload 不 `fsync`，语义是完整可见性而非断电持久性保证。 |
| `read` | 763–807 | 先 `read_record`，再安全读 record 指定 payload；size/SHA 正确才返回；payload 暂时缺失/不完整则 poll 至 deadline 后 `PublicationNotReady`。 |
| `read_bound_record` | 809–841 | 先把 record canonical round-trip 验证，再检查安全 path，直接读 exact payload 并验 checksum；不访问 visibility latest；缺失/损坏为 `NotReady`。 |

## 2. `protocol/global_state.py`

源码：[`global_state.py`](../../src/fsbdd/diloco/protocol/global_state.py)

### 2.1 校验 helper 与数据对象

| 符号 | 行 | 代码行为 |
|---|---:|---|
| `GlobalStateError` / `BootstrapInterrupted` | 26/30 | compound state 错误；可恢复 bootstrap 注入中断。 |
| `_require_text/_require_hex/_require_ordinal/_require_shape/_require_bytes` | 40–75 | 基础严格类型/identity 校验。 |
| `GlobalStateIdentities.__post_init__` / `to_dict` | 84–92 | run 非空；config/model/map 是 SHA-256；导出四 identity。 |
| `FragmentStateDescriptor.__post_init__` / `to_dict` | 102–120 | index≥0、identity/dtype 非空、shape 非负；parameter identities 必须 tuple、非空值且 fragment 内唯一。 |
| `BootstrapFragment.__post_init__` | 128–136 | descriptor typed，parameters/outer_state 必须 immutable bytes。 |
| `BaseSnapshot.__post_init__` | 141–147 | version、bytes、SHA 合法并重新 hash parameters。 |
| `_verified_base_snapshot` | 149–161 | 只在外层 section checksum 已验证后，用 `object.__new__` 构造，避免同一大 bytes 再 hash；仍验元字段。 |
| `_new_base_snapshot` | 164–172 | 对新 bytes 计算 SHA 后走 verified constructor。 |
| `FragmentGlobalState.__post_init__` | 187–192 | 调结构验证，再 hash outer state、重算 content identity。 |
| `FragmentGlobalState._validate_structure` | 194–229 | 要求 version=outer update count；base history 非空、版本连续、以 current version 结束，末项 parameters 等于 current；只有 v0 使用零 previous base identity；各 identity/bytes 合法。 |
| `_verified_fragment_state` | 231–263 | section checksum 已验证后的低重复构造路径；设置 fields 后只做结构验证。 |
| `GlobalSnapshot` / `BootstrapResult` | 266–277 | 普通 state vector/digest；bootstrap 新发/已存在 indices 与 snapshot。 |
| `LiveSetReport.to_dict` | 280–290 | current/reference/base/physical/orphan/global-head 检查结果。 |

### 2.2 编解码函数

| 函数 | 行 | 行为 |
|---|---:|---|
| `current_slot(index)` | 293–294 | `global-current-%06d`。 |
| `_section` | 297–313 | 生成 section name/bytes/SHA，可带 base version；可复用已验证 checksum。 |
| `_semantic_header` | 316–351 | 生成 schema v2 state 语义。current base 是 parameters section alias；只有旧 bases 写额外 sections。 |
| `_make_state` | 354–379 | 先构造零 content identity provisional state，按 semantic header 算 digest，再冻结回对象。 |
| `encode_global_state` | 382–394 | 重验 content identity；canonical header≤16 MiB；输出 `FSBDDGS2`、8-byte big-endian header length、header、parameters、outer state、historical bases。 |
| `_parse_json_header` | 397–411 | 验 magic/length/canonical JSON，返回 header 和 body offset。 |
| `decode_global_state` | 414–555 | 精确验 schema/identity/descriptor/base alias/section 数和名字；按声明 length 切 body，每段 hash 一次且禁止 trailing bytes；用 verified constructors 组 bases/state；最后重算 semantic content identity。 |

### 2.3 storage wrapper 与 store

| 方法/函数 | 行 | 行为 |
|---|---:|---|
| `CountingStorageBackend.__init__` | 559–563 | 包装最小 backend，初始化 read/publish count。 |
| `.root` / `.reset_counts` | 565–571 | 透传 `Path` root；计数清零。 |
| `.publish` / `.read` | 573–584 | 分别计数后透传。该 wrapper 未实现 metadata/bound 扩展，因此上层会走通用路径。 |
| `GlobalStateStore.__init__` | 587–628 | 要求 runtime-checkable backend；冻结 identities/descriptors/s_max；descriptor index 连续、identity 唯一、跨 fragment parameter identity 不重复；初始化 verified-authority cache。 |
| `_remember_authority` | 630–639 | lock 内把 exact `PublicationRecord` 与 decoded state object 绑定到 fragment。 |
| `_require_verified_authority` | 641–652 | 提交时要求 cache record 相等且 state 是同一个 object (`is`)，阻止调用者伪造等值 current state。 |
| `_descriptor` | 654–659 | 非负 index lookup，越界给领域错误。 |
| `_expectation` | 661–668 | 从 frozen store 构造 current record expectation。 |
| `_validate_published` | 670–708 | record frozen fields、decoded compound identities/descriptor、record version=sequence=state version、base window≤`S_max+1`、previous base identity 全匹配。 |
| `load_fragment` | 710–715 | 调 publication variant，仅返回 state。 |
| `load_fragment_publication` | 718–731 | backend read current slot、验证、记 authority binding，返回 state+exact record。 |
| `supports_bound_record_reads` | 734–735 | backend 是否实现 `BoundRecordStorageBackend`。 |
| `load_bound_fragment_record` | 737–752 | 用已冻结 record 读 exact payload、验证并记 authority，不重新追 current。 |
| `peek_fragment_record` | 754–770 | 有 metadata extension 时只读 record，否则读 compound payload 只取 record。 |
| `load_snapshot` | 772–786 | 顺序加载 F states；version vector 和 state content identities 形成 snapshot digest。它是一次顺序读，不声称跨 fragment 原子。 |
| `_bootstrap_state` | 789–800 | 生成 v0 base/state，previous base identity 为零。 |
| `bootstrap` | 802–882 | 要求 exact ordered fragments；先逐 slot 区分 identical existing 与 missing；对 missing 设置局部 `refuse_overwrite` visibility hook，在 record 切换前再次读取 slot，区分并拒绝 identical/conflicting race；发布 v0；可按已发布数量注入中断；最后 load snapshot。 |
| `publish_successor` | 884–903 | 自己加载 current+record，委托 from-current 版本，只返回 successor。 |
| `publish_successor_from_current` | 906–1000 | 验 supplied current 属于 store、semantic digest、expected record 和 verified object binding；publication 前 peek CAS；创建 v+1 和 bounded history；局部 `require_unchanged_base` visibility hook 再验 current 未变化，执行可选上层 hook后第三次验；发布 compound state并缓存 binding。 |
| `inspect_live_set` | 1002–1034 | 读取每个 current state，统计 retained base；若 root 是 Path，再比较 physical payload references 并检查是否错误出现 shared `global-head.json`。 |
| `load_bootstrap_plan` | 1045–1149 | 严格解析计划 JSON；构造 identities/s_max/descriptors；安全 resolve 相对 parameters/outer paths，禁止逃逸，验文件 checksum；要求 fragment index 非空连续；返回 plan 和 canonical digest。其内部 `resolve_input` closure 实现路径安全检查。 |

## 3. `protocol/proposal.py`

源码：[`proposal.py`](../../src/fsbdd/diloco/protocol/proposal.py)

### 3.1 Proposal 编解码

| 符号 | 行 | 行为 |
|---|---:|---|
| `_require_text/_require_hex/_require_integer/_require_ordinal/_require_positive/_require_bytes` | 39–75 | proposal 严格 validator。 |
| `_f32` | 78–90 | IEEE float32 pack/unpack canonicalize，拒绝 overflow/nonfinite/bool。 |
| `Proposal.__post_init__` | 108–127 | 验全部 metadata；local/progress 允许构造时为任意 int，资格/发布层再要求正；hash parameters；重算 semantic content identity。 |
| `payload_bytes` / `payload_identity` | 130–135 | 参数 bytes 长度/SHA property。 |
| `Proposal.create` | 138–188 | public高效 factory：先一次 hash 大 parameters，计算 semantic identity，再用 `object.__new__` 避免 dataclass `__post_init__` 第二次 hash。 |
| `_proposal_semantic_values` | 191–225 | 生成不含 content identity 的 proposal semantic，payload kind 固定 `complete_local_parameters`。 |
| `_proposal_semantic` | 228–242 | 从 instance 调 values helper。 |
| `encode_proposal` | 245–256 | 重验 semantic digest，输出 `FSBDDPR1 + len + canonical header + parameters`。 |
| `decode_proposal` | 259–364 | 验 magic/header/canonical/schema/descriptor/payload metadata；取 remaining bytes 作为 parameters；构造 public `Proposal`，因此执行 payload SHA 与 semantic identity 校验。 |
| `proposal_slot` | 367–371 | learner id SHA-256 + fragment index 生成固定 slot。 |

### 3.2 `ProposalStore`

| 方法 | 行 | 行为 |
|---|---:|---|
| `__init__` | 378–423 | 冻结 backend/identities/descriptors/learner ids 和 progress caps；要求 descriptors/learners 唯一有序；为每 learner×fragment 创建一个 publication lock。 |
| `supports_bound_record_reads` | 426–427 | backend capability probe。 |
| `_descriptor` / `_validate_address` | 429–441 | index lookup；要求 learner 在 frozen membership。 |
| `_expectation` | 443–450 | 生成 record frozen identity/dtype/shape expectation。 |
| `_validate_published` | 452–486 | record frozen fields，decode compound proposal，再验 identities、learner、descriptor、sequence、base version/content 与 record。 |
| `load_latest` | 488–501 | 读指定 fixed slot 并完整 decode。 |
| `peek_latest_record` | 503–523 | 优先 metadata-only read，否则读 payload 只取 record。 |
| `load_bound_record` | 525–537 | exact immutable record read+compound validate。 |
| `load_latest_if_changed` | 539–582 | metadata backend 先比较 `payload_identity`；相同返回 `(record,None)`；变化时优先 bound read。无 metadata backend 必须先完整读再比较。 |
| `discover_latest` | 584–597 | 双循环 learner/descriptors，忽略 absent slot，返回当前全部 proposals；复杂度固定 `M×F`。 |
| `_validate_for_publish` | 599–618 | proposal/store frozen identity、正 local/tokens/snapshot step 和 caps。 |
| `_in_flight_lock` | 620–621 | 定位 learner×fragment lock。 |
| `publish` | 623–691 | 非阻塞取得单 slot lock；检查 current record sequence/base 不倒退；相同 sequence 若 content 同则幂等返回，否则冲突；encode 后交 backend；hook 传 proposal；核对返回 sequence；finally release。 |

### 3.3 资格、选择、消费与权重

| 符号 | 行 | 行为 |
|---|---:|---|
| `ConsumptionFrontier.__post_init__` | 699–705 | sequence/base 初始允许 -1，其余不得小于 -1。 |
| `ConsumptionFrontiers.__post_init__` | 715–731 | identities/descriptor typed；learner ids 唯一非空；entries 与 learners 等长。 |
| `ConsumptionFrontiers.empty` | 734–746 | 为 frozen learners 建全 `(-1,-1)`。 |
| `ConsumptionFrontiers.for_learner` | 748–752 | 按 frozen learner tuple 定位，未知拒绝。 |
| `RetainedBaseIdentity.__post_init__` | 760–762 | base version≥0、content SHA。 |
| `EligibilityPolicy.__post_init__` | 780–823 | 冻结 identities/descriptor/membership/current/bases/caps/`S_max/Q/Q_fresh/max`；要求 `0≤Q_fresh≤Q≤max≤M`；retained versions 有序唯一、不未来、数量≤`S_max+1`；lambda 有限非负。 |
| `EligibilityPolicy.retained_identity` | 825–829 | 线性查 version，未保留返回 None。window 是小常数。 |
| `SelectionResult` | 833–836 | `ready`、ordered selected tuple、`proposal_id→reason`。 |
| `_identity_reason` | 839–869 | 顺序检查 run/config/model/map/fragment index+id/dtype/shape/parameter ids/learner；不重 hash immutable proposal bytes。 |
| `_eligibility_reason` | 872–904 | 依次检查 identity、consumed sequence/base、正 progress、caps、future/too stale、base retained、base identity；返回首个稳定 reason code。 |
| `_candidate_key` | 907–920 | 用 exact `Fraction` 计算 effective tokens；排序 key：staleness 升序、effective tokens 降序、sequence 降序、learner id、proposal id。 |
| `select_candidates` | 923–973 | 验 frontiers/policy identity；proposal id 全局不得重复；不合格记录 reason；每 learner 只取 key 最小，其余标 `duplicate_learner`；全局再排序；distinct 和 fresh 达门槛才 ready，且只在 ready 时截 `max_contributors`。 |
| `commit_consumption` | 976–1021 | publication 未成功原样返回；成功时 selected learner 不重复、身份/fragment/member 匹配，sequence/base 必须严格越过旧 frontier，然后只替换这些 learner entry。 |
| `CandidateWeight.__post_init__` / `to_dict` | 1033–1052 | proposal/learner/tokens/staleness 合法；raw/normalized 必须已经是 canonical float32，raw>0、normalized∈[0,1]；导出 fields。 |
| `compute_candidate_weights` | 1055–1098 | 非空 proposals；lambda float32 非负；每项 `raw=f32(f32(tokens)/f32(1+f32(lambda*s)))`；total 按顺序逐项 f32 累加；normalized 再 f32 除法。拒绝 future/nonpositive tokens。 |

## 4. `protocol/global_commit.py`

源码：[`global_commit.py`](../../src/fsbdd/diloco/protocol/global_commit.py)

### 4.1 基础校验与 envelope

| 符号 | 行 | 行为 |
|---|---:|---|
| `_require_text/_require_hex/_require_integer/_require_nonnegative/_require_bytes/_f32` | 41–83 | commit metadata 和 canonical float32 validator。 |
| `CommittedContribution.__post_init__` | 99–119 | 验 proposal/content/base/parameter SHA、正 tokens/payload bytes、非负 sequence/base/staleness、canonical非负 normalized weight。`to_dict` 再明确 f32 weight。 |
| `CommitEnvelope.__post_init__` | 140–146 | 结构验证后 hash outer optimizer bytes。 |
| `CommitEnvelope._validate_structure` | 148–217 | 验 frontiers/selected/policy/provenance identities；learner/proposal 唯一；非空 commit 要 previous/selection/update 非零、frontier 等于 selected sequence/base、f32 weight sum≈1；空 bootstrap 要三者全零；最后重算 envelope identity。 |
| `_verified_commit_envelope` | 219–255 | 已有 outer SHA 时避免再 hash；生成 semantic、直接构造并做结构验证。 |
| `_frontier_rows` | 258–268 | 按 frozen learner order 序列化 frontier rows。 |
| `_envelope_semantic_values` | 271–298 | 生成 envelope semantic，包括 selected、frontiers、outer bytes/SHA；可复用已验证 SHA。 |
| `_envelope_semantic` | 301–311 | instance adapter。 |
| `make_commit_envelope` | 314–334 | hash outer optimizer state 后构造 verified envelope。 |
| `encode_commit_envelope` | 337–353 | 重验 identity；输出 `FSBDDCM1 + len + canonical header + outer optimizer bytes`。 |
| `decode_commit_envelope` | 356–457 | 验 magic/header/schema/canonical；验 outer section length/SHA；typed rebuild frontiers 和 contributions；verified 构造并核对 serialized envelope identity。 |

### 4.2 authority 与 request

| 符号 | 代码行为 |
|---|---|
| `AtomicFragmentAuthority.__post_init__` | state/envelope typed，frontier identities/descriptor 匹配；Stage 1 version=outer count；只有 v0 selection 为空；每个 committed proposal payload bytes 等于 fragment parameters bytes。 |
| `authority_identity` | 对 state content/version 和完整 frontier rows 做 canonical digest。 |
| `version/content_identity/parameters/outer_optimizer_state/frontiers` | 直接从 state/envelope 暴露的只读 property；`outer_optimizer_state` 返回 envelope 内裸 momentum bytes，不是整个 encoded envelope。 |
| `AtomicGlobalSnapshot` | ordered authorities、version vector、digest。 |
| `AtomicBootstrapResult` | published/existing indices 和 atomic snapshot。 |
| `AtomicCommitRequest.__post_init__` | index/current/next/bytes/identity 验证；next 必须 current+1；selection fragment/version/数量匹配；每个 weight facts 等于 proposal，f32 sum≈1；policy/update SHA。 |
| `AtomicCommitResult` | resulting authority、是否实际 publish、是否 duplicate retry。 |

### 4.3 `AtomicGlobalCommitStore`

| 方法 | 行 | 行为 |
|---|---:|---|
| `__init__` | 615–638 | 冻结 ordinary store、unique learner membership、policy identity；明确要求 `store.s_max==0`；初始化 reentrant lock 和 record→authority cache。 |
| `_empty_envelope` | 640–655 | 为 descriptor/initial outer bytes 构造空 frontiers 和全零 bootstrap provenance。 |
| `bootstrap` | 657–675 | 将每个 bootstrap fragment 的 raw outer state 包进 empty envelope，再普通 bootstrap；返回 atomic snapshot。 |
| `load_fragment` | 678–680 | lock 内调用 cached loader。 |
| `_load_fragment_locked` | 682–704 | 先 peek record；cache record 未变直接返回 authority；否则优先 bound state read，decode nested envelope，验证 learner membership/policy，构造并缓存 authority。 |
| `load_snapshot` | 707–722 | 加载所有 authorities；以 authority identities 和 vector 计算 digest。 |
| `_committed_contributions` | 725–744 | 把 frozen selection proposals/weights 一一转成 envelope facts。 |
| `_matches_retry` | 747–766 | 比较 visible next authority 与 request 的 version/base/current parameters/outer state/policy/previous authority/selection/update/selected facts，全部相同才是幂等 retry。 |
| `_validate_selection_against_authority` | 768–813 | authority identity/fragment/version 必须是 selection 冻结点；selected identities/descriptor/member 唯一；frontier 未消费；Stage 1 要求 proposal fresh 且 base content=current content；progress 正。 |
| `commit` | 816–827 | 整个 commit 置于 store `RLock`，委托 `_commit_locked`。 |
| `_commit_locked` | 830–891 | 验 request/policy；加载 current；若已经 next version只接受 exact retry；否则 current version/content、参数 bytes 和 selection 必须匹配；先计算新 frontiers 和 envelope；要求 cache 仍绑定同一 current；调用 CAS 风格 state successor publication；缓存 committed authority；最后用 `_matches_retry` 自验并返回。 |

## 5. 空初始化文件

`protocol/__init__.py` 当前为空，不注册 backend 或做 re-export。
