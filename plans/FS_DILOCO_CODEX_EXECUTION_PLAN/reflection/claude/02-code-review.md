# 02. 代码实现与设计模式审查（src/fsbdd）

总体：代码质量高于典型研究代码——全量 typed frozen dataclass、fail-closed 校验、
显式错误类型分层、production/auxiliary 单向依赖 + 架构测试、355 个测试对应到
acceptance ID。以下问题多数是"高压快速迭代留下的结构债"，趁 Stage 2 开始前
处理成本最低。

## §1 DISC-01 违反：S_max=0 已经硬编码成了独立代码路径

Spec DISC-01 明文："`S_max = 0` 不得成为独立代码路径；禁止把 base 必须等于当前
版本硬编码进数据模型或聚合逻辑。" 现状（均为 fail-fast 守卫或语义断言）：

| 位置 | 内容 |
|---|---|
| `diloco/syncer/readiness.py:74` | `ReadinessConfig` 拒绝 `s_max != 0` |
| `diloco/syncer/profile_a.py:100,508` | Profile A 与 executor 双重拒绝 |
| `diloco/protocol/global_commit.py:630` | `AtomicGlobalCommitStore` 拒绝 `store.s_max != 0` |
| `diloco/protocol/global_commit.py:806` | `_validate_selection_against_authority`：`base_version != authority.version → "not fresh"` |
| `diloco/syncer/merge.py:821` | NumPy merge 拒绝非 current-base contribution，且不接 `base_source` |
| `diloco/protocol/global_commit.py:474` | `outer_update_count == version` 断言（Stage 1 专属） |

通用机制确实存在且是参数化的（`proposal.py:_eligibility_reason` 完整实现
`0<=s<=S_max`、拒绝原因分类齐全，STALE-10 的 telemetry 分类 Stage 1 就位——这部分
做得很好）。但上表意味着 Stage 4 的 "打开一个窗口" 实际是 **≥6 处协同改码 +
NumPy merge 增加 base_source 路径 + 门禁重跑**，消融两臂"只差配置值"的承诺
（DISC-03）已经形式化失效。

建议：
1. 立即把守卫改为由单一配置能力位驱动（如 `ReadinessConfig.s_max` 直接透传，
   commit store 只校验 `s_max == store.s_max` 一致性而非 `== 0`）；
2. `execute_numpy_streaming_fragment_update` 补上与 torch 版对等的 `base_source`
   参数（torch 版已支持，S0C 的 s=1 golden case 只覆盖 torch 路径，而生产 syncer
   用的是 numpy 路径——**Stage 4 验收将测不到生产代码**，这是最实质的一条）；
3. 在 Stage 2 期间用 s_max=1 的单元测试先行打通（不必等 S4-01），保证"接口通用"
   不是纸面承诺。

## §2 校验逻辑的三重/四重重复执行

防御式校验本身正确，但同一事实被反复证明，直接费用在 syncer 关键路径：

1. **哈希遍数**：一个 proposal payload 从产出到消费被完整 SHA-256 ≥4 次
   （learner 构造 `parameters_sha256`、publish 复合 payload 哈希、syncer read 复合
   校验、commit 后 `_matches_retry` 还做 165MB bytes 等值比较两次）。全链路应当
   收敛为"每个不可变边界一次"：写侧一次、读侧一次，之后凭对象身份传递（台账里
   已经修过两处同类问题，剩余的建议一次审完，登记每条路径的哈希预算）。
2. **canonical JSON 双向验证**：`_parse_record` 解析后重新 canonical 化并
   bytes 等值比较（`storage.py:260`）、`decode_commit_envelope` 同样
   （`global_commit.py:376`）。对 64KB 记录成本可忽略，模式没问题；列出只为说明
   "校验风格是统一的、可以信任"，无需修改。
3. **`__post_init__` 全量重算**：`BaseSnapshot`、`ResolvedBase`、`CommitEnvelope`
   构造即重哈希大 payload；代码里已经用 `_verified_*` 私有构造器绕开热路径
   （`global_state.py:149`、`proposal.py:181`），但 `ResolvedBase`
   （`merge.py:257-263`）没有同类出口——Stage 4 的 stale merge 每个 contribution
   打开 base 都会重哈希同一个 base 一次（M 次/update）。建议给 base source 增加
   prevalidated 路径，与 `_SelectedProposalSource.prevalidated_payload_integrity`
   （`profile_a.py:421`）对齐。

## §3 存储 API 形态：`bytes` 全量物化

`StorageBackend.publish(slot, payload: bytes, ...)` / `read() -> PublishedPayload`
强制整个 payload 以 Python bytes 驻留（`storage.py:99-117`）。后果：

- 每次读写至少一份全量拷贝（`np.frombuffer(...).copy()` 再加一份，
  `merge.py:786`）；
- 无法 mmap、无法边读边哈希边累加、无法 sendfile；
- 1–2GB fragment 时代价与 GC 压力显著。

建议在 Stage 2（异步管线 loop 本来就要动这层）把接口演化为：写侧接受
`memoryview | Iterator[bytes] | Path`，读侧返回可 mmap 的只读 buffer；校验语义
不变。这是支撑 01 §2 流水线化的前提。

另：`read()` 的重试循环（`storage.py:779-807`）每次 poll（默认 10ms）都完整重读
+ 重哈希 payload。该分支只在"记录可见但 payload 尚不完整/丢失"的异常窗口触发，
正常路径不受影响，但慢 FS 注入（Stage 2 A-PERF-02）下会变成 CPU 自旋放大器——
建议失败后按指数退避，并区分 "size 未到" 与 "hash 不符" 两种等待。

## §4 重复的 `_require_*` 校验函数族

`_require_text/_require_hex/_require_ordinal/_require_bytes/...` 在
storage.py、global_state.py、global_commit.py、readiness.py、merge.py、proposal.py
至少 6 处近似复制，仅错误类型不同。对 agent 维护的实际伤害已经发生过：新增
identity 字段时要同步 N 处 schema/validator/fixture（台账条目 2、3 的直接根因
之一）。建议收敛到 `diloco/common/validation.py`，以异常类型为参数；一次机械
重构，回归风险低。

## §5 控制面/数据面分层与 auxiliary 的规模

- `close.py`（1316 行）同时承担：契约装载、role 协调（JSON 轮询）、learner/syncer
  主循环、评估、GC 调度、结果打包。它是事实上的生产入口，却位于 auxiliary——
  "production 不得 import auxiliary" 的边界成立，但**生产运行时行为有一半住在
  auxiliary 里**（role 协调、终止条件、GC 节奏都会影响 Stage 2/3 的验收行为）。
  建议把 learner/syncer 的 role 主循环提升为 `diloco/runtime/`（或 `roles/`）的
  生产模块，auxiliary 只留契约/证据/打包。否则 Stage 3 的 kill-restart 语义将
  依赖 auxiliary 代码的行为，Checker 口径会混乱。
- `auxiliary/stress/` 12 个文件 ~14k 行，多数是一次性 RED/HARDEN harness。建议
  按 loop 归档（移入 `tests/` 或标注 retired），减少后续 session 的搜索噪声。

## §6 较小的正确性/健壮性观察

1. `PosixStorageBackend._mark_retiring`（`storage.py:501`）在每次 publish 关键路径
   上多做 3 次元数据操作（读旧记录已经必须，另加 temp 写 + rename）。Lustre 实测
   元数据余量大（39k ops/s），当前无碍；若 Stage 5 F、M 扩大可改为把 retire 时间
   写进新可见性记录本身（免独立 marker 文件）。
2. `reclaim_unreferenced_payloads` 的 `retirement_times` 每次 GC 全量重读
   `.retired/*.json`；有界（∝槽位数）但会随 fragment/learner 数线性增长，
   与 inventory 间隔相乘后注意别回到 O(历史) 扫描。
3. `AtomicGlobalCommitStore` 的 authority cache 竞态已由 RLock 修复（台账条目 8），
   修复正确；但注意该 cache 的 `is` 身份检查（`global_commit.py:874`）意味着
   多线程调用方必须持锁取 authority——如果 Stage 2 异步化 syncer，这个隐式契约
   要文档化或收进 API。
4. `LearnerRuntime.run` 的 `events` 列表默认 `retain_events=True`（`runtime.py:575`）
   是无界内存；生产端已显式传 False，但默认值应当反过来（安全默认），台账里
   "隐藏容器无界增长"教训已出现两次。
5. `_wait_json`/`_wait_roles`（`close.py:113-141`）用 0.05–0.1s 轮询共享 FS 做
   role 协调，其正确性依赖与数据面相同的 close-to-open 语义，没问题；但 JSON 解析
   失败被当作"未就绪"重试（合理，部分写入），**没有超时区分损坏与缺席**——role
   协调文件如果真损坏会等满 timeout 才报错，kill-restart 矩阵（Stage 3）建议给
   这条路径也加一个反例。
6. `torch.use_deterministic_algorithms(True)` + 逐 step `cuda.synchronize` +
   eager attention：对回归可复现性正确；Stage 2 goodput 基线必须重新评估这三项
   （goodput 分母"关闭通信的本地吞吐"也要带同样设置，否则 95% 门槛不可比）。

## §7 值得保留/推广的模式（正面清单）

- 可见性记录 = 唯一原子点、payload 内容寻址 + 全 fail-closed 读侧校验：与 STOR-01
  的映射干净，对象存储实例化路径清晰。
- consumption frontier 与参数/outer state 同记录提交（`CommitEnvelope`）：
  REC-02 的"读 current 引用即恢复全部权威状态"因此是真的，Stage 3 会很省。
- `FrozenSelection` 携带 selection_identity/authority_identity 并在 commit 时复验：
  SYNC-09 的选择稳定性有了机器可查的证据链。
- 拒绝原因枚举（`_eligibility_reason`）Stage 1 即齐全，Stage 4 的 STALE-10
  telemetry 只需接线。
- 架构测试（`tests/stage1/test_source_architecture.py`）锁依赖方向，避免 auxiliary
  反向渗透——建议 Stage 2 加一条"production 不 import numpy 之外的重依赖"同类断言。
