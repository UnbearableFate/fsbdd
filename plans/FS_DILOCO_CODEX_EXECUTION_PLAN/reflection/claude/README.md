# Claude Stage 1 结束审查（2026-07-16）

审查范围：`source/RESEARCH_PLAN.md` v1.3、`source/STAGE0-4_SPEC.md` v2.1、本执行包全部
docs/loops/plans、`src/fsbdd`（commit `a72e75e` 工作树）、`reports/`（stage0、stage1、
checkers）、`evidence/indexes`、`runtime_runs/S1-13` 的失败台账与关键 run 数据。

本目录是独立的 reflection，不修改任何 source spec、loop 卡或代码；所有条目按
"证据 → 判断 → 建议" 组织，供下一轮 ADR/计划修订取舍。

## 文件

| 文件 | 内容 |
|---|---|
| [01-system-design.md](01-system-design.md) | 训练系统设计问题：数据面放大、syncer 容量、内存、崩溃语义、研究准确率风险 |
| [02-code-review.md](02-code-review.md) | 代码实现与设计模式：DISC-01 违反点、重复校验、API 形态、可维护性 |
| [03-plan-process.md](03-plan-process.md) | 执行计划与流程：失败台账分类、9N 门禁成本、契约哈希过绑定、证据留存策略 |
| [04-research-plan-revision-notes.md](04-research-plan-revision-notes.md) | RESEARCH_PLAN.md 修订点 M1–M12：每条的位置、证据、问题与修改内容 |
| [05-RESEARCH_PLAN-v1.4-draft.md](05-RESEARCH_PLAN-v1.4-draft.md) | 应用全部修订后的完整 RESEARCH_PLAN v1.4 草案（非权威，采纳须走变更控制） |
| [06-spec-revision-notes.md](06-spec-revision-notes.md) | STAGE0-4_SPEC.md 修订点 SP1–SP10：由 M1–M12 与代码审查驱动的 Spec 级修改及理由 |
| [07-STAGE0-4_SPEC-v2.2-draft.md](07-STAGE0-4_SPEC-v2.2-draft.md) | 应用全部修订后的完整 STAGE0-4_SPEC v2.2 草案（与 05 配套采纳，非权威） |

## 最重要的十条发现（按影响排序）

| # | 发现 | 类别 | 详见 |
|---|---|---|---|
| 1 | S1-13 的 14 条失败台账中只有 3 条是训练系统真实缺陷，10 条是流程/契约/harness 自伤；证据机器本身成了最大的失败实验来源 | 流程 | 03 §1 |
| 2 | global fragment 以"参数+outer momentum+旧 base"单一复合 payload 发布（实测 309–340MB vs 154–170MB fragment），learner 采用被迫下载它不需要的 momentum，数据面读放大 2×，S_max=1 后升至 3× | 系统 | 01 §1 |
| 3 | syncer 严格串行（读→哈希→merge→写一次一个 fragment），单核处理，GH200 节点 72 CPU 只用 1 个；实测 1.06s/update，对 0.5–1B 模型外推后成为全局时钟瓶颈 | 系统 | 01 §2 |
| 4 | Stage 0-A 预注册锚点预测 stale 挽回为 **负值**（−0.0446pp，接受率 5.31%，accepted-token 效率仅 33.5%），H3 大概率不成立，但 Stage 4 仍按 6 个 loop + 每 loop 9N 门禁 + 多种子 H3 全量排程 | 研究 | 01 §6 |
| 5 | Stage-1 收窄以 `s_max != 0` fail-fast 硬编码在 ≥5 处（readiness/profile_a/global_commit/numpy merge/authority 断言），与 DISC-01"禁止独立代码路径"直接冲突，Stage 4 将变成多点改码而非"翻配置" | 代码 | 02 §1 |
| 6 | syncer 常驻内存 ∝ M×F×fragment_bytes（readiness 缓存整个 payload 的 Proposal），与 SYNC-06 的有界内存要求冲突；merge 的 streaming 记帐只覆盖了 merge 视角 | 系统 | 01 §3 |
| 7 | 每个实现 loop 一次 9N 门禁的义务对 Stage 2–4 约 12 次 9 节点 run；大多数 loop 的回归可由 1–2 节点 smoke 捕获，建议改为"stage 关闭 + 数据面语义变化 loop"触发 | 流程 | 03 §2 |
| 8 | `runtime_runs/S1-13` 已留存 857GB/102 个 run root（"不删除失败 run"无分层策略）；Stage 2 的 2h 慢 FS run 和 Stage 5 长跑会把共享 FS 配额吃穿 | 流程 | 03 §4 |
| 9 | 学习率/loss 门禁按预训练 checkpoint 微调场景校准不足：formal long run 的 0.99 ratio 门差 0.004 险些误判，runtime 预算余量仅 0.1–0.5%；smoke 的 59/60 cycle 目标已发生一次纯边界误报 | 研究 | 01 §7 |
| 10 | `minimum_step_seconds` 人为放慢 learner 以保门禁节奏，掩盖了真实速度下 latest-wins/quorum 行为与 syncer 容量余量，Stage 2 goodput 测量前必须移除 | 系统 | 01 §4 |

## 总体判断

架构方向与协议设计是扎实的：STOR-01 两原语抽象、payload-first/visibility-last、
consumption frontier 与参数/outer state 原子共同提交、latest-wins 固定发现面、
按字节均衡的 layer-aligned fragment map——这些与 spec 不变量一一对应，且大多有
反例级测试。Stage 0 的实测（可见性 p99 7.95ms、元数据 39.4k ops/s、10^5 次原子替换
零违例）给了论文一手数据。35 小时内 269 个 commit 关完 Stage 0+1 也证明执行包
"可被 agent 顺序执行"这一目标基本达成。

真正的风险集中在三处：**数据面字节放大与单线程 syncer**（决定 Stage 2/5 能否在
0.5–1B 规模成立）、**stale 假设的预测已经为负**（决定论文 C4 叙事与 Stage 4 投入），
以及**流程自伤成本**（决定剩余 23 个 loop 的日历时间）。三者都可以在进入 Stage 2
之前用少量 ADR 修正。
