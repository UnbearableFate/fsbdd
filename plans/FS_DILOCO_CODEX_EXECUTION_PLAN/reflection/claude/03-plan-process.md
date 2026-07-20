# 03. 计划与执行流程审查（agent 实现效率）

数据基线：269 个 commit 在约 35 小时内完成 Stage 0 + Stage 1（07-14 17时 →
07-16 04时）；S1-00..S1-12 十三个 loop 约一天走完；S1-13 单 loop 消耗 15+ 个 PBS
实验、14 条失败台账、857GB 证据。计划的"可被 agent 顺序执行"目标总体达成，
瓶颈清晰地集中在收尾整合与流程自重上。

## §1 失败台账的构成证明：流程自伤是第一大成本

对 `reports/stage1/S1-13-experiment-failures.json` 14 条条目按根因分类：

| 类别 | 条目 | 数量 |
|---|---|---|
| 训练系统真实缺陷 | #1 发布路径冗余（双存 base/重复哈希/训练线程哈希/全量回读）；#8 authority cache 竞态；#9 publish 二次全量回读 | 3 |
| 测试/fixture 缺陷 | #2 七个回归期望未同步；#3 fixture 缺字段 + 观察列表无界饿死 writer；#4 miniature fixture 缺文件 | 3 |
| harness/脚本缺陷 | #6 bash ERR trap 吞掉预期非零；#10 config 拷贝改变路径基准；#5 asset 无 producer commit 绑定 | 3 |
| 契约/门禁设计缺陷 | #11 smoke 阈值 59/60 结构性不可达；#12 跨契约哈希绑定把历史契约拖下水；#13 非正式 smoke 继承正式 loss 门槛 + 环境变量未传播 | 3 |
| 环境 | #7 PBS 站点默认 `-r n` 拒绝 array | 1 |
| 用户范围变更 | #14 早停 | 1 |

**10/13 的非用户失败来自计划自身的证据/契约/harness 机器**，每次失败都要支付
完整仪式：immutable 失败包 + 台账评审 + 修正 diff + 静态门 + （多数时候）同一
Checker 重授权 + 新的排队。用户在 2393139 之后批准的 recovery ADR
（`S1-13-recovery-process-adr.md`：免每候选 Phase A、免手工哈希传入、运行时自记
identity）正是对这个成本的正确回应——**建议把该 ADR 从"一次性例外"升格为
plan 级默认策略**，具体见 §3。

## §2 每 loop 9N 门禁：义务范围应当收缩

现行规则（AGENTS §5、EXECUTION_GATES X-9N-TOPOLOGY）：S1-13 之后每个实现 loop
（或其 gate 单元）都要一次全新 9N run。Stage 2–4 共 15 个 loop、3 个双 loop
单元 → 约 **12 次 9N**。观察：

1. 9N run 验证的是拓扑 + loss 趋势 + runtime 预算——本质是系统冒烟回归。S1-13 的
   经验是：所有真实缺陷都先被 1–5 节点复现或静态审查抓到，9N 从未首发发现问题
   （首个 formal 9N 一次通过；后续失败全部发生在更小拓扑或提交前）。
2. 多数 loop 不改变跨节点行为（S2-04 telemetry、S2-05 调参、S3-01 learner 本地
   恢复、S4 的 weighting/selection 纯 syncer 逻辑），2 节点 smoke 的覆盖几乎等价。
3. 每次 9N 的实际成本 = 排队 + 9 GH200 节点占用 + 证据打包 + Checker 复算 +
   （失败时）完整台账仪式。

建议：把义务改为——
- **每个 stage 关闭一次 9N**（stage checkpoint 的一部分，双重用途允许）；
- **仅当 loop 改动数据面协议语义、发布/采用路径或拓扑相关代码时**追加 loop 级 9N
  （由 loop 卡在 ORIENT 时声明并被 Checker 复核，替代现在的"任何运行代码变化都
  触发"）；
- 其余 loop 的完成定义降为 2 节点真实 smoke + 完整测试套件。
这保留了"真实系统每阶段回归"的价值，把 12 次 9N 压到约 5–6 次。

## §3 契约哈希过绑定：从"文件身份"退到"语义断言"

台账 #12 是标本：改一个前瞻性阈值 → gate 契约 SHA 变化 → 一个静态测试要求
**历史** reproduction 契约追踪当前哈希 → 冒烟在 pytest 预检失败。同类还有 #5
（asset 无 producer 绑定，反向的身份缺失）与 #13（非正式 run 继承正式阈值）。
S1-13 期间已作的修正方向正确（语义检查替代跨契约哈希、历史契约不再重绑定、
运行时自记 identity），建议成文为 plan 级规则：

1. 哈希绑定只用于"消费关系"：run 消费什么就绑定什么（config、asset marker、
   代码 commit）；**不为"同一 loop 的其他文档"建立哈希互指**；
2. 已完成/历史契约永不因当前编辑重绑定；
3. 阈值、拓扑、schema 用语义断言测试（值、结构、单调关系），文件哈希只出现在
   evidence manifest；
4. 调用方不得手工传入仓库文件哈希（已在 recovery ADR 中，收进 AGENTS §4.1）。

## §4 证据留存：857GB 没有分层策略

`runtime_runs/S1-13` = 857GB / 102 个 run root，其中大头是失败/中间 smoke 的
全量 payload（单次 smoke 的 shared root 就是 67GB）。"不删除失败 run"（docs/03
§3.10）没有区分**结论所需证据**与**可再生字节**。照此策略，Stage 2 的 2h 慢 FS
run 与 Stage 5 的 1000+ update 长跑会产生 TB 级死重，直接威胁共享 FS 配额——
而配额耗尽会杀掉的恰恰是正式实验。

建议新增 `docs/14_EVIDENCE_RETENTION.md`：
- 永久层：manifest、checksums、logs、role 记录、analysis、失败台账引用的一切小
  文件（<100MB/run）；
- 可回收层：payload/visibility 目录、dataset 副本——台账评审完成 + Checker 结论
  落盘后 N 天可回收，回收动作本身写入 index（保留 checksum，字节可再生或不再需要）；
- 每次 stage checkpoint 附带一次留存审计（当前用量、可回收清单）。

## §5 loop 粒度：收尾 mega-loop 是结构性风险

S1-00..S1-12 单调顺滑（每 loop 一天内、Checker 一次过为主），因为它们是"单一
失败事实"的真 loop。S1-13 违背了 docs/03 §3.1 自己的定义——它同时承担：首次
真实多节点整合、首个 9N 基准、asset 供应链、M=4 长跑、容量验证、Stage 1 关闭
审计。六件事互相阻塞，任何一处失败都在最贵的资源等级上支付。Stage 2–4 的收尾
loop（S2-05、S3-04、S4-06）有同样的形状。

建议：
1. 把"首次真实整合 smoke"从"stage 关闭"中拆出（例如 S2-00 式的先导 loop 在低
   资源级别打通管线，S2-05 只做正式测量与关闭）；
2. 长跑/容量类验证前**必须先有单节点合成基准**：S1-13 的 syncer 1.7s/update 完全
   可以在 1 节点上用合成 payload 在提交任何 5 节点 run 之前测出（一个 30 行
   benchmark 能省掉两次五节点失败 + 两轮台账）。已有的 capacity preflight 是事后
   补的，应制度化为"任何频率/预算门禁 run 之前，其速率假设必须有单节点微基准
   支撑"（写入 AGENTS §4.1 admissibility）。

## §6 文档冗余与单一事实源

同一约束出现在 3–4 处：9N 门禁（AGENTS §5 / EXECUTION_GATES / docs/06 / README
§7）、token 纪律（AGENTS §4.3 / docs/03 §3.11 / docs/13 §D）、loop 状态机
（AGENTS §4 / docs/03 / 模板）。S1-13 期间修订 early-close 就要同步改 5 个文件
（RESEARCH_PLAN、SPEC、loop 卡、PROGRESS、CHANGELOG）——这次做对了，但每次
修订的一致性成本是 O(副本数)。建议：每类规则指定唯一权威文件，其余位置只留
一行链接；docs/06 是好例子（阈值只在它那里）。

## §7 token/上下文纪律：设计好，执行数据缺失

docs/13 §13.6 承诺"每个 loop 记录 attempts、queue/active time、Checker cycles、
token 起止"，loop_states 里只有零散字段。没有这些数据，"流程改进是否有效"在
Stage 2 checkpoint 时将无法复盘（正如 13.3 只能用 git 时间窗估计 Stage 0 成本）。
建议：loop state schema 增加固定 `cost:` 块（pbs_jobs、queue_seconds、
active_seconds、checker_rounds、failure_classes[]、token_start/end），PERSIST 时
必填——都是现成可采的数字，边际成本接近零。

## §8 RED/fixture 的可维护性规则

台账 #2、#3、#4、#12 的共同形状：**过度精确的断言**（精确 schema 全集、源码行
字面量、canonical 字节相等、跨文件哈希）在正确的代码演化下产生假阳性，然后在
最贵的环境里爆炸。建议在 docs/03 §3.4 增补：

1. 断言语义而非表示（禁止对生产源码做行字面量断言——#2 的 formatter 断行事故）；
2. 身份性字段新增时，走一个"schema 演进清单"：`rg` 所有 validator/fixture/
   analyzer 引用点并同 commit 更新（#2、#3 的直接根因）；
3. 并发回归必须有界 + 必然终止（stop event 在 finally 设置、观察集合有界）——
   已发生两次（#3、#8 的 harness 侧），应成为 HARDEN 检查单固定项；
4. 预期非零退出的命令一律放进 `if` 条件语境（#6 的 bash ERR trap 教训），加一条
   PBS 脚本静态 lint。

## §9 保留项（不要在减负时误伤）

- 失败台账制度本身：14 条记录的因果链质量极高，是本次审查能定量归因的唯一原因；
  减负对象是每条失败的**重授权仪式**，不是记录本身。
- Checker Phase A（提交前静态审查）：#1 的四个实现缺陷有三个是 post-run 代码
  review 发现的——如果 Phase A 更早看性能路径（而不只看契约完备性），第一次
  五节点失败可能避免。方向是把 Phase A 的检查单加一条"频率/字节预算与实现路径
  的量级核对"，而不是取消 Phase A。
- 冻结阈值 + 预注册：防止事后调参的价值已经兑现（#11 的阈值修正走了 ADR 而非
  静默放宽）；需要的只是 §5 说的"阈值必须有微基准依据"。
- index-first / raw-on-demand 上下文纪律：本次审查即受益（indexes + 台账几乎
  覆盖全部事实，raw 只需抽查）。
