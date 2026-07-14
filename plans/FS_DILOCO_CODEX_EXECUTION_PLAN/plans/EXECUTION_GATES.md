# 用户补充执行门禁

这些门禁不替代 source acceptance，而是 Codex 每-loop 关闭条件。

| ID | 适用 | 条件 |
|---|---|---|
| X-CODEX-01 | 全部 | 按AGENTS/loop协议执行并持久化 |
| X-MIYABI-01 | 所有Miyabi任务 | 使用`miyabi-development`，记录commit/host routing |
| X-STACK-01 | Stage1–4 | PyTorch + Hugging Face真实训练路径 |
| X-PREFLIGHT-01 | 全部 | RED前完整 requirement/反例/观察点/聚合/evidence matrix |
| X-PACKAGE-01 | 所有正式runtime run | 提交前与finalization后通用package validator通过 |
| X-MEASURE-01 | 性能/并发 | 共同观察区间与聚合公式预先冻结，不合成非同步local rates |
| X-COST-01 | L2–L4 | 低层阶梯和Checker Phase A通过；失败后先降级复现 |
| X-9N-TOPOLOGY | S1-13及后续每loop | 8 learners +1 syncer，9 distinct compute nodes |
| X-50X10 | S1-13及后续每loop | H=50 local optimizer steps，global_cycle≥10 |
| X-LOSS | S1-13及后续每loop | finite且冻结的稳健趋势门禁通过 |
| X-RUNTIME | S1-13及后续每loop | active runtime在预注册/兼容baseline范围 |
| X-EVIDENCE | 全部 | raw evidence+hash+Checker可复算 |

`plans/PROGRESS.yaml` 中 `nine_node_gate_required: true` 的每个loop都必须有新的run ID。

Checker 独立性默认通过每 loop 一个无 Maker 聊天继承的 Checker context 实现。L2–L4
由该 context 先做 Phase A precheck、后做 Phase B final；不得用多个重复 final Checker
代替清晰的 blocked→fix→follow-up 链。
