# 用户补充执行门禁

这些门禁不替代 source acceptance，而是 Codex 每-loop 关闭条件。

| ID | 适用 | 条件 |
|---|---|---|
| X-CODEX-01 | 全部 | 按AGENTS/loop协议执行并持久化 |
| X-MIYABI-01 | 所有Miyabi任务 | 使用`miyabi-development`，记录commit/host routing |
| X-STACK-01 | Stage1–4 | PyTorch + Hugging Face真实训练路径 |
| X-9N-TOPOLOGY | S1-13及后续每loop | 8 learners +1 syncer，9 distinct compute nodes |
| X-50X10 | S1-13及后续每loop | H=50 local optimizer steps，global_cycle≥10 |
| X-LOSS | S1-13及后续每loop | finite且冻结的稳健趋势门禁通过 |
| X-RUNTIME | S1-13及后续每loop | active runtime在预注册/兼容baseline范围 |
| X-EVIDENCE | 全部 | raw evidence+hash+Checker可复算 |

`plans/PROGRESS.yaml` 中 `nine_node_gate_required: true` 的每个loop都必须有新的run ID。
