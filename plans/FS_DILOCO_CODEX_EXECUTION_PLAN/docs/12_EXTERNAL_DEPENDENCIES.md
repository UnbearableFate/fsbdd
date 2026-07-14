# 12. 外部依赖与版本固定

## Miyabi Codex Skill

- Repository: [UnbearableFate/miyabi-development](https://github.com/UnbearableFate/miyabi-development)
- Role: Codex 的 Miyabi host routing、login-node安全、PBS interactive/batch、Python环境、Git同步和ML launcher参考。
- Policy: 不 vendor 到本执行包；每个正式 run 记录实际 commit。
- Update: 只在 loop 边界评估并 `pull --ff-only`；更新后至少重跑环境/launcher smoke。
- Conflict: 当前站点规则与 skill 的更新版本优先于本包中的示例。

该 skill 的工作流把 login nodes 作为控制面，将 Torch/HF、训练、CUDA、MPI等运行时工作放到PBS compute nodes。本项目额外收窄：其 torchrun/Accelerate例子只能作为环境/launcher参考，不能建立跨learner数据面；FS协议仍是唯一算法通信介质。

## Python/ML Dependencies

正式项目必须在 `uv.lock` 固定：

- Python
- torch / CUDA compatibility
- transformers
- datasets
- tokenizers
- safetensors
- accelerate（若使用）
- testing/analysis packages

不在本计划中硬编码版本，因为必须以Miyabi当前module/CUDA支持和实际compute-node验证为准。项目pin一旦成为baseline的一部分，升级需要独立loop/ADR并失效化不兼容runtime baseline。

## Model/Data Assets

模型、tokenizer、dataset revision和预处理代码都以manifest/hash固定。门禁运行使用预置cache/offline模式，不依赖在线Hub可用性。
