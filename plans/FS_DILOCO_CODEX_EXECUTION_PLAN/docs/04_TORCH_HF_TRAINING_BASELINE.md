# 04. PyTorch + Hugging Face 训练基线

## 4.1 设计目标

训练基线必须足够普通，保证系统研究结果不依赖特殊训练框架，同时显式暴露 FS-based Decoupled DiLoCo 所需的所有边界：

- 完整 local optimizer step；
- per-fragment local step/token counters；
- fragment snapshot；
- global fragment adoption；
- mixed-version model；
- independent data shard；
- loss/throughput telemetry；
- frozen evaluation snapshot。

## 4.2 推荐栈

- `torch`
- `transformers`
- `tokenizers`
- `datasets`
- `safetensors`
- `accelerate`（可选，单进程 only）
- `pytest`, `hypothesis`
- `uv`

实际版本在 Miyabi 兼容性验证后 pin 到 lockfile。不得只写开放范围而不保存锁文件。

## 4.3 模型

固定 9 节点回归 profile：

- `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`；
- GPT-2 small dense decoder-only causal LM，预期 124,439,808 参数；
- 从同一离线 cached pretrained checkpoint 初始化；
- input embedding、完整 Transformer blocks、lm_head 构成逻辑层；
- final norm 等零散参数按规格确定性归属；
- tied embedding/lm_head 只同步一次。

该 profile 的完整冻结字段见 `configs/miyabi_8l1s_50x10.yaml` 和
`docs/06_8L1S_50X10_GATE.md`。source spec 的约 160M/0.5–1B research profiles 保持为独立
正式验收 workload；它们不能复用 GPT-2 门禁的 runtime baseline。

## 4.4 自定义训练循环

伪流程：

```text
load frozen config / data manifest / fragment map
bootstrap or read current global fragments
create full local HF model
create independent AdamW inner optimizer
for each local optimizer step:
    batch = next(independent data shard)
    loss = causal_lm_forward(batch)
    backward / grad clipping / optimizer.step / scheduler.step
    mark safe boundary
    increment per-fragment steps/tokens
    enqueue due fragment snapshots by offsets
    poll latest global records
    apply fetched fragment atomically at safe boundary
    emit structured metrics
    if run-complete record observed: stop at safe boundary
```

禁止让 HF Trainer 自动 checkpoint、global_step、distributed process group 或 scheduler 取代本协议语义。

## 4.5 数据

9 节点回归固定使用 `Salesforce/wikitext` revision
`b08601e04326c79dfdd32d625aee71d232d685c3`、config
`wikitext-2-raw-v1`：非空行原样 tokenize 后追加 EOS，拼接为不重叠的 512-token blocks，
最后 remainder 丢弃，train blocks 按 `block_index mod 8` 分给 learners。validation 只做
initial/final frozen snapshot NLL，test 禁止用于门禁调参与阈值选择。

- 数据必须预先 tokenized 为 8 个不可变 train shards 与一个 validation artifact，写 dataset manifest 和 hashes；
- learner `i` 使用独立 deterministic shard/stream；
- sequence length、packing、EOS/padding、tokenizer revision 固定；
- proposal/throughput 的 processed-input tokens 与 loss aggregate 的 loss-bearing target tokens 分开记录；后者排除 padding、`-100` label 和每个 packed sequence 在 causal shift 后没有 target 的首 token；
- 多节点门禁前预置 cache，门禁运行设置 offline 模式，避免网络下载与抖动；
- 可从共享 cache 复制到 node-local scratch 以隔离训练数据 I/O，但 protocol state 必须保留在目标共享 FS，并记录复制时间是否计入 active runtime。

FineWeb-Edu/C4 仍用于原研究计划中的长跑与质量主张，不得与 WikiText-2 gate 数据混成同一
compatibility key。

## 4.6 优化器与调度

- inner optimizer：AdamW 类，每 learner 独立；
- inner LR schedule：按 local optimizer steps 或 tokens，必须明确；
- adoption 默认保留 inner moments，只覆盖目标 fragment 参数并重置该 fragment counters；
- outer optimizer：per-fragment momentum/Nesterov SGD 主线；
- control：无 momentum SGD、outer lr=1 direct averaging；
- gradient accumulation 时只有 `optimizer.step()` 完成才增加 local step；
- fp32 merge accumulation；
- gradient clipping、weight decay、warmup、precision 全部进入 frozen config。

## 4.7 资产预置与离线运行

9 节点前必须：

- 确认 tokenizer/model/data 均可从指定 shared cache 读取；
- 拒绝 mutable `main`、未解析 revision 或与配置不符的 cache snapshot；
- 保存 model config/tokenizer/dataset manifest digests；
- 做 1-node 真实 10-step smoke；
- 推荐设置：
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
  - `HF_DATASETS_OFFLINE=1`
- 不在每个 learner 同时下载或预处理；
- 不在 login 节点导入或 materialize 数据。

## 4.8 Loss 口径

每个 learner 记录：

- raw training loss；
- processed-input tokens 与 loss-bearing target tokens；
- local optimizer step；
- fragment version vector digest；
- throughput；
- grad norm（可选但建议）；
- NaN/Inf/overflow 信息。

聚合分析使用 token-weighted loss。门禁只证明训练路径有效与趋势正常，不构成 matched-token 质量结论；Stage 4 的 H3 仍按独立 eval 与多种子执行。

## 4.9 单 learner 与多 learner 的隔离

- learner 之间不共享 optimizer、RNG、dataloader cursor；
- 不建立 PyTorch distributed process group；
- 一台节点只运行一个 learner、只使用一个 GPU；
- learner ID 来自 frozen role map，不从 MPI rank隐式决定持久身份；
- coallocated launcher 的 rank 只用于启动时选择 role；run record 中持久身份必须显式。
