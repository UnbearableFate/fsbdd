# common/model 函数级代码参考

本章逐个说明 `common` 与 `model` 下的类方法和模块函数。纯数据类无显式方法时列出其字段语义；多个完全同构的 `to_dict` 会在同一行逐名列出，但没有省略符号。

## 1. `common/identity.py`

源码：[`identity.py`](../../src/fsbdd/diloco/common/identity.py)

| 符号 | 行 | 代码行为 |
|---|---:|---|
| `IdentityError` | 12 | canonical identity 输入不受支持时的 `ValueError`。 |
| `_canonical(value)` | 16–34 | dataclass instance 先 `asdict`；`Path→str`；mapping 要求全 string key 并按 key 排序递归；非 string/bytes sequence 转 list；拒绝非有限 float 和其他对象。特别避免把 dataclass class 当 instance。 |
| `canonical_bytes(value)` | 37–44 | 对 `_canonical` 结果输出 UTF-8 JSON；排序 key、紧凑分隔、保留 Unicode、禁止 NaN。所有语义 identity 共用这一字节规范。 |
| `canonical_digest(value)` | 47–48 | `sha256(canonical_bytes(value)).hexdigest()`。 |
| `file_digest(path)` | 51–56 | 以 1 MiB chunk 流式计算文件 SHA-256，不把整文件读入内存。 |

## 2. `common/logging.py`

源码：[`logging.py`](../../src/fsbdd/diloco/common/logging.py)

| 符号 | 行 | 代码行为 |
|---|---:|---|
| `StructuredLogger.__init__` | 13–40 | 校验 `Path`、非空 role/run、正整数 `fsync_every`；创建父目录，以 append text 模式打开 JSONL，初始化 mutex 和 emit count。 |
| `StructuredLogger.emit` | 42–85 | 拒绝调用者覆盖 schema/time/pid/run/role/event 保留字段；构造 UTC 微秒时间、monotonic ns 和 PID；在 lock 内写一行 canonical-like JSON；每 N 条 flush+`os.fsync`；返回同一 record。`allow_nan=False` 使非有限 telemetry 失败。 |
| `StructuredLogger.sync` | 87–90 | lock 内强制 flush+fsync。 |
| `StructuredLogger.close` | 92–98 | 幂等检查 closed；最终 flush+fsync 后关闭。 |

## 3. `model/model_registry.py`

源码：[`model_registry.py`](../../src/fsbdd/diloco/model/model_registry.py)

### 3.1 数据类

| 类型/方法 | 数据与约束 |
|---|---|
| `MiscAssignment` | 声明一个 misc module path 及相邻 `left_layer/right_layer`。 |
| `ExplicitMapping` | 模型 family、embedding path、有序 block paths、head path、misc 和 reconstructable buffers；用于内建 family 之外的显式可检查映射。 |
| `ParameterRecord` | 唯一 storage identity、owner layer/name、全部 aliases、shape/dtype/numel/sync bytes。`to_dict` 用 `dataclasses.asdict`。 |
| `BufferRecord` | buffer identity/aliases/shape/dtype/numel/classification。`to_dict` 同上。 |
| `LogicalLayer` | 前向顺序 index/name/kind/module path，所拥有 parameter identities、参数量和同步字节。`to_dict` 同上。 |
| `Coverage` | unique trainable、owned、duplicate/unowned、buffer 分类计数。`to_dict` 同上。 |
| `LogicalLayerRegistry._body` | 生成不含自身 digest 的规范语义 body，显式把 tuple 转 JSON list。 |
| `LogicalLayerRegistry.to_dict` | 返回 `_body` 加 `digest`。 |

### 3.2 函数

| 函数 | 行 | 代码行为 |
|---|---:|---|
| `_module(root,path)` | 119–131 | 逐段解析 dotted module path；数字段作为 index，其余用 attribute；不存在统一转为 `RegistryError`。 |
| `_children_paths(root,path)` | 134–142 | 找到 block container，读取非空 `named_children()`，返回完整 child paths。 |
| `_builtin_mapping(model)` | 145–220 | 拒绝 encoder-decoder；按 `config.model_type` 生成 GPT-NeoX/Llama/GPT-2 映射。GPT-2 把 WPE 和 final norm 作为 misc，并把 attention bias/masked_bias 声明为 reconstructable；未知类型要求显式 mapping。 |
| `_global_aliases(model)` | 223–239 | 用 `named_parameters(remove_duplicate=False)` 按 Python object id 聚合 trainable parameter 的全部名字，保留 tied aliases。 |
| `_global_buffers(model)` | 242–256 | 对 buffer 做同样 object-id/alias 聚合。 |
| `_under(name,module_path)` | 259–260 | 判断名字等于 module path 或在其 dotted subtree。 |
| `_parameter_bytes(parameter,sync_dtype_bytes)` | 263–267 | `numel × frozen sync dtype bytes`，异常转 `RegistryError`。 |
| `_parameter_shape(parameter)` | 270–274 | 把 concrete shape 转整数 tuple。 |
| `validate_parameter_ownership(owners)` | 277–281 | 按 identity 计数，任何不等于 1 的 owner 都报 duplicate ownership。 |
| `build_logical_layer_registry(...)` | 284–542 | 主构建器。严格验证 mapping/path/buffer policy；识别 logical owner 和 tied embedding ownership；先统计 base layer bytes，再按较小侧/平局左侧分配 misc；拒绝 unowned/shared-across-unsupported layers；为每个物理 parameter 生成 canonical identity 和唯一 record；构造 layers、coverage、misc report 和 registry digest。 |
| `write_registry_report(registry,destination)` | 545–553 | fail-if-exists；创建父目录，写 sorted+indented JSON+换行；返回 digest。 |

`build_logical_layer_registry` 的 owner 选择细节：若一个物理 parameter 同时命中 logical layer 0 和最后一层，只允许 embedding/head tied 情形并归 layer 0；命中其他多个 logical layers 直接失败。misc module 若与已有 owner 重叠也失败。

## 4. `model/fragment_map.py`

源码：[`fragment_map.py`](../../src/fsbdd/diloco/model/fragment_map.py)

### 4.1 数据类

`FragmentLayerSummary.to_dict`、`MiscOwnershipSummary.to_dict`、`Fragment.to_dict`、`FragmentObjective.to_dict`、`BalanceSummary.to_dict` 都返回 `dataclasses.asdict`。其中：

- `Fragment` 保存 `[start_layer,end_layer_exclusive)` 连续区间、展开的 layer/parameter identities、numel 和 bytes；
- `FragmentObjective` 保存最优 maximum bytes、按 `F*segment-total` 计算的精确 scaled deviation、exclusive cut indices；
- `BalanceSummary` 同时保存 float 统计和精确 numerator/denominator；
- `FragmentMap._body` 生成不含 digest 的完整 report；`FragmentMap.to_dict` 附加 digest。

### 4.2 函数

| 函数 | 行 | 代码行为 |
|---|---:|---|
| `_validate_fragment_count(F,L,production)` | 122–135 | 至少两逻辑层；production 要求 `1<F<L`；nonproduction oracle 只允许 `F=1`。 |
| `_minimum_maximum(weights,F)` | 138–162 | 动态规划求所有非空连续 F 分区中最小可能最大 segment bytes；零字节 segment 不可用。 |
| `partition_layer_bytes(weights,F)` | 165–225 | 校验非负整数、总 bytes>0、正字节 layer 数足够。先冻结 minimax cap，再用 DP 最小化 scaled total deviation；state 的 `(deviation,cuts)` tuple 自动以字典序最早 cuts 破平局。返回精确目标。 |
| `_validate_registry(registry)` | 228–280 | 重算 registry digest；检查层 index、parameter identity 唯一全覆盖、coverage、numel/bytes 和每层合计。 |
| `_misc_summaries(registry)` | 283–300 | 要求 misc report 精确字段集合并转为 typed summaries。 |
| `build_fragment_map(registry,F,production=True)` | 303–390 | 运行最优分区；按 cuts 展开完整 layers/parameters；计算 balance exact/float 统计；构造 digest；最后调用 `validate_fragment_map` 自检。 |
| `validate_fragment_map(map,registry)` | 393–522 | 全量重验 digest/registry binding、layers/tied/misc、连续非空区间、参数恰好一次覆盖、bytes/count、实际 objective 等于重新求出的最优 objective、balance 完全 canonical。 |
| `write_fragment_map_report(map,destination)` | 525–535 | fail-if-exists 写 JSON report，返回 map digest。 |

## 5. `model/learner_assets.py`

源码：[`learner_assets.py`](../../src/fsbdd/diloco/model/learner_assets.py)

### 5.1 profile 数据与基础 helper

| 符号 | 行 | 代码行为 |
|---|---:|---|
| `LearnerAssetError` | 19 | profile/asset/preprocessing contract 错误。 |
| `_freeze` / `_thaw` | 23–36 | dict→`MappingProxyType`、list→tuple 的递归只读化及反向可序列化转换。 |
| `_strict` | 39–48 | 要求 dict 且字段集合精确匹配，分别报告 missing/unknown。 |
| `_text` / `_integer` / `_number` | 51–69 | 校验非空 string、带下界且排除 bool 的 integer、有限带下界 numeric。 |
| `_sha256` / `_revision` | 72–87 | 分别要求 64 位小写 SHA-256 和 40 位不可变 commit。 |
| `_relative_path` | 90–95 | 要求非空、安全相对 POSIX path，不允许 absolute/`..`。 |
| `FrozenLearnerProfile.to_dict` | 104–105 | thaw raw mapping。 |
| `FrozenLearnerProfile.model/tokenizer/dataset/training` | 108–121 | 四个只读 section property。 |
| `_validate_files` | 164–178 | 检查非空 file list、精确字段、路径唯一、正 bytes、SHA；dataset row 还要求 split。 |

### 5.2 profile 与本地 snapshot

| 函数 | 行 | 代码行为 |
|---|---:|---|
| `load_learner_profile(path)` | 181–364 | 读取并严格校验全部 schema。区分早期 GPT-2 expected-splits profile 和 smoke/long-run profile；冻结 dataset text/packing policy、optimizer/scheduler、precision、shard/fragment/version vector 和 comparison flags；对 canonical JSON 算 digest 并递归只读化。 |
| `_repository_cache_name` | 367–369 | 生成 Hugging Face cache 目录名 `models--org--repo` 或 `datasets--...`。 |
| `snapshot_path` | 372–383 | 解析 `<cache>/<repo>/snapshots/<revision>`，必须已是目录。 |
| `_hash_file` | 386–391 | 8 MiB chunk SHA-256。 |
| `_verify_files` | 394–407 | 对 profile 的每个文件检查存在、size、SHA，附加 `resolved_path`。 |
| `verify_profile_assets` | 410–445 | 解析并校验 model/tokenizer/dataset 三个 frozen roots，返回完整 inventory。 |
| `_write_json` | 448–451 | sorted+indented JSON+换行写入。 |
| `_expected_split` | 454–458 | 从 profile 可选 `expected_splits` thaw 指定 split，否则 `None`。 |
| `_validated_token_rows` | 461–476 | tokenizer `input_ids` 必须恰有 count 行，每行只能是 int token。 |

### 5.3 packed shards

| 函数 | 行 | 代码行为 |
|---|---:|---|
| `materialize_packed_shards(...)` | 479–742 | fail-if-exists；验证 assets；选择 frozen source files（long-run 按 UTF-8 bytewise relative path 排序 glob）；local-only tokenizer；parquet 逐 batch 读取；只跳过空串，每非空行 tokenize 后 append EOS；维护连续 pending token stream，切固定 block，按 `packed_blocks % shard_count` 写 uint32-le；支持 smoke/long-run 最小量后在当前 row 边界停止；校验预注册 counts；fsync shards，写 manifest+complete marker，最后原子替换临时目录。异常关闭 streams 并删临时目录。 |
| `validate_materialized_shards(...)` | 745–942 | 验 marker 绑定 manifest SHA/profile；精确验 manifest schema、source accounting、token equation、shard 数/命名/bytes/count/checksum、modulo distribution、expected split 及 smoke/long-run 下限。指定 `required_shard_index` 时只 hash 该 learner 的大 shard，但仍检查全部 metadata/size。 |

## 6. `model/huggingface.py`

源码：[`huggingface.py`](../../src/fsbdd/diloco/model/huggingface.py)

| 函数 | 行 | 代码行为 |
|---|---:|---|
| `load_frozen_causal_lm` | 15–45 | 从已验证 `model_root` 用 `AutoModelForCausalLM.from_pretrained(local_files_only=True)` 加载；固定 attention implementation、fp32、`use_cache`，设 causal LM loss type，移到 device，核对 exact parameter count。 |
| `build_frozen_adamw` | 48–64 | 从 profile 精确构造 `torch.optim.AdamW`；betas 必须两项。gradient clip 不在 optimizer 内，而在 runtime。 |
| `model_parameter_digest` | 67–81 | 按 parameter name 排序；每项 hash 长度前缀的 name/shape/dtype JSON 和 fp32 CPU contiguous bytes，形成整个本地模型 digest。 |

## 7. `model/model_state.py`

源码：[`model_state.py`](../../src/fsbdd/diloco/model/model_state.py)

| 符号 | 行 | 代码行为 |
|---|---:|---|
| `FrozenModelState` | 22–25 | 保存 config identity、model identity 和 ordered `BootstrapFragment`。 |
| `freeze_hf_model_fragments` | 28–139 | 先验证 fragment map；lazy import safetensors；canonicalize 完整 HF config；用 registry owner names 精确取每个唯一 trainable parameter，核对 shape/dtype，CPU contiguous 后按 fragment safetensors 序列化；outer state 初始化为 canonical metadata；构造 descriptor 和 fragment hashes；拒绝重复/遗漏；model identity 绑定 model class/config/registry/map/每 fragment payload SHA。 |

这里生成的 descriptor dtype 是 `safetensors`、shape 是 parameter identity 数；Stage 1 在线训练路径另由 `build_fragment_descriptors` 生成 flat `float32` descriptor。两者服务不同 bootstrap 表示，不能混用。

## 8. `model/evaluation.py`

源码：[`evaluation.py`](../../src/fsbdd/diloco/model/evaluation.py)

### 8.1 校验与数据对象

| 符号 | 代码行为 |
|---|---|
| `_text` / `_hex` / `_ordinal` | 非空 string、64 位小写 SHA、非负整数校验。 |
| `EvaluationAccessAudit.__post_init__` | 把 root resolve 为绝对路径。 |
| `EvaluationAccessAudit.read_bytes` | 要求 read 在 root 内；拒绝路径中 current/latest/visibility；只允许一次 `manifest.json` purpose 或 `fragments/*.state` purpose；记录所有 operation 和分类计数。 |
| `EvaluationAccessAudit.to_dict` | 输出 instrumentation contract、operations 和五类 read counters。 |
| `FrozenEvaluationFragment.__post_init__` | 验 fragment identity/version、state/content/authority SHA、严格 `fragments/*.state` 相对路径、正 bytes 和 checksum。`to_dict` 返回 fields。 |
| `_manifest_semantic` | 构造不含 snapshot identity/time 的冻结语义：identities、learners、policy、三条 vectors、fragment records 和 no-latest load contract。 |
| `FrozenEvaluationManifest.__post_init__` | learner 唯一、fragment index 完整连续、identity/time 合法，并重算 semantic digest。`version_vector`、`content_identities` 是派生 property；`semantic` 和 `to_dict` 分别返回 digest 前/完整表示。 |
| `FrozenEvaluationPlan.__post_init__` | authority tuple 必须完整连续，全部 identities/learner frontier/policy 一致。`capture` 从 `AtomicGlobalCommitStore.load_snapshot` 一次冻结并记录 clock；两个 vector property 从 authorities 派生。 |
| `LoadedEvaluationSnapshot.__post_init__` | authority vectors 必须等于 manifest；audit 必须恰好 1 manifest + F frozen payload read，且 0 current/latest/unauthorized。 |

### 8.2 物化与加载函数

| 函数 | 行 | 代码行为 |
|---|---:|---|
| `_fragment_record` | 344–360 | 从 authority 和 encoded state 生成 manifest row；记录 state/parameter bytes 和 SHA。 |
| `materialize_evaluation_snapshot` | 363–427 | destination 必须新建；逐 authority `encode_global_state`，以 `xb` 写 exact bytes、flush+fsync；构造 semantic/snapshot identity；最后以 `xb` 写 canonical manifest+换行并 fsync。 |
| `_manifest_from_dict` | 430–493 | 要求 top-level 精确字段和值 contract；构造 typed identities/fragments/manifest；重复的 version/content/authority vectors 必须和 rows 一致。 |
| `load_evaluation_manifest` | 496–512 | 通过 access audit 读取 manifest，JSON parse 后要求重新 canonical encode+换行与原 bytes 完全相同，再 typed decode。 |
| `load_evaluation_snapshot` | 515–581 | 先冻结 manifest rows；逐精确 path audit/read/checksum；decode state 和 nested commit envelope，构造 authority；核对全部 identity/version/frontier/policy/bytes/checksum；累计 state/parameter bytes，返回带 audit 的 loaded snapshot。 |

## 9. 空包初始化文件

`src/fsbdd/diloco/__init__.py`、`common/__init__.py`、`model/__init__.py` 当前为空，不做 re-export、注册或 import side effect。调用方应从具体模块导入。

