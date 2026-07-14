# S1-01 model adapter contract

The registry selects adapters by the exact Hugging Face `config.model_type`
and exact module structure. It does not infer boundaries from fuzzy parameter
name matches.

Supported built-in structures:

| Family | Embedding | Ordered blocks | Misc | Head |
|---|---|---|---|---|
| `gpt_neox` | `gpt_neox.embed_in` | `gpt_neox.layers.*` | `gpt_neox.final_layer_norm` | `embed_out` |
| `llama` | `model.embed_tokens` | `model.layers.*` | `model.norm` | `lm_head` |

An unknown family must provide `ExplicitMapping`, which names the embedding,
every complete block in forward order, the head, and each misc module with its
two adjacent logical-layer indices. Misc parameters go to the smaller adjacent
side in synchronization bytes; ties go left. Shared embedding/head parameters
have one synchronization owner at the embedding and retain all aliases in the
report. Any other cross-layer sharing or unowned trainable parameter fails
closed.

The real-structure command is compute-node only and creates no weights or data:

```bash
python -m fsbdd.hf_registry_summary \
  --model EleutherAI/pythia-160m \
  --revision main \
  --output <evidence-root>/analysis/pythia-160m-registry.json
```
