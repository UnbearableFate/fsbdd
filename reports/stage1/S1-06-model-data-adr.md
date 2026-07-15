# S1-06 Model and Data Profile ADR

Status: preflight-frozen except exact materialized file hashes, which are a
blocking pre-RED field in the two tracked profile JSON files.

## Decision

S1-06 keeps two real workloads separate.

The regression workload is the exact `docs/06` contract:
`openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e`
and `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`,
`wikitext-2-raw-v1`. Tokenization, EOS insertion, packed-512 construction,
eight-way block-index sharding, optimizer, bf16 precision, and all expected
counts follow section 6.3 without local reinterpretation.

The research workload is pretrained
`EleutherAI/pythia-160m@50f5173d932e8e61f858120bcb800b97af589f46`
(the same immutable revision structurally checked in S1-01) with tokenizer
from that revision, and
`HuggingFaceFW/fineweb-edu@87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
The smoke source is `sample/10BT/000_00000.parquet`. The long-run stream is
all `sample/10BT/*.parquet` paths in bytewise lexicographic order at that
immutable revision and stops only after at least 1,000,000,000 packed input
tokens have been materialized.

For FineWeb-Edu, only `text == ""` is skipped. Other strings are not stripped
or normalized. Each retained text is encoded with `add_special_tokens=false`,
the Pythia EOS token is appended, streams are concatenated, and nonoverlapping
512-token blocks are emitted; the final remainder is dropped. Block `j` is
assigned to learner `j mod 4`. Learner `i` uses deterministic epoch
permutations from seed `20260714+i`. A smoke artifact is the canonical prefix
of this same stream sufficient for every shard to execute ten batch-4 steps;
it is not a different preprocessing policy.

Both workloads use independent data cursors and RNG state, AdamW
`lr=5e-5`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.01`, global
gradient clipping `1.0`, a constant schedule advanced by completed local
optimizer steps, bf16 model compute, and fp32 optimizer state. S1-06 does not
publish proposals or adopt global fragments.

## Identity and offline policy

Raw downloads occur once on a Miyabi compute node through immutable Hub
revisions. Exact SHA-256 values for config, tokenizer, weights, parquet files,
packed token streams, and per-shard artifacts are recorded in asset manifests.
A visibility-last completion marker binds the manifest digest. Real smokes set
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`, validate the marker and hashes, and use only local
files. There is no runtime download fallback.

No Torch dependency version is changed by this decision. S1-06 uses the
existing locked environment (`torch 2.13.0+cu132`). Any future Torch change
requires user permission.

## Measurement decision

Each workload reports ten complete optimizer-step boundaries. Input-token and
causal loss-bearing target-token counts remain separate. Profile throughput is
`total input tokens / (final safe boundary - training start)` for that one
observed interval. Per-step rates are diagnostic only and are never averaged or
summed into a claimed aggregate. Loss aggregation is target-token weighted.

Formal evidence must prove at least one parameter changed, every loss and
fragment update norm is finite, gradient accumulation does not advance local
step early, and `torch.distributed` remains uninitialized.

## Consequences

The GPT-2 workload remains comparable to the future 9-node regression gate.
The Pythia/FineWeb-Edu compatibility key is distinct and is the 160M Stage 1
long-run profile. A smoke result is a runtime-path check, not a matched-token
quality or throughput claim. S1-13 must materialize the full >=1B-token prefix
and may not substitute WikiText-2 for the research workload.
