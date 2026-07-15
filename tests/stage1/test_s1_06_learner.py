from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from fsbdd.learner import (  # noqa: E402
    ConstantStepScheduler,
    LearnerError,
    LearnerProgress,
    LearnerRng,
    LearnerRuntime,
    PackedTokenShard,
)
from fsbdd.learner_assets import (  # noqa: E402
    LearnerAssetError,
    load_learner_profile,
    materialize_packed_shards,
    validate_materialized_shards,
    verify_profile_assets,
)
from fsbdd.learner_smoke import (  # noqa: E402
    LearnerSmokeError,
    _fragment_parameter_groups,
    main as learner_smoke_main,
    profile_set_digest,
    summarize_runs,
)
from fsbdd.logging import StructuredLogger  # noqa: E402
from fsbdd.manifest import validate_manifest  # noqa: E402
from fsbdd.model_registry import build_logical_layer_registry  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
GPT2_PROFILE = ROOT / "configs/stage1/s1_06_gpt2_wikitext_smoke.json"
PYTHIA_PROFILE = ROOT / "configs/stage1/s1_06_pythia160m_fineweb_smoke.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_profile(path: Path, *, revision: str = "a" * 40) -> dict:
    model_bytes = b"model"
    tokenizer_bytes = b"tokenizer"
    source_bytes = b"parquet-placeholder"
    profile = {
        "schema_version": 1,
        "profile_id": "s1-06-test-packed-v1",
        "model": {
            "repository": "org/model",
            "revision": revision,
            "expected_parameter_count": 10,
            "attention_implementation": "eager",
            "use_cache": False,
            "files": [
                {
                    "path": "model.bin",
                    "bytes": len(model_bytes),
                    "sha256": hashlib.sha256(model_bytes).hexdigest(),
                }
            ],
        },
        "tokenizer": {
            "repository": "org/model",
            "revision": revision,
            "eos_token_id": 0,
            "add_special_tokens": False,
            "files": [
                {
                    "path": "tokenizer.json",
                    "bytes": len(tokenizer_bytes),
                    "sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
                }
            ],
        },
        "dataset": {
            "repository": "org/data",
            "revision": revision,
            "config": "sample",
            "text_field": "text",
            "skip_rule": "text_equals_empty_string",
            "normalization": "none",
            "append_eos": True,
            "sequence_length": 8,
            "shard_count": 2,
            "shard_rule": "packed_block_index_mod_shard_count",
            "epoch_seed_base": 17,
            "source_files": [
                {
                    "split": "train",
                    "path": "sample.parquet",
                    "bytes": len(source_bytes),
                    "sha256": hashlib.sha256(source_bytes).hexdigest(),
                }
            ],
            "smoke": {
                "minimum_blocks_per_shard": 2,
                "source_policy": "canonical_prefix_until_minimum_then_finish_current_row",
            },
            "long_run": {
                "source_path_pattern": "*.parquet",
                "source_order": "bytewise_lexicographic",
                "minimum_processed_input_tokens": 1_000_000_000,
                "minimum_packed_blocks": 125_000_000,
            },
        },
        "training": {
            "learner_id": "learner-0",
            "learner_index": 0,
            "fragment_count": 2,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "optimizer_steps": 2,
            "precision": "fp32",
            "optimizer": {
                "class": "AdamW",
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "grad_clip_norm": 1.0,
            },
            "scheduler": {"class": "constant", "basis": "local_optimizer_steps"},
            "seed": 17,
            "initial_fragment_versions": [0, 3],
            "comparison": {
                "matched_tokens": False,
                "matched_compute": False,
                "matched_communication": False,
                "claim": "test_only",
            },
        },
    }
    path.write_text(json.dumps(profile), encoding="utf-8")
    return profile


def _fake_cache(tmp_path: Path, profile: dict) -> Path:
    hub = tmp_path / "hub"
    model = hub / "models--org--model" / "snapshots" / profile["model"]["revision"]
    dataset = hub / "datasets--org--data" / "snapshots" / profile["dataset"]["revision"]
    model.mkdir(parents=True)
    dataset.mkdir(parents=True)
    (model / "model.bin").write_bytes(b"model")
    (model / "tokenizer.json").write_bytes(b"tokenizer")
    (dataset / "sample.parquet").write_bytes(b"parquet-placeholder")
    return hub


def _tiny_model(seed: int = 1606) -> torch.nn.Module:
    torch.manual_seed(seed)
    return transformers.GPTNeoXForCausalLM(
        transformers.GPTNeoXConfig(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=16,
            use_cache=False,
        )
    )


def _groups(model: torch.nn.Module, count: int = 2) -> tuple[tuple[torch.nn.Parameter, ...], ...]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    split = len(parameters) // count
    return (parameters[:split], parameters[split:])


def _batches(count: int, *, padding: bool = True) -> list[dict[str, torch.Tensor]]:
    result = []
    for index in range(count):
        input_ids = (torch.arange(32).reshape(4, 8) + index) % 64
        mask = torch.ones_like(input_ids)
        if padding and index % 2:
            mask[:, -2:] = 0
        labels = input_ids.clone()
        labels[mask == 0] = -100
        result.append({"input_ids": input_ids, "labels": labels, "attention_mask": mask})
    return result


class _Clock:
    def __init__(self) -> None:
        self.value = -1_000_000_000

    def __call__(self) -> int:
        self.value += 1_000_000_000
        return self.value


def _runtime(
    model: torch.nn.Module,
    *,
    progress: LearnerProgress | None = None,
    scheduler: ConstantStepScheduler | None = None,
    logger: StructuredLogger | None = None,
    gradient_accumulation: int = 2,
    lr: float = 0.01,
    optimizer_type: str = "adamw",
    rng: LearnerRng | None = None,
    rng_seed: int = 1606,
    update_norm_interval: int = 1,
    retain_events: bool = True,
    clock_ns=None,
) -> LearnerRuntime:
    progress = progress or LearnerProgress.initialize("learner-a", (2, 5))
    scheduler = scheduler or ConstantStepScheduler(progress.local_optimizer_steps)
    rng = rng or LearnerRng.initialize(progress.learner_id, rng_seed, torch.device("cpu"))
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=lr)
        if optimizer_type == "adamw"
        else torch.optim.SGD(model.parameters(), lr=lr)
    )
    return LearnerRuntime(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
        rng=rng,
        fragment_parameters=_groups(model),
        device=torch.device("cpu"),
        precision="fp32",
        gradient_accumulation_steps=gradient_accumulation,
        max_grad_norm=1.0,
        comparison={
            "matched_tokens": False,
            "matched_compute": False,
            "matched_communication": False,
            "claim": "test_only",
        },
        logger=logger,
        update_norm_interval=update_norm_interval,
        retain_events=retain_events,
        clock_ns=clock_ns or _Clock(),
    )


def test_tracked_profiles_freeze_distinct_real_workloads_and_no_torch_change() -> None:
    gpt2 = load_learner_profile(GPT2_PROFILE)
    pythia = load_learner_profile(PYTHIA_PROFILE)
    assert gpt2.model["revision"] == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    assert pythia.model["revision"] == "50f5173d932e8e61f858120bcb800b97af589f46"
    assert pythia.dataset["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert pythia.dataset["long_run"]["minimum_processed_input_tokens"] >= 1_000_000_000
    assert gpt2.digest != pythia.digest
    assert profile_set_digest((gpt2, pythia)) == profile_set_digest((pythia, gpt2))
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "torch"\nversion = "2.13.0+cu132"' in lock


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["model"].update(revision="main"), "immutable 40-character"),
        (lambda row: row["dataset"].update(normalization="strip"), "skip only empty"),
        (lambda row: row["training"].update(gradient_accumulation_steps=0), "must be an integer"),
        (lambda row: row["training"]["scheduler"].update(basis="tokens"), "only constant"),
    ],
)
def test_profile_schema_fails_closed(tmp_path: Path, mutation, message: str) -> None:
    path = tmp_path / "profile.json"
    row = _write_profile(path)
    mutation(row)
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(LearnerAssetError, match=message):
        load_learner_profile(path)


def test_asset_identity_and_visibility_last_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_path = tmp_path / "profile.json"
    raw = _write_profile(profile_path)
    profile = load_learner_profile(profile_path)
    hub = _fake_cache(tmp_path, raw)
    inventory = verify_profile_assets(profile, hub)
    assert inventory["profile_digest"] == profile.digest

    class Tokenizer:
        eos_token_id = 0

        @staticmethod
        def encode(text: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            return [ord(character) % 31 + 1 for character in text]

    class Column:
        def __init__(self, values: list[str]) -> None:
            self.values = values

        def to_pylist(self) -> list[str]:
            return self.values

    class Batch:
        def __init__(self, values: list[str]) -> None:
            self.values = values

        def column(self, index: int) -> Column:
            assert index == 0
            return Column(self.values)

    class ParquetFile:
        def __init__(self, path: str) -> None:
            assert path.endswith("sample.parquet")

        @staticmethod
        def iter_batches(**_kwargs):
            yield Batch(["", "abcdefg", "hijklmn", "opqrstu", "vwxyzab", "moretext"])

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: Tokenizer())
    import pyarrow.parquet

    monkeypatch.setattr(pyarrow.parquet, "ParquetFile", ParquetFile)
    output = tmp_path / "packed"
    manifest = materialize_packed_shards(profile, hub, output)
    assert manifest["mode"] == "smoke_prefix"
    assert manifest["shards"][0]["blocks"] >= 2
    assert manifest["shards"][1]["blocks"] >= 2
    assert (output / "complete.json").is_file()
    assert validate_materialized_shards(profile, output) == manifest
    with pytest.raises(LearnerAssetError, match="overwrite"):
        materialize_packed_shards(profile, hub, output)
    with pytest.raises(LearnerAssetError, match="long-run token budget"):
        materialize_packed_shards(profile, hub, tmp_path / "long-run", mode="long_run")

    first = PackedTokenShard(profile, output, learner_index=0)
    second = PackedTokenShard(profile, output, learner_index=1)
    assert first.path != second.path
    second_initial_state = second.state_dict()
    batch = first.next_batch(2)
    state = first.state_dict()
    assert state["learner_index"] != second_initial_state["learner_index"]
    assert second.state_dict() == second_initial_state
    resumed = PackedTokenShard(profile, output, learner_index=0, epoch=state["epoch"], position=state["position"])
    assert torch.equal(first.next_batch(1)["input_ids"], resumed.next_batch(1)["input_ids"])
    assert second.state_dict() == second_initial_state
    assert batch["input_ids"].shape == (2, 8)

    shard = output / manifest["shards"][0]["path"]
    shard.write_bytes(shard.read_bytes()[:-4])
    with pytest.raises(LearnerAssetError, match="truncated"):
        validate_materialized_shards(profile, output)


def test_optimizer_boundary_accumulation_tokens_loss_and_fragment_counters(tmp_path: Path) -> None:
    model = _tiny_model()
    log = tmp_path / "learner.jsonl"
    runtime = _runtime(
        model,
        logger=StructuredLogger(log, role="learner", run_id="test-run"),
    )
    result = runtime.run(_batches(4), optimizer_steps=2)
    assert result.optimizer_steps_completed == 2
    assert result.parameters_changed and result.parameter_update_norm > 0
    assert runtime.scheduler.step_count == 2
    assert len(result.events) == 2
    assert [event.local_optimizer_step for event in result.events] == [1, 2]
    assert all(event.microbatches == 2 for event in result.events)
    expected_input = sum(int(batch["attention_mask"].sum()) for batch in _batches(4))
    expected_targets = sum(int(batch["attention_mask"][:, 1:].sum()) for batch in _batches(4))
    assert result.processed_input_tokens == expected_input
    assert result.loss_bearing_target_tokens == expected_targets
    assert result.progress["local_optimizer_steps"] == 2
    assert [row["local_steps_since_adoption"] for row in result.progress["fragments"]] == [2, 2]
    assert [row["processed_input_tokens_since_adoption"] for row in result.progress["fragments"]] == [expected_input] * 2
    assert math.isclose(result.common_interval_input_tokens_per_second, expected_input / 4.0)
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["safe_boundary", "safe_boundary"]
    assert all(record["lr_schedule_basis"] == "local_optimizer_steps" for record in records)
    assert all(record["distributed_initialized"] is False for record in records)
    assert all(set(record["inactive_metrics"].values()) == {
        "not_exercised_until_S1-07",
        "not_exercised_until_S1-08",
    } for record in records)


def test_long_run_sampling_and_dynamic_stop_keep_complete_loss_log(tmp_path: Path) -> None:
    progress = LearnerProgress.initialize("learner-a", (2, 5))
    log = tmp_path / "sampled.jsonl"
    runtime = _runtime(
        _tiny_model(),
        progress=progress,
        gradient_accumulation=1,
        logger=StructuredLogger(log, role="learner", run_id="sampled"),
        update_norm_interval=2,
        retain_events=False,
    )
    result = runtime.run(
        _batches(5),
        optimizer_steps=5,
        stop_requested=lambda: progress.local_optimizer_steps >= 3,
    )
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert result.optimizer_steps_completed == 3
    assert result.events == ()
    assert [record["local_optimizer_step"] for record in records] == [1, 2, 3]
    assert [bool(record["fragment_update_norms"]) for record in records] == [True, True, False]


def test_padding_aware_accumulation_matches_one_combined_token_mean_update() -> None:
    model = _tiny_model()
    reference = copy.deepcopy(model)
    reference_batches = _batches(2)
    batches = copy.deepcopy(reference_batches)
    # A caller may provide an attention mask without also converting padding
    # labels to -100. The runtime owns the accounting semantics and must make
    # the model loss use the same loss-bearing target set.
    batches[1]["labels"][batches[1]["attention_mask"] == 0] = batches[1]["input_ids"][
        batches[1]["attention_mask"] == 0
    ]
    runtime = _runtime(model, gradient_accumulation=2, optimizer_type="sgd")
    runtime.run(batches, optimizer_steps=1)

    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    combined = {
        key: torch.cat([batch[key] for batch in reference_batches], dim=0)
        for key in reference_batches[0]
    }
    reference(**combined).loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_delayed_independent_batch_source_completes_without_peer_or_storage() -> None:
    supplied = iter(_batches(1))
    delay_seconds = 0.02

    class DelayedSource:
        requests = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.requests += 1
            time.sleep(delay_seconds)
            return next(supplied)

    source = DelayedSource()
    result = _runtime(
        _tiny_model(),
        gradient_accumulation=1,
        clock_ns=time.monotonic_ns,
    ).run(source, optimizer_steps=1)
    assert source.requests == 1
    assert result.optimizer_steps_completed == 1
    assert result.events[0].step_latency_seconds >= delay_seconds


def test_resume_preserves_progress_versions_and_scheduler_basis() -> None:
    original = LearnerProgress.initialize("learner-r", (3, 9))
    original.local_optimizer_steps = 5
    original.processed_input_tokens = 100
    original.loss_bearing_target_tokens = 90
    original.fragments[0].local_steps_since_adoption = 2
    original.fragments[1].local_steps_since_adoption = 4
    restored = LearnerProgress.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    scheduler = ConstantStepScheduler(5)
    model = _tiny_model()
    result = _runtime(model, progress=restored, scheduler=scheduler, gradient_accumulation=1).run(
        _batches(1), optimizer_steps=1
    )
    assert result.progress["local_optimizer_steps"] == 6
    assert result.events[0].local_optimizer_step == 6
    assert result.events[0].fragment_global_versions == (3, 9)
    assert scheduler.step_count == 6


class _RandomLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.bias = torch.nn.Parameter(torch.tensor(0.5))
        self.draws: list[float] = []

    def forward(self, input_ids, **_kwargs):
        draw = torch.rand((), device=self.weight.device)
        self.draws.append(float(draw.detach()))
        prediction = input_ids.float().mean() * self.weight + self.bias
        return SimpleNamespace(loss=(prediction * draw - 1.0).square())


def test_two_learners_have_disjoint_model_optimizer_and_owned_rng_streams() -> None:
    model_a = _RandomLossModel()
    model_b = copy.deepcopy(model_a)
    progress_a = LearnerProgress.initialize("learner-a", (2, 5))
    progress_b = LearnerProgress.initialize("learner-b", (2, 5))
    rng_a = LearnerRng.initialize(progress_a.learner_id, 101, torch.device("cpu"))
    rng_b = LearnerRng.initialize(progress_b.learner_id, 202, torch.device("cpu"))
    runtime_a = _runtime(
        model_a,
        progress=progress_a,
        rng=rng_a,
        gradient_accumulation=1,
    )
    runtime_b = _runtime(
        model_b,
        progress=progress_b,
        rng=rng_b,
        gradient_accumulation=1,
    )
    assert runtime_a.model is not runtime_b.model
    assert runtime_a.optimizer is not runtime_b.optimizer
    assert runtime_a.progress is not runtime_b.progress
    assert not ({id(parameter) for parameter in model_a.parameters()} & {id(parameter) for parameter in model_b.parameters()})
    assert rng_a is not rng_b
    assert rng_a.state_sha256() != rng_b.state_sha256()

    process_rng_before = torch.get_rng_state().clone()
    result_a = runtime_a.run(_batches(1), optimizer_steps=1)
    assert torch.equal(torch.get_rng_state(), process_rng_before)
    result_b = runtime_b.run(_batches(1), optimizer_steps=1)
    assert torch.equal(torch.get_rng_state(), process_rng_before)
    assert model_a.draws != model_b.draws
    assert result_a.rng["seed"] == 101 and result_b.rng["seed"] == 202
    assert result_a.rng["scope"] == result_b.rng["scope"] == "runtime_owned_forked_torch_rng"
    assert result_a.rng["process_global_state_restored"] is True
    assert result_b.rng["process_global_state_restored"] is True

    restored_rng = LearnerRng.from_state_dict(rng_a.state_dict())
    assert restored_rng.state_dict() == rng_a.state_dict()
    with pytest.raises(LearnerError, match="already owned"):
        _runtime(
            _RandomLossModel(),
            progress=LearnerProgress.initialize("learner-a", (2, 5)),
            rng=rng_a,
            gradient_accumulation=1,
        )


def test_gpt2_registry_covers_position_embedding_static_buffers_and_tied_head() -> None:
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_layer=2, n_head=2, n_embd=16, n_positions=32, vocab_size=64)
    )
    registry = build_logical_layer_registry(model)
    assert registry.family == "gpt2"
    assert registry.coverage.owned == registry.coverage.unique_trainable
    assert registry.coverage.unclassified_buffers == 0
    assert len(registry.tied_identities) == 1
    names = {alias for record in registry.parameters for alias in record.aliases}
    assert "transformer.wpe.weight" in names
    groups = _fragment_parameter_groups(model, 2)
    assert {id(parameter) for group in groups for parameter in group} == {
        id(parameter) for parameter in model.parameters()
    }


class _MissingLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(logits=input_ids.float() * self.weight)


class _NaNLossModel(_MissingLossModel):
    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(loss=self.weight * torch.tensor(float("nan")))


class _FiniteLossInfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, value):
        return value.clone()

    @staticmethod
    def backward(_ctx, gradient):
        return torch.full_like(gradient, float("inf"))


class _OverflowGradientModel(_MissingLossModel):
    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(loss=_FiniteLossInfiniteGradient.apply(self.weight))


@pytest.mark.parametrize(
    ("model_factory", "batches", "message"),
    [
        (_MissingLossModel, _batches(1), "scalar loss"),
        (_NaNLossModel, _batches(1), "NaN or Inf"),
        (_OverflowGradientModel, _batches(1), "gradient norm is NaN Inf or overflowed"),
        (_tiny_model, [{}], "empty batch"),
        (_tiny_model, [], "data source exhausted"),
    ],
)
def test_failure_semantics_zero_grad_without_optimizer_boundary(model_factory, batches, message: str) -> None:
    model = model_factory()
    parameters = tuple(model.parameters())
    midpoint = max(1, len(parameters) // 2)
    groups = (parameters[:midpoint], parameters[midpoint:])
    if not groups[1]:
        groups = (parameters,)
        progress = LearnerProgress.initialize("bad", (0,))
    else:
        progress = LearnerProgress.initialize("bad", (0, 0))
    runtime = LearnerRuntime(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
        scheduler=ConstantStepScheduler(),
        progress=progress,
        rng=LearnerRng.initialize(progress.learner_id, 9, torch.device("cpu")),
        fragment_parameters=groups,
        device=torch.device("cpu"),
        precision="fp32",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        comparison={"claim": "test"},
        clock_ns=_Clock(),
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    with pytest.raises(LearnerError, match=message):
        runtime.run(batches, optimizer_steps=1)
    assert runtime.progress.local_optimizer_steps == 0
    assert all(torch.equal(left, right) for left, right in zip(before, model.parameters(), strict=True))


def test_zero_lr_is_rejected_as_no_real_parameter_update() -> None:
    model = _tiny_model()
    runtime = _runtime(model, gradient_accumulation=1, lr=0.0)
    with pytest.raises(LearnerError, match="changed no model parameter"):
        runtime.run(_batches(1), optimizer_steps=1)


def test_runtime_source_has_no_distributed_trainer_global_step_or_protocol_io() -> None:
    import fsbdd.learner as learner_source

    source = inspect.getsource(learner_source)
    assert "init_process_group" not in source
    assert "torchrun" not in source
    assert "Trainer" not in source
    assert "global_step" not in source
    assert "ProposalStore" not in source
    assert "GlobalStateStore" not in source


def _run_record(profile_id: str, *, parameter_count: int | None = None) -> dict:
    events = []
    for step in range(1, 11):
        events.append(
            {
                "local_optimizer_step": step,
                "token_weighted_loss": 4.0 - step * 0.01,
                "fragment_update_norms": [0.1, 0.2],
                "fragment_local_steps": [step, step],
                "fragment_processed_input_tokens": [step * 32, step * 32],
                "inactive_metrics": {"gpu_to_cpu_seconds": "not_exercised_until_S1-07"},
                "comparison": {"claim": "runtime_smoke_only"},
            }
        )
    row = {
        "status": "pass",
        "profile": {"profile_id": profile_id},
        "runtime": {
            "learner_id": "tiny-learner" if parameter_count is None else "learner-0",
            "optimizer_steps_completed": 10,
            "events": events,
            "parameters_changed": True,
            "parameter_update_norm": 1.0,
            "training_start_monotonic_ns": 0,
            "training_end_monotonic_ns": 10_000_000_000,
            "processed_input_tokens": 320,
            "loss_bearing_target_tokens": 280,
            "token_weighted_loss": 3.95,
            "common_interval_input_tokens_per_second": 32.0,
            "rng": {
                "learner_id": "tiny-learner" if parameter_count is None else "learner-0",
                "seed": 1606 if parameter_count is None else 20260714,
                "scope": "runtime_owned_forked_torch_rng",
                "owner_pid": 123,
                "device_type": "cpu" if parameter_count is None else "cuda",
                "device_index": None if parameter_count is None else 0,
                "activation_count_before": 0,
                "activation_count_after": 1,
                "state_sha256_before": "a" * 64,
                "state_sha256_after": "b" * 64,
                "process_global_state_restored": True,
            },
        },
        "forbidden_runtime": {
            "torch_distributed_initialized": False,
            "hf_trainer_used": False,
            "proposal_published": False,
            "global_fragment_adopted": False,
        },
    }
    if parameter_count is not None:
        row["model"] = {
            "parameter_count": parameter_count,
            "optimizer_state_dtypes": ["torch.float32"],
            "initial_parameter_sha256": "a" * 64,
            "final_parameter_sha256": "b" * 64,
        }
        row["offline_environment"] = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
        row["runtime"]["processed_input_tokens"] = 20_480
        row["runtime"]["loss_bearing_target_tokens"] = 20_440
        row["runtime"]["common_interval_input_tokens_per_second"] = 2_048.0
        row["runtime"]["events"][-1]["fragment_processed_input_tokens"] = [20_480, 20_480]
        for event in row["runtime"]["events"]:
            event["microbatches"] = 1
    else:
        for event in row["runtime"]["events"]:
            event["microbatches"] = 2
    return row


def test_summary_keeps_three_workloads_and_common_intervals_separate(tmp_path: Path) -> None:
    paths = []
    records = {
        "tiny": _run_record("tiny"),
        "gpt2": _run_record(
            "s1-06-gpt2-wikitext-packed512-v1", parameter_count=124_439_808
        ),
        "pythia": _run_record(
            "s1-06-pythia160m-finewebedu-packed512-v1", parameter_count=162_322_944
        ),
    }
    for name, record in records.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "summary.json"
    summary = summarize_runs(*paths, output)
    assert set(summary["profiles"]) == {"tiny", "gpt2_wikitext", "pythia160m_fineweb"}
    assert summary["aggregation"]["combined_rank_rate"] is None
    assert summary["profiles"]["tiny"]["common_interval_input_tokens_per_second"] == 32
    assert summary["profiles"]["gpt2_wikitext"]["common_interval_input_tokens_per_second"] == 2048
    assert summary["profiles"]["pythia160m_fineweb"]["common_interval_input_tokens_per_second"] == 2048
    broken = _run_record("tiny")
    broken["runtime"]["common_interval_input_tokens_per_second"] = 96
    paths[0].write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(LearnerSmokeError, match="common-interval"):
        summarize_runs(*paths, tmp_path / "broken-summary.json")


def test_manifest_cli_serializes_path_output_and_builds_valid_l1_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    modules = tmp_path / "modules.txt"
    modules.write_text("nv-hpcx/25.9\n", encoding="utf-8")
    output = tmp_path / "manifest.json"
    result = learner_smoke_main(
        [
            "manifest",
            "--repository",
            "https://example.invalid/fsbdd.git",
            "--branch",
            "codex/S1-06-learner-inner-training",
            "--commit",
            "a" * 40,
            "--run-id",
            "s1-06-test",
            "--config-sha256",
            "b" * 64,
            "--research-sha256",
            "c" * 64,
            "--spec-sha256",
            "d" * 64,
            "--skill-repository",
            "https://example.invalid/miyabi-development.git",
            "--skill-commit",
            "e" * 40,
            "--initial-hostname",
            "miyabi-g1",
            "--project-root",
            str(ROOT),
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--job-id",
            "123.opbs",
            "--queue",
            "debug-g",
            "--group",
            "xg24i002",
            "--node-type",
            "Miyabi GPU compute node",
            "--timestamp-utc",
            "2026-07-15T00:00:00Z",
            "--modules-file",
            str(modules),
            "--output",
            str(output),
        ]
    )
    assert result == 0
    status = json.loads(capsys.readouterr().out)
    assert status == {"status": "building", "output": str(output)}
    manifest = json.loads(output.read_text(encoding="utf-8"))
    validate_manifest(manifest, finalized=False)
    assert manifest["environment"] == {
        "modules": ["nv-hpcx/25.9"],
        "pbs_job_id": "123.opbs",
        "pbs_queue": "debug-g",
        "pbs_group": "xg24i002",
        "node_type": "Miyabi GPU compute node",
    }
