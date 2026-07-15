from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .fragment_map import build_fragment_map
from .identity import canonical_digest
from .learner import (
    ConstantStepScheduler,
    LearnerProgress,
    LearnerRng,
    LearnerRuntime,
    PackedTokenShard,
)
from .learner_assets import (
    FrozenLearnerProfile,
    load_learner_profile,
    materialize_packed_shards,
    verify_profile_assets,
)
from .logging import StructuredLogger
from .manifest import build_manifest
from .model_registry import build_logical_layer_registry
from .learner_publish import build_fragment_parameter_groups


class LearnerSmokeError(RuntimeError):
    pass


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise LearnerSmokeError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_set_digest(profiles: Sequence[FrozenLearnerProfile]) -> str:
    if not profiles or len({profile.profile_id for profile in profiles}) != len(profiles):
        raise LearnerSmokeError("profile set must be non-empty with unique identities")
    return canonical_digest(
        {profile.profile_id: profile.digest for profile in sorted(profiles, key=lambda item: item.profile_id)}
    )


def _parameter_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        value = parameter.detach().float().cpu().contiguous()
        header = json.dumps(
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _fragment_parameter_groups(model: Any, fragment_count: int) -> tuple[tuple[Any, ...], ...]:
    registry = build_logical_layer_registry(model)
    fragment_map = build_fragment_map(registry, fragment_count)
    try:
        return build_fragment_parameter_groups(model, registry, fragment_map)
    except RuntimeError as error:
        raise LearnerSmokeError(str(error)) from error


def _registry_summary(model: Any, fragment_count: int) -> dict[str, Any]:
    registry = build_logical_layer_registry(model)
    fragment_map = build_fragment_map(registry, fragment_count)
    return {
        "registry_digest": registry.digest,
        "fragment_map_digest": fragment_map.digest,
        "logical_layers": len(registry.layers),
        "fragment_count": fragment_map.fragment_count,
        "parameter_count": sum(record.numel for record in registry.parameters),
        "unique_parameter_identities": len(registry.parameters),
        "buffer_count": len(registry.buffers),
    }


def _batch_iterator(shard: PackedTokenShard, batch_size: int) -> Iterator[dict[str, Any]]:
    while True:
        yield shard.next_batch(batch_size)


def _offline_environment() -> dict[str, str]:
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    result = {name: os.environ.get(name, "") for name in required}
    if any(value != "1" for value in result.values()):
        raise LearnerSmokeError(f"real smoke requires offline-only environment: {result}")
    return result


def _load_real_model(profile: FrozenLearnerProfile, inventory: Mapping[str, Any], device: Any) -> Any:
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    model = AutoModelForCausalLM.from_pretrained(
        inventory["model_root"],
        local_files_only=True,
        attn_implementation=str(profile.model["attention_implementation"]),
        dtype=torch.float32,
    )
    model.config.use_cache = bool(profile.model["use_cache"])
    model.loss_type = "ForCausalLM"
    model.to(device)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != int(profile.model["expected_parameter_count"]):
        raise LearnerSmokeError(
            f"model parameter count mismatch: expected {profile.model['expected_parameter_count']}, observed {count}"
        )
    return model


def _optimizer(profile: FrozenLearnerProfile, model: Any) -> Any:
    import torch

    spec = profile.training["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["lr"]),
        betas=tuple(float(value) for value in spec["betas"]),
        eps=float(spec["eps"]),
        weight_decay=float(spec["weight_decay"]),
    )


def _optimizer_state_dtypes(optimizer: Any) -> tuple[str, ...]:
    import torch

    dtypes = sorted(
        {
            str(value.dtype)
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        }
    )
    if dtypes != ["torch.float32"]:
        raise LearnerSmokeError(f"optimizer floating state must be fp32, observed {dtypes}")
    return tuple(dtypes)


def run_real_profile(
    profile_path: Path,
    hub_cache: Path,
    materialized_root: Path,
    output: Path,
    *,
    log_path: Path,
) -> dict[str, Any]:
    import torch

    offline = _offline_environment()
    profile = load_learner_profile(profile_path)
    inventory = verify_profile_assets(profile, hub_cache)
    if not torch.cuda.is_available():
        raise LearnerSmokeError("real learner smoke requires one CUDA GPU")
    device = torch.device("cuda", 0)
    torch.manual_seed(int(profile.training["seed"]))
    torch.cuda.manual_seed_all(int(profile.training["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = _load_real_model(profile, inventory, device)
    fragment_count = int(profile.training["fragment_count"])
    registry = _registry_summary(model, fragment_count)
    fragment_groups = _fragment_parameter_groups(model, fragment_count)
    initial_parameter_sha256 = _parameter_digest(model)
    optimizer = _optimizer(profile, model)
    scheduler = ConstantStepScheduler()
    progress = LearnerProgress.initialize(
        str(profile.training["learner_id"]),
        tuple(int(value) for value in profile.training["initial_fragment_versions"]),
    )
    rng = LearnerRng.initialize(
        progress.learner_id,
        int(profile.training["seed"]),
        device,
    )
    shard = PackedTokenShard(
        profile,
        materialized_root,
        learner_index=int(profile.training["learner_index"]),
    )
    initial_data_state = shard.state_dict()
    logger = StructuredLogger(log_path, role="learner", run_id=profile.profile_id)
    runtime = LearnerRuntime(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
        rng=rng,
        fragment_parameters=fragment_groups,
        device=device,
        precision=str(profile.training["precision"]),
        gradient_accumulation_steps=int(profile.training["gradient_accumulation_steps"]),
        max_grad_norm=float(profile.training["optimizer"]["grad_clip_norm"]),
        comparison=profile.training["comparison"],
        logger=logger,
    )
    run = runtime.run(
        _batch_iterator(shard, int(profile.training["batch_size"])),
        optimizer_steps=int(profile.training["optimizer_steps"]),
    )
    final_parameter_sha256 = _parameter_digest(model)
    if final_parameter_sha256 == initial_parameter_sha256:
        raise LearnerSmokeError("real learner run did not change its parameter digest")
    state_dtypes = _optimizer_state_dtypes(optimizer)
    result = {
        "schema_version": 1,
        "status": "pass",
        "kind": "real_hf_smoke",
        "profile": {
            "path": str(profile_path),
            "profile_id": profile.profile_id,
            "digest": profile.digest,
        },
        "host": {
            "hostname": socket.gethostname().split(".")[0],
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "torch_version": torch.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "offline_environment": offline,
        "assets": {
            "model_root": inventory["model_root"],
            "tokenizer_root": inventory["tokenizer_root"],
            "dataset_root": inventory["dataset_root"],
            "model_file_hashes": {row["path"]: row["sha256"] for row in inventory["model_files"]},
            "tokenizer_file_hashes": {row["path"]: row["sha256"] for row in inventory["tokenizer_files"]},
            "source_file_hashes": {row["path"]: row["sha256"] for row in inventory["source_files"]},
            "materialized_manifest_sha256": _file_digest(materialized_root / "manifest.json"),
            "materialized_marker_sha256": _file_digest(materialized_root / "complete.json"),
        },
        "model": {
            **registry,
            "initial_parameter_sha256": initial_parameter_sha256,
            "final_parameter_sha256": final_parameter_sha256,
            "model_parameter_dtype": "torch.float32",
            "compute_precision": str(profile.training["precision"]),
            "optimizer_state_dtypes": list(state_dtypes),
        },
        "data": {
            "initial_state": initial_data_state,
            "final_state": shard.state_dict(),
        },
        "runtime": run.to_dict(),
        "forbidden_runtime": {
            "torch_distributed_initialized": False,
            "hf_trainer_used": False,
            "proposal_published": False,
            "global_fragment_adopted": False,
        },
    }
    _write_json_new(output, result)
    return result


def _tiny_batches(torch: Any, *, microbatches: int) -> Iterator[dict[str, Any]]:
    for index in range(microbatches):
        input_ids = (torch.arange(64).reshape(4, 16) + index * 7) % 128
        attention_mask = torch.ones_like(input_ids)
        if index % 3 == 1:
            attention_mask[:, -2:] = 0
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        yield {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def run_tiny(output: Path, *, log_path: Path, device_name: str = "cuda") -> dict[str, Any]:
    import torch
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    if device_name == "cuda" and not torch.cuda.is_available():
        raise LearnerSmokeError("tiny CUDA smoke requires an available GPU")
    device = torch.device(device_name)
    torch.manual_seed(1606)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    model = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            max_position_embeddings=32,
            use_cache=False,
        )
    ).to(device)
    fragment_count = 2
    groups = _fragment_parameter_groups(model, fragment_count)
    before = _parameter_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = ConstantStepScheduler()
    progress = LearnerProgress.initialize("tiny-learner", (7, 11))
    rng = LearnerRng.initialize(progress.learner_id, 1606, device)
    runtime = LearnerRuntime(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
        rng=rng,
        fragment_parameters=groups,
        device=device,
        precision="fp32",
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        comparison={
            "matched_tokens": False,
            "matched_compute": False,
            "matched_communication": False,
            "claim": "semantic_smoke_only",
        },
        logger=StructuredLogger(log_path, role="learner", run_id="s1-06-tiny"),
    )
    run = runtime.run(_tiny_batches(torch, microbatches=20), optimizer_steps=10)
    after = _parameter_digest(model)
    if before == after:
        raise LearnerSmokeError("tiny run did not change model parameters")
    result = {
        "schema_version": 1,
        "status": "pass",
        "kind": "tiny_real_hf_smoke",
        "profile": {"profile_id": "s1-06-tiny-neox", "digest": canonical_digest({"seed": 1606})},
        "host": {
            "hostname": socket.gethostname().split(".")[0],
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "model": {
            **_registry_summary(model, fragment_count),
            "initial_parameter_sha256": before,
            "final_parameter_sha256": after,
        },
        "runtime": run.to_dict(),
        "forbidden_runtime": {
            "torch_distributed_initialized": False,
            "hf_trainer_used": False,
            "proposal_published": False,
            "global_fragment_adopted": False,
        },
    }
    _write_json_new(output, result)
    return result


def summarize_runs(tiny: Path, gpt2: Path, pythia: Path, output: Path) -> dict[str, Any]:
    inputs = {
        "tiny": json.loads(tiny.read_text(encoding="utf-8")),
        "gpt2_wikitext": json.loads(gpt2.read_text(encoding="utf-8")),
        "pythia160m_fineweb": json.loads(pythia.read_text(encoding="utf-8")),
    }
    profiles: dict[str, Any] = {}
    expected_real = {
        "gpt2_wikitext": {
            "profile_id": "s1-06-gpt2-wikitext-packed512-v1",
            "parameter_count": 124_439_808,
        },
        "pythia160m_fineweb": {
            "profile_id": "s1-06-pythia160m-finewebedu-packed512-v1",
            "parameter_count": 162_322_944,
        },
    }
    for name, row in inputs.items():
        if row.get("status") != "pass":
            raise LearnerSmokeError(f"{name} run did not pass")
        runtime = row.get("runtime", {})
        events = runtime.get("events", [])
        if runtime.get("optimizer_steps_completed") != 10 or len(events) != 10:
            raise LearnerSmokeError(f"{name} must report ten optimizer boundaries")
        expected_microbatches = 2 if name == "tiny" else 1
        if any(event.get("microbatches") != expected_microbatches for event in events):
            raise LearnerSmokeError(f"{name} gradient-accumulation boundaries disagree with the profile")
        losses = [event.get("token_weighted_loss") for event in events]
        norms = [norm for event in events for norm in event.get("fragment_update_norms", [])]
        if not losses or not norms or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in [*losses, *norms]):
            raise LearnerSmokeError(f"{name} loss or fragment norm is missing/nonfinite")
        if not runtime.get("parameters_changed") or runtime.get("parameter_update_norm", 0) <= 0:
            raise LearnerSmokeError(f"{name} did not prove parameter updates")
        rng = runtime.get("rng", {})
        expected_seed = 1606 if name == "tiny" else 20260714
        rng_hashes = (rng.get("state_sha256_before"), rng.get("state_sha256_after"))
        if (
            rng.get("learner_id") != runtime.get("learner_id")
            or rng.get("seed") != expected_seed
            or rng.get("scope") != "runtime_owned_forked_torch_rng"
            or not isinstance(rng.get("owner_pid"), int)
            or rng.get("owner_pid", 0) <= 0
            or rng.get("activation_count_before") != 0
            or rng.get("activation_count_after") != 1
            or rng.get("process_global_state_restored") is not True
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in rng_hashes
            )
        ):
            raise LearnerSmokeError(f"{name} learner RNG ownership evidence is invalid")
        forbidden = row.get("forbidden_runtime", {})
        if forbidden != {
            "torch_distributed_initialized": False,
            "hf_trainer_used": False,
            "proposal_published": False,
            "global_fragment_adopted": False,
        }:
            raise LearnerSmokeError(f"{name} exercised a forbidden runtime")
        if name in expected_real:
            expected = expected_real[name]
            if row.get("profile", {}).get("profile_id") != expected["profile_id"]:
                raise LearnerSmokeError(f"{name} profile identity mismatch")
            model = row.get("model", {})
            if (
                model.get("parameter_count") != expected["parameter_count"]
                or model.get("optimizer_state_dtypes") != ["torch.float32"]
                or model.get("initial_parameter_sha256") == model.get("final_parameter_sha256")
            ):
                raise LearnerSmokeError(f"{name} model or optimizer identity mismatch")
            if row.get("offline_environment") != {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }:
                raise LearnerSmokeError(f"{name} was not an offline-only run")
            if runtime.get("processed_input_tokens") != 20_480 or runtime.get("loss_bearing_target_tokens") != 20_440:
                raise LearnerSmokeError(f"{name} packed-512 token accounting mismatch")
        first = events[0]
        last = events[-1]
        interval = (
            runtime["training_end_monotonic_ns"] - runtime["training_start_monotonic_ns"]
        ) / 1_000_000_000
        expected_rate = runtime["processed_input_tokens"] / interval
        if not math.isclose(
            runtime["common_interval_input_tokens_per_second"],
            expected_rate,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise LearnerSmokeError(f"{name} throughput is not its common-interval rate")
        if first["local_optimizer_step"] + 9 != last["local_optimizer_step"]:
            raise LearnerSmokeError(f"{name} optimizer-step sequence is not contiguous")
        if any(value != 10 for value in last["fragment_local_steps"]):
            raise LearnerSmokeError(f"{name} per-fragment optimizer-step counters disagree")
        if any(value != runtime["processed_input_tokens"] for value in last["fragment_processed_input_tokens"]):
            raise LearnerSmokeError(f"{name} per-fragment input-token counters disagree")
        profiles[name] = {
            "profile_id": row["profile"]["profile_id"],
            "optimizer_steps": runtime["optimizer_steps_completed"],
            "processed_input_tokens": runtime["processed_input_tokens"],
            "loss_bearing_target_tokens": runtime["loss_bearing_target_tokens"],
            "token_weighted_loss": runtime["token_weighted_loss"],
            "parameter_update_norm": runtime["parameter_update_norm"],
            "common_interval_seconds": interval,
            "common_interval_input_tokens_per_second": expected_rate,
            "fragment_count": len(first["fragment_local_steps"]),
            "final_fragment_local_steps": last["fragment_local_steps"],
            "final_fragment_processed_input_tokens": last["fragment_processed_input_tokens"],
            "inactive_metrics": last["inactive_metrics"],
            "comparison": last["comparison"],
            "rng": rng,
        }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "profiles": profiles,
        "aggregation": {
            "throughput": "per-profile total input tokens divided by its own training-start-to-final-boundary interval",
            "combined_rank_rate": None,
            "loss": "per-profile loss numerator divided by loss-bearing target tokens",
        },
        "safe_boundary": "after accumulated backward gradient clip optimizer step scheduler step and zero_grad",
        "scheduler_basis": "local_optimizer_steps",
        "torch_distributed_initialized": False,
        "independent_profile_evidence": True,
    }
    _write_json_new(output, summary)
    return summary


def write_manifest(args: argparse.Namespace) -> None:
    modules = [
        line.strip()
        for line in Path(args.modules_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hostname = socket.gethostname().split(".")[0]
    identities = {
        "code": {
            "repository": args.repository,
            "branch": args.branch,
            "commit": args.commit,
            "dirty": False,
        },
        "config": {"sha256": args.config_sha256},
        "source": {
            "research_plan_sha256": args.research_sha256,
            "stage0_4_spec_sha256": args.spec_sha256,
        },
        "skill": {"repository": args.skill_repository, "commit": args.skill_commit},
        "execution": {
            "identity": args.run_id,
            "initial_hostname": args.initial_hostname,
            "compute_hostname": hostname,
            "workflow": "one-node-single-process-hf-learner-smoke",
        },
        "roles": {
            "declared": {"learner_smoke": [hostname]},
            "actual": {"learner_smoke": [hostname]},
        },
        "paths": {"project_root": args.project_root, "evidence_root": args.evidence_root},
    }
    manifest = build_manifest(
        "S1-06",
        "L1",
        args.timestamp_utc,
        "submission_utc",
        identities,
    )
    manifest["environment"] = {
        "modules": modules,
        "pbs_job_id": args.job_id,
        "pbs_queue": args.queue,
        "pbs_group": args.group,
        "node_type": args.node_type,
    }
    _write_json_new(Path(args.output), manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fsbdd.learner_smoke")
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--profile", type=Path, required=True)
    materialize.add_argument("--hub-cache", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--split", default="train")
    materialize.add_argument("--mode", choices=("smoke", "long_run"), default="smoke")
    real = commands.add_parser("run-real")
    real.add_argument("--profile", type=Path, required=True)
    real.add_argument("--hub-cache", type=Path, required=True)
    real.add_argument("--materialized-root", type=Path, required=True)
    real.add_argument("--output", type=Path, required=True)
    real.add_argument("--log", type=Path, required=True)
    tiny = commands.add_parser("run-tiny")
    tiny.add_argument("--output", type=Path, required=True)
    tiny.add_argument("--log", type=Path, required=True)
    tiny.add_argument("--device", default="cuda")
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--tiny", type=Path, required=True)
    summarize.add_argument("--gpt2", type=Path, required=True)
    summarize.add_argument("--pythia", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    manifest = commands.add_parser("manifest")
    for name in (
        "repository",
        "branch",
        "commit",
        "run-id",
        "config-sha256",
        "research-sha256",
        "spec-sha256",
        "skill-repository",
        "skill-commit",
        "initial-hostname",
        "project-root",
        "evidence-root",
        "job-id",
        "queue",
        "group",
        "node-type",
        "timestamp-utc",
    ):
        manifest.add_argument(f"--{name}", required=True)
    manifest.add_argument("--modules-file", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        profile = load_learner_profile(args.profile)
        summary = materialize_packed_shards(
            profile,
            args.hub_cache,
            args.output,
            split=args.split,
            mode=args.mode,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "run-real":
        summary = run_real_profile(
            args.profile,
            args.hub_cache,
            args.materialized_root,
            args.output,
            log_path=args.log,
        )
        print(json.dumps({"status": summary["status"], "profile": summary["profile"]}, sort_keys=True))
        return 0
    if args.command == "run-tiny":
        summary = run_tiny(args.output, log_path=args.log, device_name=args.device)
        print(json.dumps({"status": summary["status"], "profile": summary["profile"]}, sort_keys=True))
        return 0
    if args.command == "summarize":
        summary = summarize_runs(args.tiny, args.gpt2, args.pythia, args.output)
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "manifest":
        write_manifest(args)
        print(json.dumps({"status": "building", "output": str(args.output)}, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
