from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fsbdd.diloco.model.fragment_map import build_fragment_map
from fsbdd.diloco.common.identity import canonical_digest
from fsbdd.diloco.model.learner_assets import (
    load_learner_profile,
    materialize_packed_shards,
    validate_materialized_shards,
    verify_profile_assets,
)
from fsbdd.diloco.learner.publication import (
    FragmentPublicationSchedule,
    build_fragment_descriptors,
    build_fragment_parameter_groups,
    serialize_fragment_parameters,
)
from fsbdd.diloco.model.huggingface import (
    load_frozen_causal_lm,
    model_parameter_digest,
)
from fsbdd.diloco.model.model_registry import build_logical_layer_registry


class Stage1AssetError(RuntimeError):
    pass


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _copy_validated_dataset(
    profile: Any, source: Path, destination: Path
) -> dict[str, Any]:
    manifest = validate_materialized_shards(profile, source)
    if destination.exists():
        raise Stage1AssetError(f"refusing to overwrite dataset asset: {destination}")
    shutil.copytree(source, destination, copy_function=os.link)
    validate_materialized_shards(profile, destination)
    return manifest


def _dataset_asset(
    profile: Any,
    hub_cache: Path,
    destination: Path,
    *,
    split: str,
    mode: str,
    reuse: Path | None,
) -> dict[str, Any]:
    if reuse is not None:
        manifest = _copy_validated_dataset(profile, reuse, destination)
        if manifest.get("split") != split:
            raise Stage1AssetError(
                "reused dataset split differs from the requested split"
            )
        return manifest
    return materialize_packed_shards(
        profile,
        hub_cache,
        destination,
        split=split,
        mode=mode,
    )


def _bootstrap_model(
    *,
    profile: Any,
    hub_cache: Path,
    destination: Path,
    learner_count: int,
    h: int,
) -> dict[str, Any]:
    import torch

    inventory = verify_profile_assets(profile, hub_cache)
    model = load_frozen_causal_lm(profile, inventory, torch.device("cpu"))
    registry = build_logical_layer_registry(model)
    fragment_map = build_fragment_map(registry, int(profile.training["fragment_count"]))
    descriptors = build_fragment_descriptors(fragment_map)
    groups = build_fragment_parameter_groups(model, registry, fragment_map)
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "registry.json", registry.to_dict())
    _write_json(destination / "fragment-map.json", fragment_map.to_dict())
    fragment_rows: list[dict[str, Any]] = []
    fragment_bytes: list[int] = []
    for descriptor, group in zip(descriptors, groups, strict=True):
        expected_bytes = int(descriptor.shape[0]) * 4
        payload = serialize_fragment_parameters(group, expected_bytes)
        relative = f"fragments/fragment-{descriptor.index:03d}.fp32le.bin"
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            if stream.write(payload) != len(payload):
                raise Stage1AssetError("bootstrap fragment write was incomplete")
            stream.flush()
            os.fsync(stream.fileno())
        fragment_bytes.append(len(payload))
        fragment_rows.append(
            {
                "descriptor": descriptor.to_dict(),
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    schedules = [
        FragmentPublicationSchedule.unified(
            fragment_bytes,
            h,
            learner_phase_offset=(learner_index * h) // learner_count,
        ).to_dict()
        for learner_index in range(learner_count)
    ]
    result = {
        "schema_version": 1,
        "status": "complete",
        "profile_id": profile.profile_id,
        "profile_digest": profile.digest,
        "model_repository": str(profile.model["repository"]),
        "model_revision": str(profile.model["revision"]),
        "model_root": inventory["model_root"],
        "model_files": [
            {key: row[key] for key in ("path", "bytes", "sha256")}
            for row in inventory["model_files"]
        ],
        "initial_model_parameter_sha256": model_parameter_digest(model),
        "registry_path": "registry.json",
        "registry_sha256": _hash_file(destination / "registry.json"),
        "registry_digest": registry.digest,
        "fragment_map_path": "fragment-map.json",
        "fragment_map_sha256": _hash_file(destination / "fragment-map.json"),
        "fragment_map_digest": fragment_map.digest,
        "fragment_count": len(descriptors),
        "fragments": fragment_rows,
        "fragment_bytes": fragment_bytes,
        "h": h,
        "learner_count": learner_count,
        "per_learner_schedules": schedules,
    }
    result["bootstrap_identity"] = canonical_digest(result)
    _write_json(destination / "manifest.json", result)
    del groups, descriptors, fragment_map, registry, model
    gc.collect()
    return result


def _dataset_summary(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root": str(root.resolve()),
        "manifest_sha256": _hash_file(root / "manifest.json"),
        "complete_sha256": _hash_file(root / "complete.json"),
        "profile_digest": manifest["profile_digest"],
        "split": manifest["split"],
        "mode": manifest["mode"],
        "packed_blocks": manifest["packed_blocks"],
        "packed_input_tokens": manifest["packed_blocks"] * manifest["sequence_length"],
        "shards": manifest["shards"],
        "source_files": manifest["source_files"],
    }


def prepare_assets(
    config_path: Path,
    hub_cache: Path,
    output: Path,
    *,
    reuse_gpt_train: Path | None = None,
    reuse_pythia_smoke: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise Stage1AssetError(f"refusing to overwrite asset bundle: {output}")
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    if config.get("profile_id") != "s1-13-stage1-close-v1":
        raise Stage1AssetError("unexpected Stage 1 closure config")
    nine = config["workloads"]["nine_node"]
    long = config["workloads"]["long_run"]
    project_root = config_path.resolve().parents[2]
    gpt_profile_path = project_root / nine["learner_profile"]
    pythia_profile_path = project_root / long["learner_profile"]
    gpt_profile = load_learner_profile(gpt_profile_path)
    pythia_profile = load_learner_profile(pythia_profile_path)
    if (
        gpt_profile.model["revision"] != nine["model_revision"]
        or gpt_profile.dataset["revision"] != nine["dataset_revision"]
        or pythia_profile.model["revision"] != long["model_revision"]
        or pythia_profile.dataset["revision"] != long["dataset_revision"]
    ):
        raise Stage1AssetError("closure config and frozen learner profiles disagree")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        dataset_root = temporary / "datasets"
        gpt_train = _dataset_asset(
            gpt_profile,
            hub_cache,
            dataset_root / "gpt2-train",
            split="train",
            mode="smoke",
            reuse=reuse_gpt_train,
        )
        gpt_validation = _dataset_asset(
            gpt_profile,
            hub_cache,
            dataset_root / "gpt2-validation",
            split="validation",
            mode="smoke",
            reuse=None,
        )
        pythia_smoke = _dataset_asset(
            pythia_profile,
            hub_cache,
            dataset_root / "pythia-smoke",
            split="train",
            mode="smoke",
            reuse=reuse_pythia_smoke,
        )
        pythia_long = _dataset_asset(
            pythia_profile,
            hub_cache,
            dataset_root / "pythia-long",
            split="train",
            mode="long_run",
            reuse=None,
        )
        h = int(config["protocol"]["H"])
        nine_bootstrap = _bootstrap_model(
            profile=gpt_profile,
            hub_cache=hub_cache,
            destination=temporary / "workloads" / "nine_node",
            learner_count=int(nine["learner_count"]),
            h=h,
        )
        long_bootstrap = _bootstrap_model(
            profile=pythia_profile,
            hub_cache=hub_cache,
            destination=temporary / "workloads" / "long_run",
            learner_count=int(long["learner_count"]),
            h=h,
        )
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "kind": "s1_13_immutable_asset_bundle",
            "config_source_path": str(config_path.resolve()),
            "config_source_sha256": hashlib.sha256(config_raw).hexdigest(),
            "profiles": {
                "nine_node": {
                    "path": str(gpt_profile_path),
                    "digest": gpt_profile.digest,
                },
                "long_run": {
                    "path": str(pythia_profile_path),
                    "digest": pythia_profile.digest,
                },
            },
            "datasets": {
                "gpt2_train": _dataset_summary(dataset_root / "gpt2-train", gpt_train),
                "gpt2_validation": _dataset_summary(
                    dataset_root / "gpt2-validation", gpt_validation
                ),
                "pythia_smoke": _dataset_summary(
                    dataset_root / "pythia-smoke", pythia_smoke
                ),
                "pythia_long": _dataset_summary(
                    dataset_root / "pythia-long", pythia_long
                ),
            },
            "workloads": {
                "nine_node": {
                    "root": str((temporary / "workloads" / "nine_node").resolve()),
                    "manifest_sha256": _hash_file(
                        temporary / "workloads" / "nine_node" / "manifest.json"
                    ),
                    "bootstrap_identity": nine_bootstrap["bootstrap_identity"],
                },
                "long_run": {
                    "root": str((temporary / "workloads" / "long_run").resolve()),
                    "manifest_sha256": _hash_file(
                        temporary / "workloads" / "long_run" / "manifest.json"
                    ),
                    "bootstrap_identity": long_bootstrap["bootstrap_identity"],
                },
            },
        }
        # Paths above must identify the final immutable bundle, not its staging name.
        temporary_text = str(temporary.resolve())
        final_text = str(output.resolve())
        manifest = json.loads(json.dumps(manifest).replace(temporary_text, final_text))
        _write_json(temporary / "manifest.json", manifest)
        marker = {
            "schema_version": 1,
            "status": "complete",
            "kind": "s1_13_immutable_asset_bundle",
            "manifest_sha256": _hash_file(temporary / "manifest.json"),
        }
        _write_json(temporary / "complete.json", marker)
        marker_sha256 = _hash_file(temporary / "complete.json")
        for workload, bootstrap in (
            ("nine_node", nine_bootstrap),
            ("long_run", long_bootstrap),
        ):
            resolved = json.loads(json.dumps(config))
            resolved["resolved_runtime_fields"] = {
                "fragment_descriptors": [
                    row["descriptor"] for row in bootstrap["fragments"]
                ],
                "fragment_bytes": bootstrap["fragment_bytes"],
                "fragment_offsets": bootstrap["per_learner_schedules"][0]["offsets"],
                "per_learner_offsets": [
                    row["offsets"] for row in bootstrap["per_learner_schedules"]
                ],
                "asset_marker_sha256": marker_sha256,
                "asset_bundle_root": final_text,
                "workload_bootstrap_identity": bootstrap["bootstrap_identity"],
            }
            resolved["selected_workload"] = workload
            _write_json(temporary / "resolved" / f"{workload}.json", resolved)
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare immutable S1-13 model/data/bootstrap assets"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--hub-cache", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--reuse-gpt-train", type=Path)
    prepare.add_argument("--reuse-pythia-smoke", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        result = prepare_assets(
            arguments.config,
            arguments.hub_cache,
            arguments.output,
            reuse_gpt_train=arguments.reuse_gpt_train,
            reuse_pythia_smoke=arguments.reuse_pythia_smoke,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
