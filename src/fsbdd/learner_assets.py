from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from .identity import canonical_digest


class LearnerAssetError(ValueError):
    pass


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _strict(value: object, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LearnerAssetError(f"{name} must be an object")
    missing = fields - value.keys()
    unknown = value.keys() - fields
    if missing:
        raise LearnerAssetError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise LearnerAssetError(f"{name} has unknown fields: {sorted(unknown)}")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LearnerAssetError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise LearnerAssetError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LearnerAssetError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise LearnerAssetError(f"{name} must be finite and >= {minimum}")
    return result


def _sha256(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise LearnerAssetError(f"{name} must be a lowercase SHA-256")
    return text


def _revision(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise LearnerAssetError(f"{name} must be an immutable 40-character commit")
    return text


def _relative_path(value: object, name: str) -> str:
    text = _text(value, name)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise LearnerAssetError(f"{name} must be a safe relative path")
    return text


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenLearnerProfile:
    profile_id: str
    digest: str
    raw: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.raw)

    @property
    def model(self) -> Mapping[str, Any]:
        return self.raw["model"]

    @property
    def tokenizer(self) -> Mapping[str, Any]:
        return self.raw["tokenizer"]

    @property
    def dataset(self) -> Mapping[str, Any]:
        return self.raw["dataset"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.raw["training"]


_TOP = {"schema_version", "profile_id", "model", "tokenizer", "dataset", "training"}
_MODEL = {
    "repository",
    "revision",
    "expected_parameter_count",
    "attention_implementation",
    "use_cache",
    "files",
}
_TOKENIZER = {"repository", "revision", "eos_token_id", "add_special_tokens", "files"}
_DATASET_BASE = {
    "repository",
    "revision",
    "config",
    "text_field",
    "skip_rule",
    "normalization",
    "append_eos",
    "sequence_length",
    "shard_count",
    "shard_rule",
    "epoch_seed_base",
    "source_files",
}
_TRAINING = {
    "learner_id",
    "learner_index",
    "fragment_count",
    "batch_size",
    "gradient_accumulation_steps",
    "optimizer_steps",
    "precision",
    "optimizer",
    "scheduler",
    "seed",
    "initial_fragment_versions",
    "comparison",
}


def _validate_files(value: object, name: str, *, dataset: bool = False) -> None:
    if not isinstance(value, list) or not value:
        raise LearnerAssetError(f"{name} must be a non-empty list")
    seen: set[str] = set()
    fields = {"path", "bytes", "sha256"} | ({"split"} if dataset else set())
    for index, item in enumerate(value):
        row = _strict(item, f"{name}[{index}]", fields)
        path = _relative_path(row["path"], f"{name}[{index}].path")
        if path in seen:
            raise LearnerAssetError(f"{name} contains duplicate path: {path}")
        seen.add(path)
        _integer(row["bytes"], f"{name}[{index}].bytes", minimum=1)
        _sha256(row["sha256"], f"{name}[{index}].sha256")
        if dataset:
            _text(row["split"], f"{name}[{index}].split")


def load_learner_profile(path: Path) -> FrozenLearnerProfile:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LearnerAssetError(f"cannot load learner profile: {error}") from error
    top = _strict(raw, "profile", _TOP)
    if top["schema_version"] != 1:
        raise LearnerAssetError("unsupported learner profile schema_version")
    profile_id = _text(top["profile_id"], "profile_id")

    model = _strict(top["model"], "model", _MODEL)
    tokenizer = _strict(top["tokenizer"], "tokenizer", _TOKENIZER)
    dataset_fields = _DATASET_BASE | ({"expected_splits"} if profile_id.startswith("s1-06-gpt2") else {"smoke", "long_run"})
    dataset = _strict(top["dataset"], "dataset", dataset_fields)
    training = _strict(top["training"], "training", _TRAINING)

    for name, section in (("model", model), ("tokenizer", tokenizer), ("dataset", dataset)):
        _text(section["repository"], f"{name}.repository")
        _revision(section["revision"], f"{name}.revision")
    if model["repository"] != tokenizer["repository"] or model["revision"] != tokenizer["revision"]:
        raise LearnerAssetError("model and tokenizer must use the same frozen repository revision")
    _integer(model["expected_parameter_count"], "model.expected_parameter_count", minimum=1)
    _text(model["attention_implementation"], "model.attention_implementation")
    if not isinstance(model["use_cache"], bool):
        raise LearnerAssetError("model.use_cache must be boolean")
    _validate_files(model["files"], "model.files")
    _integer(tokenizer["eos_token_id"], "tokenizer.eos_token_id")
    if tokenizer["add_special_tokens"] is not False:
        raise LearnerAssetError("tokenizer.add_special_tokens must be false")
    _validate_files(tokenizer["files"], "tokenizer.files")

    for key in ("text_field", "skip_rule", "normalization", "shard_rule"):
        _text(dataset[key], f"dataset.{key}")
    if dataset["skip_rule"] != "text_equals_empty_string" or dataset["normalization"] != "none":
        raise LearnerAssetError("dataset text policy must skip only empty strings without normalization")
    if dataset["append_eos"] is not True:
        raise LearnerAssetError("dataset.append_eos must be true")
    sequence_length = _integer(dataset["sequence_length"], "dataset.sequence_length", minimum=2)
    shard_count = _integer(dataset["shard_count"], "dataset.shard_count", minimum=1)
    _integer(dataset["epoch_seed_base"], "dataset.epoch_seed_base")
    _validate_files(dataset["source_files"], "dataset.source_files", dataset=True)
    if profile_id.startswith("s1-06-gpt2"):
        if not isinstance(dataset["expected_splits"], dict):
            raise LearnerAssetError("dataset.expected_splits must be an object")
    else:
        smoke = _strict(dataset["smoke"], "dataset.smoke", {"minimum_blocks_per_shard", "source_policy"})
        _integer(smoke["minimum_blocks_per_shard"], "dataset.smoke.minimum_blocks_per_shard", minimum=1)
        if smoke["source_policy"] != "canonical_prefix_until_minimum_then_finish_current_row":
            raise LearnerAssetError("unsupported smoke source policy")
        long_run = _strict(
            dataset["long_run"],
            "dataset.long_run",
            {"source_path_pattern", "source_order", "minimum_processed_input_tokens", "minimum_packed_blocks"},
        )
        if long_run["source_order"] != "bytewise_lexicographic":
            raise LearnerAssetError("long-run source order must be bytewise lexicographic")
        minimum_tokens = _integer(
            long_run["minimum_processed_input_tokens"],
            "dataset.long_run.minimum_processed_input_tokens",
            minimum=1_000_000_000,
        )
        minimum_blocks = _integer(long_run["minimum_packed_blocks"], "dataset.long_run.minimum_packed_blocks", minimum=1)
        if minimum_blocks * sequence_length < minimum_tokens:
            raise LearnerAssetError("long-run packed block count does not satisfy its token budget")

    for key in (
        "learner_id",
        "precision",
    ):
        _text(training[key], f"training.{key}")
    learner_index = _integer(training["learner_index"], "training.learner_index")
    if learner_index >= shard_count:
        raise LearnerAssetError("training.learner_index must select a frozen data shard")
    fragment_count = _integer(training["fragment_count"], "training.fragment_count", minimum=1)
    _integer(training["batch_size"], "training.batch_size", minimum=1)
    _integer(training["gradient_accumulation_steps"], "training.gradient_accumulation_steps", minimum=1)
    _integer(training["optimizer_steps"], "training.optimizer_steps", minimum=1)
    _integer(training["seed"], "training.seed")
    versions = training["initial_fragment_versions"]
    if not isinstance(versions, list) or len(versions) != fragment_count:
        raise LearnerAssetError("training.initial_fragment_versions must match fragment_count")
    for index, version in enumerate(versions):
        _integer(version, f"training.initial_fragment_versions[{index}]")
    optimizer = _strict(
        training["optimizer"],
        "training.optimizer",
        {"class", "lr", "betas", "eps", "weight_decay", "grad_clip_norm"},
    )
    if optimizer["class"] != "AdamW":
        raise LearnerAssetError("training optimizer must be AdamW")
    for key in ("lr", "eps", "weight_decay", "grad_clip_norm"):
        _number(optimizer[key], f"training.optimizer.{key}", minimum=0.0)
    betas = optimizer["betas"]
    if not isinstance(betas, list) or len(betas) != 2 or any(not 0 <= _number(beta, "optimizer beta") < 1 for beta in betas):
        raise LearnerAssetError("training.optimizer.betas must contain two values in [0,1)")
    scheduler = _strict(training["scheduler"], "training.scheduler", {"class", "basis"})
    if scheduler != {"class": "constant", "basis": "local_optimizer_steps"}:
        raise LearnerAssetError("only constant local-optimizer-step scheduling is supported")
    comparison = _strict(
        training["comparison"],
        "training.comparison",
        {"matched_tokens", "matched_compute", "matched_communication", "claim"},
    )
    for key in ("matched_tokens", "matched_compute", "matched_communication"):
        if not isinstance(comparison[key], bool):
            raise LearnerAssetError(f"training.comparison.{key} must be boolean")
    _text(comparison["claim"], "training.comparison.claim")

    canonical = json.loads(json.dumps(raw, sort_keys=True))
    return FrozenLearnerProfile(profile_id, canonical_digest(canonical), _freeze(canonical))


def _repository_cache_name(repository: str, *, repo_type: str) -> str:
    prefix = "models" if repo_type == "model" else "datasets"
    return f"{prefix}--{repository.replace('/', '--')}"


def snapshot_path(hub_cache: Path, repository: str, revision: str, *, repo_type: str) -> Path:
    path = hub_cache / _repository_cache_name(repository, repo_type=repo_type) / "snapshots" / revision
    if not path.is_dir():
        raise LearnerAssetError(f"frozen {repo_type} snapshot is missing: {path}")
    return path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_files(root: Path, rows: object, name: str) -> tuple[dict[str, Any], ...]:
    assert isinstance(rows, tuple)
    verified: list[dict[str, Any]] = []
    for index, frozen_row in enumerate(rows):
        row = _thaw(frozen_row)
        path = root / row["path"]
        if not path.is_file():
            raise LearnerAssetError(f"{name}[{index}] is missing: {path}")
        size = path.stat().st_size
        digest = _hash_file(path)
        if size != row["bytes"] or digest != row["sha256"]:
            raise LearnerAssetError(f"{name}[{index}] identity mismatch: {path}")
        verified.append({**row, "resolved_path": str(path)})
    return tuple(verified)


def verify_profile_assets(profile: FrozenLearnerProfile, hub_cache: Path) -> dict[str, Any]:
    model_root = snapshot_path(
        hub_cache,
        str(profile.model["repository"]),
        str(profile.model["revision"]),
        repo_type="model",
    )
    tokenizer_root = snapshot_path(
        hub_cache,
        str(profile.tokenizer["repository"]),
        str(profile.tokenizer["revision"]),
        repo_type="model",
    )
    dataset_root = snapshot_path(
        hub_cache,
        str(profile.dataset["repository"]),
        str(profile.dataset["revision"]),
        repo_type="dataset",
    )
    return {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "profile_digest": profile.digest,
        "model_root": str(model_root),
        "tokenizer_root": str(tokenizer_root),
        "dataset_root": str(dataset_root),
        "model_files": _verify_files(model_root, profile.model["files"], "model.files"),
        "tokenizer_files": _verify_files(tokenizer_root, profile.tokenizer["files"], "tokenizer.files"),
        "source_files": _verify_files(dataset_root, profile.dataset["source_files"], "dataset.source_files"),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _expected_split(profile: FrozenLearnerProfile, split: str) -> dict[str, Any] | None:
    expected = profile.dataset.get("expected_splits")
    if not isinstance(expected, Mapping) or split not in expected:
        return None
    return _thaw(expected[split])


def materialize_packed_shards(
    profile: FrozenLearnerProfile,
    hub_cache: Path,
    output: Path,
    *,
    split: str = "train",
    mode: str = "smoke",
) -> dict[str, Any]:
    if mode not in {"smoke", "long_run"}:
        raise LearnerAssetError("materialization mode must be smoke or long_run")
    if output.exists():
        raise LearnerAssetError(f"refusing to overwrite materialized dataset: {output}")
    inventory = verify_profile_assets(profile, hub_cache)
    source_rows = [row for row in inventory["source_files"] if row["split"] == split]
    long_run = profile.dataset.get("long_run")
    if mode == "long_run":
        if not isinstance(long_run, Mapping):
            raise LearnerAssetError("profile has no long-run source contract")
        dataset_root = Path(inventory["dataset_root"])
        pattern = _relative_path(long_run["source_path_pattern"], "long-run source pattern")
        paths = sorted(
            dataset_root.glob(pattern),
            key=lambda path: path.relative_to(dataset_root).as_posix().encode("utf-8"),
        )
        if not paths:
            raise LearnerAssetError("no frozen long-run source files are staged")
        source_rows = [
            {
                "split": split,
                "path": path.relative_to(dataset_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
                "resolved_path": str(path),
            }
            for path in paths
        ]
    if not source_rows:
        raise LearnerAssetError(f"profile has no source files for split {split!r}")

    from pyarrow import parquet
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        inventory["tokenizer_root"],
        local_files_only=True,
    )
    expected_eos = int(profile.tokenizer["eos_token_id"])
    if tokenizer.eos_token_id != expected_eos:
        raise LearnerAssetError("tokenizer EOS identity mismatch")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    streams: list[Any] = []
    try:
        sequence_length = int(profile.dataset["sequence_length"])
        shard_count = int(profile.dataset["shard_count"])
        shard_paths = [temporary / f"shard-{index:03d}.uint32le.bin" for index in range(shard_count)]
        streams = [path.open("wb") for path in shard_paths]
        shard_hashes = [hashlib.sha256() for _ in range(shard_count)]
        shard_blocks = [0 for _ in range(shard_count)]
        token_stream_hash = hashlib.sha256()
        pending: list[int] = []
        pending_start = 0
        total_rows = 0
        nonempty_rows = 0
        tokens_before_drop = 0
        packed_blocks = 0
        stop = False
        smoke = profile.dataset.get("smoke")
        minimum_per_shard = (
            int(smoke["minimum_blocks_per_shard"])
            if mode == "smoke" and isinstance(smoke, Mapping)
            else None
        )
        minimum_long_run_blocks = (
            int(long_run["minimum_packed_blocks"])
            if mode == "long_run" and isinstance(long_run, Mapping)
            else None
        )

        for source in source_rows:
            file = parquet.ParquetFile(source["resolved_path"])
            for batch in file.iter_batches(columns=[str(profile.dataset["text_field"])], batch_size=256):
                texts = batch.column(0).to_pylist()
                for text in texts:
                    total_rows += 1
                    if not isinstance(text, str):
                        raise LearnerAssetError("dataset text field must contain strings")
                    if text == "":
                        continue
                    nonempty_rows += 1
                    encoded = tokenizer.encode(text, add_special_tokens=False)
                    row_tokens = [*encoded, expected_eos]
                    if any(token < 0 or token > np.iinfo(np.uint32).max for token in row_tokens):
                        raise LearnerAssetError("token ID does not fit uint32")
                    row_bytes = np.asarray(row_tokens, dtype="<u4").tobytes()
                    token_stream_hash.update(row_bytes)
                    tokens_before_drop += len(row_tokens)
                    pending.extend(row_tokens)
                    while len(pending) - pending_start >= sequence_length:
                        block = pending[pending_start : pending_start + sequence_length]
                        pending_start += sequence_length
                        block_bytes = np.asarray(block, dtype="<u4").tobytes()
                        shard = packed_blocks % shard_count
                        streams[shard].write(block_bytes)
                        shard_hashes[shard].update(block_bytes)
                        shard_blocks[shard] += 1
                        packed_blocks += 1
                    if pending_start and pending_start >= 4 * sequence_length:
                        pending = pending[pending_start:]
                        pending_start = 0
                    if minimum_per_shard is not None and min(shard_blocks) >= minimum_per_shard:
                        stop = True
                        break
                    if minimum_long_run_blocks is not None and packed_blocks >= minimum_long_run_blocks:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break

        remainder = len(pending) - pending_start
        for stream in streams:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        streams = []

        expected = _expected_split(profile, split)
        if expected is not None:
            observed = {
                "source_rows": total_rows,
                "nonempty_rows": nonempty_rows,
                "tokens_before_drop": tokens_before_drop,
                "packed_blocks": packed_blocks,
                "remainder_tokens": remainder,
            }
            for key, value in observed.items():
                if expected.get(key) != value:
                    raise LearnerAssetError(
                        f"{split} preprocessing drift for {key}: expected {expected.get(key)}, observed {value}"
                    )
            if "shard_block_counts" in expected and expected["shard_block_counts"] != shard_blocks:
                raise LearnerAssetError("preprocessed shard block counts drifted")
        if minimum_per_shard is not None and min(shard_blocks) < minimum_per_shard:
            raise LearnerAssetError("source exhausted before every smoke shard was complete")
        if minimum_long_run_blocks is not None and packed_blocks < minimum_long_run_blocks:
            raise LearnerAssetError("staged sources exhausted before the long-run token budget")

        shard_inventory = []
        for index, path in enumerate(shard_paths):
            shard_inventory.append(
                {
                    "learner_index": index,
                    "path": path.name,
                    "blocks": shard_blocks[index],
                    "tokens": shard_blocks[index] * sequence_length,
                    "bytes": path.stat().st_size,
                    "sha256": shard_hashes[index].hexdigest(),
                }
            )
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "profile_id": profile.profile_id,
            "profile_digest": profile.digest,
            "split": split,
            "mode": (
                "long_run_prefix"
                if minimum_long_run_blocks is not None
                else "smoke_prefix"
                if minimum_per_shard is not None
                else "complete_split"
            ),
            "token_dtype": "uint32-le",
            "sequence_length": sequence_length,
            "source_files": [
                {key: row[key] for key in ("split", "path", "bytes", "sha256")}
                for row in source_rows
            ],
            "source_rows_consumed": total_rows,
            "nonempty_rows_consumed": nonempty_rows,
            "tokens_before_drop": tokens_before_drop,
            "token_stream_sha256": token_stream_hash.hexdigest(),
            "packed_blocks": packed_blocks,
            "remainder_tokens": remainder,
            "shards": shard_inventory,
        }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        marker = {
            "schema_version": 1,
            "status": "complete",
            "profile_digest": profile.digest,
            "manifest_sha256": _hash_file(manifest_path),
        }
        _write_json(temporary / "complete.json", marker)
        os.replace(temporary, output)
        return manifest
    except Exception:
        for stream in streams:
            stream.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_materialized_shards(
    profile: FrozenLearnerProfile,
    root: Path,
) -> dict[str, Any]:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        marker = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LearnerAssetError(f"materialized dataset metadata is unreadable: {error}") from error
    if marker != {
        "schema_version": 1,
        "status": "complete",
        "profile_digest": profile.digest,
        "manifest_sha256": _hash_file(root / "manifest.json"),
    }:
        raise LearnerAssetError("materialized dataset completion marker mismatch")
    if manifest.get("status") != "complete" or manifest.get("profile_digest") != profile.digest:
        raise LearnerAssetError("materialized dataset profile identity mismatch")
    sequence_length = int(profile.dataset["sequence_length"])
    for row in manifest.get("shards", []):
        path = root / _relative_path(row.get("path"), "materialized shard path")
        expected_bytes = _integer(row.get("bytes"), "materialized shard bytes", minimum=1)
        if expected_bytes != row.get("blocks") * sequence_length * np.dtype("<u4").itemsize:
            raise LearnerAssetError("materialized shard shape and byte count disagree")
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise LearnerAssetError(f"materialized shard is missing or truncated: {path}")
        if _hash_file(path) != _sha256(row.get("sha256"), "materialized shard sha256"):
            raise LearnerAssetError(f"materialized shard checksum mismatch: {path}")
    return manifest
