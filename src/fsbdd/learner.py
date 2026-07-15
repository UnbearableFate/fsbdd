from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .learner_assets import FrozenLearnerProfile, LearnerAssetError, validate_materialized_shards
from .logging import StructuredLogger


class LearnerError(RuntimeError):
    pass


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LearnerError(f"{name} must be a non-negative integer")
    return value


@dataclasses.dataclass(slots=True)
class FragmentProgress:
    global_version: int
    local_steps_since_adoption: int = 0
    processed_input_tokens_since_adoption: int = 0
    proposal_sequence: int = 0

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _nonnegative_int(getattr(self, field.name), f"fragment.{field.name}")

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(slots=True)
class LearnerProgress:
    learner_id: str
    local_optimizer_steps: int
    processed_input_tokens: int
    loss_bearing_target_tokens: int
    fragments: list[FragmentProgress]

    def __post_init__(self) -> None:
        if not isinstance(self.learner_id, str) or not self.learner_id:
            raise LearnerError("learner_id must be a non-empty string")
        for name in (
            "local_optimizer_steps",
            "processed_input_tokens",
            "loss_bearing_target_tokens",
        ):
            _nonnegative_int(getattr(self, name), name)
        if not self.fragments:
            raise LearnerError("learner progress requires at least one fragment")
        if any(not isinstance(fragment, FragmentProgress) for fragment in self.fragments):
            raise LearnerError("learner fragments must be FragmentProgress values")

    @classmethod
    def initialize(cls, learner_id: str, versions: Sequence[int]) -> LearnerProgress:
        if not versions:
            raise LearnerError("initial fragment version vector must be non-empty")
        return cls(
            learner_id=learner_id,
            local_optimizer_steps=0,
            processed_input_tokens=0,
            loss_bearing_target_tokens=0,
            fragments=[FragmentProgress(_nonnegative_int(version, "global_version")) for version in versions],
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LearnerProgress:
        expected = {
            "learner_id",
            "local_optimizer_steps",
            "processed_input_tokens",
            "loss_bearing_target_tokens",
            "fragments",
        }
        if set(value) != expected or not isinstance(value["fragments"], list):
            raise LearnerError("serialized learner progress schema mismatch")
        fragments = []
        for row in value["fragments"]:
            if not isinstance(row, dict):
                raise LearnerError("serialized fragment progress must be an object")
            try:
                fragments.append(FragmentProgress(**row))
            except TypeError as error:
                raise LearnerError("serialized fragment progress schema mismatch") from error
        return cls(
            learner_id=value["learner_id"],
            local_optimizer_steps=value["local_optimizer_steps"],
            processed_input_tokens=value["processed_input_tokens"],
            loss_bearing_target_tokens=value["loss_bearing_target_tokens"],
            fragments=fragments,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "local_optimizer_steps": self.local_optimizer_steps,
            "processed_input_tokens": self.processed_input_tokens,
            "loss_bearing_target_tokens": self.loss_bearing_target_tokens,
            "fragments": [fragment.to_dict() for fragment in self.fragments],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class SafeBoundaryEvent:
    learner_id: str
    local_optimizer_step: int
    microbatches: int
    processed_input_tokens_step: int
    processed_input_tokens_total: int
    loss_bearing_target_tokens_step: int
    loss_bearing_target_tokens_total: int
    token_weighted_loss: float
    loss_numerator: float
    gradient_norm: float
    learning_rates: tuple[float, ...]
    lr_schedule_basis: str
    step_start_monotonic_ns: int
    safe_boundary_monotonic_ns: int
    step_latency_seconds: float
    step_input_tokens_per_second: float
    fragment_global_versions: tuple[int, ...]
    fragment_local_steps: tuple[int, ...]
    fragment_processed_input_tokens: tuple[int, ...]
    fragment_update_norms: tuple[float, ...]
    distributed_initialized: bool
    comparison: Mapping[str, Any]
    inactive_metrics: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["learning_rates"] = list(self.learning_rates)
        for key in (
            "fragment_global_versions",
            "fragment_local_steps",
            "fragment_processed_input_tokens",
            "fragment_update_norms",
        ):
            value[key] = list(value[key])
        return value


@dataclasses.dataclass(frozen=True, slots=True)
class LearnerRunSummary:
    learner_id: str
    training_start_monotonic_ns: int
    training_end_monotonic_ns: int
    optimizer_steps_completed: int
    processed_input_tokens: int
    loss_bearing_target_tokens: int
    token_weighted_loss: float
    common_interval_input_tokens_per_second: float
    parameter_update_norm: float
    parameters_changed: bool
    distributed_initialized: bool
    progress: Mapping[str, Any]
    events: tuple[SafeBoundaryEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "training_start_monotonic_ns": self.training_start_monotonic_ns,
            "training_end_monotonic_ns": self.training_end_monotonic_ns,
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "processed_input_tokens": self.processed_input_tokens,
            "loss_bearing_target_tokens": self.loss_bearing_target_tokens,
            "token_weighted_loss": self.token_weighted_loss,
            "common_interval_input_tokens_per_second": self.common_interval_input_tokens_per_second,
            "parameter_update_norm": self.parameter_update_norm,
            "parameters_changed": self.parameters_changed,
            "distributed_initialized": self.distributed_initialized,
            "progress": dict(self.progress),
            "events": [event.to_dict() for event in self.events],
        }


class StepScheduler(Protocol):
    step_count: int

    def step(self) -> None: ...


@dataclasses.dataclass(slots=True)
class ConstantStepScheduler:
    step_count: int = 0

    def step(self) -> None:
        self.step_count += 1

    def state_dict(self) -> dict[str, int]:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"step_count"}:
            raise LearnerError("constant scheduler state schema mismatch")
        self.step_count = _nonnegative_int(state["step_count"], "scheduler.step_count")


class PackedTokenShard:
    def __init__(
        self,
        profile: FrozenLearnerProfile,
        materialized_root: Path,
        *,
        learner_index: int,
        epoch: int = 0,
        position: int = 0,
    ) -> None:
        try:
            manifest = validate_materialized_shards(profile, materialized_root)
        except LearnerAssetError as error:
            raise LearnerError(str(error)) from error
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not 0 <= learner_index < len(shards):
            raise LearnerError("learner index is outside the materialized shard set")
        row = shards[learner_index]
        if row.get("learner_index") != learner_index:
            raise LearnerError("materialized shard learner identity mismatch")
        self.profile_digest = profile.digest
        self.learner_index = learner_index
        self.sequence_length = int(manifest["sequence_length"])
        self.blocks = int(row["blocks"])
        if self.blocks <= 0:
            raise LearnerError("materialized learner shard is empty")
        self.path = materialized_root / row["path"]
        self._tokens = np.memmap(
            self.path,
            dtype="<u4",
            mode="r",
            shape=(self.blocks, self.sequence_length),
        )
        self.epoch = _nonnegative_int(epoch, "data epoch")
        self.position = _nonnegative_int(position, "data position")
        if self.position > self.blocks:
            raise LearnerError("data position exceeds shard length")
        self._seed_base = int(profile.dataset["epoch_seed_base"]) + learner_index
        self._order = self._epoch_order(self.epoch)

    def _epoch_order(self, epoch: int) -> np.ndarray:
        return np.random.default_rng(self._seed_base + epoch).permutation(self.blocks)

    def state_dict(self) -> dict[str, Any]:
        return {
            "profile_digest": self.profile_digest,
            "learner_index": self.learner_index,
            "epoch": self.epoch,
            "position": self.position,
        }

    def next_batch(self, batch_size: int) -> dict[str, Any]:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise LearnerError("batch_size must be a positive integer")
        rows: list[np.ndarray] = []
        while len(rows) < batch_size:
            if self.position == self.blocks:
                self.epoch += 1
                self.position = 0
                self._order = self._epoch_order(self.epoch)
            take = min(batch_size - len(rows), self.blocks - self.position)
            indices = self._order[self.position : self.position + take]
            rows.extend(np.asarray(self._tokens[index], dtype=np.int64) for index in indices)
            self.position += take
        import torch

        input_ids = torch.from_numpy(np.stack(rows, axis=0))
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }


def _distributed_initialized(torch: Any) -> bool:
    distributed = getattr(torch, "distributed", None)
    return bool(
        distributed is not None
        and distributed.is_available()
        and distributed.is_initialized()
    )


def _batch_counts(batch: Mapping[str, Any], torch: Any) -> tuple[int, int]:
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.numel() == 0:
        raise LearnerError("input_ids must be a non-empty rank-2 tensor")
    if not isinstance(labels, torch.Tensor) or labels.shape != input_ids.shape:
        raise LearnerError("labels must be a tensor matching input_ids")
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        active = torch.ones_like(input_ids, dtype=torch.bool)
    elif isinstance(attention_mask, torch.Tensor) and attention_mask.shape == input_ids.shape:
        active = attention_mask.to(dtype=torch.bool)
    else:
        raise LearnerError("attention_mask must match input_ids when supplied")
    processed = int(active.sum().item())
    target_active = active[:, 1:] & labels[:, 1:].ne(-100)
    targets = int(target_active.sum().item())
    if processed <= 0 or targets <= 0:
        raise LearnerError("batch has no processed input or loss-bearing target tokens")
    return processed, targets


def _extract_loss(output: Any, torch: Any) -> Any:
    loss = getattr(output, "loss", None)
    if loss is None and isinstance(output, Mapping):
        loss = output.get("loss")
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise LearnerError("model output must contain a scalar loss tensor")
    if not bool(torch.isfinite(loss.detach()).item()):
        raise LearnerError("model loss is NaN or Inf")
    return loss


class LearnerRuntime:
    def __init__(
        self,
        *,
        model: Any,
        optimizer: Any,
        scheduler: StepScheduler,
        progress: LearnerProgress,
        fragment_parameters: Sequence[Sequence[Any]],
        device: Any,
        precision: str,
        gradient_accumulation_steps: int,
        max_grad_norm: float,
        comparison: Mapping[str, Any],
        logger: StructuredLogger | None = None,
        clock_ns: Any = time.monotonic_ns,
    ) -> None:
        import torch

        if _distributed_initialized(torch):
            raise LearnerError("torch.distributed must remain uninitialized")
        if len(fragment_parameters) != len(progress.fragments):
            raise LearnerError("fragment parameter groups must match fragment progress")
        if any(not group for group in fragment_parameters):
            raise LearnerError("every fragment parameter group must be non-empty")
        flattened = [parameter for group in fragment_parameters for parameter in group]
        if not flattened or len({id(parameter) for parameter in flattened}) != len(flattened):
            raise LearnerError("fragment parameter groups must uniquely cover trainable parameters")
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if {id(parameter) for parameter in flattened} != {id(parameter) for parameter in model_parameters}:
            raise LearnerError("fragment parameter groups must cover every trainable model parameter")
        optimizer_parameters = [
            parameter for group in optimizer.param_groups for parameter in group["params"]
        ]
        if (
            len(optimizer_parameters) != len(model_parameters)
            or len({id(parameter) for parameter in optimizer_parameters}) != len(optimizer_parameters)
            or {id(parameter) for parameter in optimizer_parameters}
            != {id(parameter) for parameter in model_parameters}
        ):
            raise LearnerError("optimizer parameters must exactly match the local model")
        if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps <= 0:
            raise LearnerError("gradient_accumulation_steps must be positive")
        if not isinstance(max_grad_norm, (int, float)) or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise LearnerError("max_grad_norm must be finite and positive")
        if precision not in {"fp32", "bf16"}:
            raise LearnerError("precision must be fp32 or bf16")
        if precision == "bf16" and getattr(device, "type", None) == "cuda" and not torch.cuda.is_bf16_supported():
            raise LearnerError("configured bf16 compute is unsupported; runtime fallback is forbidden")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.progress = progress
        self.fragment_parameters = tuple(tuple(group) for group in fragment_parameters)
        self.device = device
        self.precision = precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = float(max_grad_norm)
        self.comparison = dict(comparison)
        self.logger = logger
        self.clock_ns = clock_ns

    def _autocast(self, torch: Any) -> Any:
        enabled = self.precision == "bf16"
        device_type = getattr(self.device, "type", str(self.device))
        if device_type not in {"cpu", "cuda"}:
            raise LearnerError(f"unsupported autocast device: {device_type}")
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=enabled)

    def _move_batch(self, batch: Mapping[str, Any], torch: Any) -> dict[str, Any]:
        allowed = {"input_ids", "labels", "attention_mask", "position_ids", "token_type_ids"}
        if not batch:
            raise LearnerError("empty batch is invalid")
        unknown = set(batch) - allowed
        if unknown:
            raise LearnerError(f"unsupported batch fields: {sorted(unknown)}")
        moved = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                raise LearnerError(f"batch field {key} must be a tensor")
            moved[key] = value.to(self.device, non_blocking=False)
        return moved

    def run(self, batches: Iterable[Mapping[str, Any]], *, optimizer_steps: int) -> LearnerRunSummary:
        import torch

        if not isinstance(optimizer_steps, int) or isinstance(optimizer_steps, bool) or optimizer_steps <= 0:
            raise LearnerError("optimizer_steps must be positive")
        if _distributed_initialized(torch):
            raise LearnerError("torch.distributed must remain uninitialized")
        iterator = iter(batches)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        training_start = self.clock_ns()
        events: list[SafeBoundaryEvent] = []
        run_loss_numerator = 0.0
        run_target_tokens = 0
        run_input_tokens = 0
        total_update_squared = 0.0
        initial_step = self.progress.local_optimizer_steps

        for _ in range(optimizer_steps):
            step_start = self.clock_ns()
            before = [
                [parameter.detach().clone() for parameter in group]
                for group in self.fragment_parameters
            ]
            step_loss_numerator = 0.0
            step_targets = 0
            step_inputs = 0
            try:
                for _microbatch in range(self.gradient_accumulation_steps):
                    try:
                        raw_batch = next(iterator)
                    except StopIteration as error:
                        raise LearnerError("data source exhausted before requested optimizer steps") from error
                    batch = self._move_batch(raw_batch, torch)
                    processed, targets = _batch_counts(batch, torch)
                    with self._autocast(torch):
                        output = self.model(**batch)
                        loss = _extract_loss(output, torch)
                    # HF causal-LM losses are means over valid shifted targets.
                    # Accumulate exact loss sums, then normalize once across
                    # all microbatches so padding cannot bias the update.
                    (loss * targets).backward()
                    step_loss_numerator += float(loss.detach().float().item()) * targets
                    step_targets += targets
                    step_inputs += processed
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(step_targets)
                try:
                    gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as error:
                    raise LearnerError("gradient norm is NaN Inf or overflowed") from error
                gradient_norm = float(gradient_norm_tensor.detach().float().item())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            except Exception:
                self.optimizer.zero_grad(set_to_none=True)
                raise

            if getattr(self.device, "type", None) == "cuda":
                torch.cuda.synchronize(self.device)
            fragment_norms: list[float] = []
            for group, originals in zip(self.fragment_parameters, before, strict=True):
                squared = 0.0
                for parameter, original in zip(group, originals, strict=True):
                    delta = parameter.detach().float() - original.float()
                    squared += float(torch.sum(delta * delta).item())
                if not math.isfinite(squared) or squared < 0:
                    raise LearnerError("fragment update norm is nonfinite")
                fragment_norms.append(math.sqrt(squared))
                total_update_squared += squared

            self.progress.local_optimizer_steps += 1
            self.progress.processed_input_tokens += step_inputs
            self.progress.loss_bearing_target_tokens += step_targets
            for fragment in self.progress.fragments:
                fragment.local_steps_since_adoption += 1
                fragment.processed_input_tokens_since_adoption += step_inputs
            boundary = self.clock_ns()
            latency_seconds = (boundary - step_start) / 1_000_000_000
            if latency_seconds <= 0:
                raise LearnerError("optimizer-step observation interval must be positive")
            event = SafeBoundaryEvent(
                learner_id=self.progress.learner_id,
                local_optimizer_step=self.progress.local_optimizer_steps,
                microbatches=self.gradient_accumulation_steps,
                processed_input_tokens_step=step_inputs,
                processed_input_tokens_total=self.progress.processed_input_tokens,
                loss_bearing_target_tokens_step=step_targets,
                loss_bearing_target_tokens_total=self.progress.loss_bearing_target_tokens,
                token_weighted_loss=step_loss_numerator / step_targets,
                loss_numerator=step_loss_numerator,
                gradient_norm=gradient_norm,
                learning_rates=tuple(float(group["lr"]) for group in self.optimizer.param_groups),
                lr_schedule_basis="local_optimizer_steps",
                step_start_monotonic_ns=step_start,
                safe_boundary_monotonic_ns=boundary,
                step_latency_seconds=latency_seconds,
                step_input_tokens_per_second=step_inputs / latency_seconds,
                fragment_global_versions=tuple(fragment.global_version for fragment in self.progress.fragments),
                fragment_local_steps=tuple(fragment.local_steps_since_adoption for fragment in self.progress.fragments),
                fragment_processed_input_tokens=tuple(
                    fragment.processed_input_tokens_since_adoption for fragment in self.progress.fragments
                ),
                fragment_update_norms=tuple(fragment_norms),
                distributed_initialized=_distributed_initialized(torch),
                comparison=self.comparison,
                inactive_metrics={
                    "gpu_to_cpu_seconds": "not_exercised_until_S1-07",
                    "cpu_to_fs_seconds": "not_exercised_until_S1-07",
                    "fs_to_cpu_seconds": "not_exercised_until_S1-08",
                    "cpu_to_gpu_seconds": "not_exercised_until_S1-08",
                    "snapshot_skip_count": "not_exercised_until_S1-07",
                    "snapshot_replacement_count": "not_exercised_until_S1-07",
                    "pending_upload_count": "not_exercised_until_S1-07",
                    "fragment_version_lag": "not_exercised_until_S1-08",
                    "extra_local_steps_before_adoption": "not_exercised_until_S1-08",
                },
            )
            if event.distributed_initialized:
                raise LearnerError("torch.distributed became initialized during learner execution")
            events.append(event)
            if self.logger is not None:
                self.logger.emit("safe_boundary", **event.to_dict())
            run_loss_numerator += step_loss_numerator
            run_target_tokens += step_targets
            run_input_tokens += step_inputs

        training_end = events[-1].safe_boundary_monotonic_ns
        interval_seconds = (training_end - training_start) / 1_000_000_000
        if interval_seconds <= 0:
            raise LearnerError("training observation interval must be positive")
        if not math.isfinite(total_update_squared) or total_update_squared <= 0:
            raise LearnerError("requested training run changed no model parameter")
        if self.scheduler.step_count != self.progress.local_optimizer_steps:
            raise LearnerError("scheduler and learner local optimizer steps diverged")
        if _distributed_initialized(torch):
            raise LearnerError("torch.distributed became initialized during learner execution")
        return LearnerRunSummary(
            learner_id=self.progress.learner_id,
            training_start_monotonic_ns=training_start,
            training_end_monotonic_ns=training_end,
            optimizer_steps_completed=self.progress.local_optimizer_steps - initial_step,
            processed_input_tokens=run_input_tokens,
            loss_bearing_target_tokens=run_target_tokens,
            token_weighted_loss=run_loss_numerator / run_target_tokens,
            common_interval_input_tokens_per_second=run_input_tokens / interval_seconds,
            parameter_update_norm=math.sqrt(total_update_squared),
            parameters_changed=True,
            distributed_initialized=False,
            progress=self.progress.to_dict(),
            events=tuple(events),
        )
