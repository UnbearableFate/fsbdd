from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import math
import os
import struct
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .global_state import FragmentStateDescriptor
from .identity import canonical_digest


class MergeError(RuntimeError):
    """A fragment merge input or transition violates the frozen contract."""


_HEX = frozenset("0123456789abcdef")
_F32_BYTES = 4


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MergeError(f"{name} must be a non-empty string")
    return value


def _require_hex(value: object, name: str) -> str:
    value = _require_text(value, name)
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise MergeError(f"{name} must be a lowercase SHA-256 identity")
    return value


def _require_nonnegative(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MergeError(f"{name} must be a non-negative integer")
    return value


def _require_positive(value: object, name: str) -> int:
    value = _require_nonnegative(value, name)
    if value == 0:
        raise MergeError(f"{name} must be positive")
    return value


def _f32(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MergeError(f"{name} must be numeric")
    try:
        result = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, ValueError) as error:
        raise MergeError(f"{name} is not representable as float32") from error
    if not math.isfinite(result):
        raise MergeError(f"{name} must be finite")
    return result


def _require_fp32_payload(payload: object, name: str) -> bytes:
    if not isinstance(payload, bytes):
        raise MergeError(f"{name} must be immutable bytes")
    if not payload or len(payload) % _F32_BYTES:
        raise MergeError(f"{name} must be a non-empty little-endian fp32 payload")
    return payload


@dataclasses.dataclass(frozen=True, slots=True)
class ContributionFact:
    proposal_id: str
    content_identity: str
    learner_id: str
    base_version: int
    base_content_identity: str
    processed_tokens: int
    staleness: int
    normalized_weight: float
    parameters_sha256: str
    payload_bytes: int

    def __post_init__(self) -> None:
        _require_text(self.proposal_id, "proposal_id")
        _require_hex(self.content_identity, "proposal content identity")
        _require_text(self.learner_id, "learner_id")
        _require_nonnegative(self.base_version, "base_version")
        _require_hex(self.base_content_identity, "base content identity")
        _require_positive(self.processed_tokens, "processed_tokens")
        _require_nonnegative(self.staleness, "staleness")
        weight = _f32(self.normalized_weight, "normalized_weight")
        if weight < 0:
            raise MergeError("normalized_weight must be non-negative")
        _require_hex(self.parameters_sha256, "parameters_sha256")
        payload_bytes = _require_positive(self.payload_bytes, "payload_bytes")
        if payload_bytes % _F32_BYTES:
            raise MergeError("payload_bytes must be divisible by four")

    @property
    def f32_weight(self) -> float:
        return _f32(self.normalized_weight, "normalized_weight")

    def identity_facts(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "content_identity": self.content_identity,
            "learner_id": self.learner_id,
            "base_version": self.base_version,
            "base_content_identity": self.base_content_identity,
            "processed_tokens": self.processed_tokens,
            "staleness": self.staleness,
            "normalized_weight": self.f32_weight,
            "parameters_sha256": self.parameters_sha256,
            "payload_bytes": self.payload_bytes,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentOuterState:
    update_count: int
    momentum_buffer: bytes | None = None

    def __post_init__(self) -> None:
        _require_nonnegative(self.update_count, "outer update_count")
        if self.momentum_buffer is not None:
            _require_fp32_payload(self.momentum_buffer, "momentum_buffer")

    def identity_facts(self) -> dict[str, object]:
        return {
            "update_count": self.update_count,
            "momentum_buffer_sha256": (
                None
                if self.momentum_buffer is None
                else hashlib.sha256(self.momentum_buffer).hexdigest()
            ),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class OuterSGDPolicy:
    learning_rate: float
    momentum: float
    nesterov: bool
    name: str = "sgd"

    def __post_init__(self) -> None:
        _require_text(self.name, "outer optimizer name")
        learning_rate = _f32(self.learning_rate, "learning_rate")
        momentum = _f32(self.momentum, "momentum")
        if learning_rate <= 0:
            raise MergeError("learning_rate must be positive")
        if not 0 <= momentum < 1:
            raise MergeError("momentum must be in [0, 1)")
        if not isinstance(self.nesterov, bool):
            raise MergeError("nesterov must be boolean")
        if self.nesterov and momentum == 0:
            raise MergeError("Nesterov requires positive momentum")

    @property
    def f32_learning_rate(self) -> float:
        return _f32(self.learning_rate, "learning_rate")

    @property
    def f32_momentum(self) -> float:
        return _f32(self.momentum, "momentum")

    def identity_facts(self) -> dict[str, object]:
        return {
            "name": self.name,
            "learning_rate": self.f32_learning_rate,
            "momentum": self.f32_momentum,
            "nesterov": self.nesterov,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentMergeRequest:
    descriptor: FragmentStateDescriptor
    fragment_map_identity: str
    current_version: int
    current_content_identity: str
    current_parameters: bytes
    outer_state: FragmentOuterState
    selection_identity: str
    contributions: tuple[ContributionFact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, FragmentStateDescriptor):
            raise MergeError("descriptor must be FragmentStateDescriptor")
        if self.descriptor.dtype != "float32":
            raise MergeError("streaming merge requires a float32 fragment descriptor")
        _require_hex(self.fragment_map_identity, "fragment_map_identity")
        _require_nonnegative(self.current_version, "current_version")
        _require_hex(self.current_content_identity, "current_content_identity")
        current = _require_fp32_payload(self.current_parameters, "current_parameters")
        if (
            len(self.descriptor.shape) != 1
            or self.descriptor.shape[0] <= 0
            or self.descriptor.shape[0] * _F32_BYTES != len(current)
        ):
            raise MergeError(
                "flat fragment descriptor shape differs from current fp32 payload"
            )
        if not isinstance(self.outer_state, FragmentOuterState):
            raise MergeError("outer_state must be FragmentOuterState")
        if self.outer_state.update_count != self.current_version:
            raise MergeError("outer state update_count must equal current version")
        _require_hex(self.selection_identity, "selection_identity")
        if not isinstance(self.contributions, tuple) or not self.contributions:
            raise MergeError("contributions must be a non-empty tuple")
        proposal_ids: set[str] = set()
        learner_ids: set[str] = set()
        weight_sum = _f32(0.0, "weight sum")
        for contribution in self.contributions:
            if not isinstance(contribution, ContributionFact):
                raise MergeError("contributions must contain ContributionFact values")
            if contribution.proposal_id in proposal_ids:
                raise MergeError("selected contributions contain a duplicate proposal")
            if contribution.learner_id in learner_ids:
                raise MergeError("selected contributions contain a duplicate learner")
            proposal_ids.add(contribution.proposal_id)
            learner_ids.add(contribution.learner_id)
            if contribution.payload_bytes != len(current):
                raise MergeError("contribution payload shape differs from current fragment")
            if self.current_version - contribution.base_version != contribution.staleness:
                raise MergeError("contribution staleness differs from current minus base")
            weight_sum = _f32(weight_sum + contribution.f32_weight, "weight sum")
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=2e-6):
            raise MergeError("float32 normalized weights must sum to one")
        if (
            self.outer_state.momentum_buffer is not None
            and len(self.outer_state.momentum_buffer) != len(current)
        ):
            raise MergeError("momentum buffer shape differs from current fragment")


class StreamingContributionSource(Protocol):
    @contextlib.contextmanager
    def open_payload(self, contribution: ContributionFact) -> Iterator[bytes]: ...


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedBase:
    version: int
    content_identity: str
    parameters: bytes
    parameters_sha256: str

    def __post_init__(self) -> None:
        _require_nonnegative(self.version, "resolved base version")
        _require_hex(self.content_identity, "resolved base identity")
        parameters = _require_fp32_payload(self.parameters, "resolved base parameters")
        _require_hex(self.parameters_sha256, "resolved base parameters_sha256")
        if hashlib.sha256(parameters).hexdigest() != self.parameters_sha256:
            raise MergeError("resolved base parameter checksum mismatch")


class StreamingBaseSource(Protocol):
    @contextlib.contextmanager
    def open_base(self, contribution: ContributionFact) -> Iterator[ResolvedBase]: ...


@dataclasses.dataclass(frozen=True, slots=True)
class PayloadLocation:
    identity: str
    relative_path: str

    def __post_init__(self) -> None:
        _require_text(self.identity, "payload location identity")
        relative = Path(_require_text(self.relative_path, "payload relative_path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise MergeError("payload location must remain below its frozen root")


@dataclasses.dataclass(frozen=True, slots=True)
class SourceMetrics:
    opens: int
    bytes_read: int
    active_payloads: int
    maximum_active_payloads: int


class ImmutableFileContributionSource:
    """Read immutable selected payloads one at a time from a frozen root."""

    def __init__(self, root: Path, locations: Sequence[PayloadLocation]) -> None:
        self.root = Path(root).resolve()
        indexed: dict[str, Path] = {}
        for location in locations:
            if not isinstance(location, PayloadLocation):
                raise MergeError("locations must contain PayloadLocation values")
            if location.identity in indexed:
                raise MergeError(f"duplicate payload location: {location.identity}")
            indexed[location.identity] = self.root / location.relative_path
        self._locations = indexed
        self._opens = 0
        self._bytes_read = 0
        self._active = 0
        self._maximum_active = 0

    @contextlib.contextmanager
    def open_payload(self, contribution: ContributionFact) -> Iterator[bytes]:
        try:
            path = self._locations[contribution.proposal_id]
        except KeyError as error:
            raise MergeError(
                f"selected proposal has no immutable payload location: {contribution.proposal_id}"
            ) from error
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise MergeError(f"cannot read selected payload: {path}") from error
        self._opens += 1
        self._bytes_read += len(payload)
        self._active += 1
        self._maximum_active = max(self._maximum_active, self._active)
        try:
            yield payload
        finally:
            self._active -= 1

    def metrics(self) -> SourceMetrics:
        return SourceMetrics(
            opens=self._opens,
            bytes_read=self._bytes_read,
            active_payloads=self._active,
            maximum_active_payloads=self._maximum_active,
        )


class ImmutableFileBaseSource:
    def __init__(
        self,
        root: Path,
        locations: Sequence[PayloadLocation],
        bases: Mapping[str, tuple[int, str, str]],
    ) -> None:
        self._files = ImmutableFileContributionSource(root, locations)
        self._bases = dict(bases)

    @contextlib.contextmanager
    def open_base(self, contribution: ContributionFact) -> Iterator[ResolvedBase]:
        try:
            version, content_identity, parameters_sha256 = self._bases[
                contribution.base_content_identity
            ]
        except KeyError as error:
            raise MergeError("declared base is absent from the retained base source") from error
        placeholder = ContributionFact(
            proposal_id=contribution.base_content_identity,
            content_identity=contribution.base_content_identity,
            learner_id=contribution.learner_id,
            base_version=version,
            base_content_identity=content_identity,
            processed_tokens=contribution.processed_tokens,
            staleness=contribution.staleness,
            normalized_weight=contribution.normalized_weight,
            parameters_sha256=contribution.parameters_sha256,
            payload_bytes=contribution.payload_bytes,
        )
        with self._files.open_payload(placeholder) as payload:
            yield ResolvedBase(version, content_identity, payload, parameters_sha256)

    def metrics(self) -> SourceMetrics:
        return self._files.metrics()


class MergePolicy(ABC):
    name: str

    @abstractmethod
    def identity_facts(self) -> Mapping[str, object]:
        raise NotImplementedError

    @abstractmethod
    def accumulate(self, accumulator: Any, base: Any, local: Any, weight: float) -> None:
        raise NotImplementedError


class DirectWeightedAverage(MergePolicy):
    name = "direct_weighted_average"

    def identity_facts(self) -> Mapping[str, object]:
        return {"name": self.name, "formula": "sum(weight * (declared_base - local))"}

    def accumulate(self, accumulator: Any, base: Any, local: Any, weight: float) -> None:
        # Reuse the one live local tensor for its weighted pseudo-gradient so
        # reduction needs no hidden M-sized collection or extra fragment tensor.
        local.neg_().add_(base).mul_(_f32(weight, "normalized_weight"))
        accumulator.add_(local)


@dataclasses.dataclass(frozen=True, slots=True)
class ByteAccounting:
    fragment_index: int
    fragment_bytes: int
    current_input_bytes: int
    local_payload_reads: int
    local_payload_bytes: int
    retained_base_reads: int
    retained_base_bytes: int
    successor_output_bytes: int
    full_model_operations: int = 0

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryAccounting:
    maximum_active_local_payloads: int
    maximum_active_base_payloads: int
    maximum_live_tensor_bytes: int
    fragment_bytes: int
    process_rss_at_entry_bytes: int | None = None
    maximum_observed_process_rss_bytes: int | None = None
    peak_process_rss_bytes: int | None = None

    @property
    def maximum_tensor_fragment_multiples(self) -> float:
        return self.maximum_live_tensor_bytes / self.fragment_bytes

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            **dataclasses.asdict(self),
            "maximum_tensor_fragment_multiples": self.maximum_tensor_fragment_multiples,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentUpdateResult:
    parameters: bytes
    outer_state: FragmentOuterState
    merged_gradient: bytes
    update_identity: str
    update_facts: Mapping[str, object]
    byte_accounting: ByteAccounting
    memory_accounting: MemoryAccounting

    def __post_init__(self) -> None:
        _require_fp32_payload(self.parameters, "result parameters")
        _require_fp32_payload(self.merged_gradient, "merged gradient")
        _require_hex(self.update_identity, "update_identity")


class _TensorLedger:
    def __init__(self) -> None:
        self.live = 0
        self.maximum = 0

    def add(self, size: int) -> None:
        self.live += size
        self.maximum = max(self.maximum, self.live)

    def remove(self, size: int) -> None:
        self.live -= size
        if self.live < 0:
            raise AssertionError("tensor memory ledger underflow")


def _tensor_from_payload(payload: bytes, name: str) -> Any:
    import numpy as np
    import torch

    payload = _require_fp32_payload(payload, name)
    array = np.frombuffer(payload, dtype="<f4").copy()
    tensor = torch.from_numpy(array)
    if not bool(torch.isfinite(tensor).all()):
        raise MergeError(f"{name} contains NaN or Inf")
    return tensor


def _payload_from_tensor(tensor: Any, name: str) -> bytes:
    import torch

    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise MergeError(f"{name} must be a CPU float32 tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise MergeError(f"{name} contains NaN or Inf")
    return tensor.detach().contiguous().numpy().astype("<f4", copy=False).tobytes()


def build_update_facts(
    request: FragmentMergeRequest,
    merge_policy: MergePolicy,
    outer_policy: OuterSGDPolicy,
) -> dict[str, object]:
    if not isinstance(merge_policy, MergePolicy):
        raise MergeError("merge_policy must implement MergePolicy")
    if not isinstance(outer_policy, OuterSGDPolicy):
        raise MergeError("outer_policy must be OuterSGDPolicy")
    return {
        "schema_version": 1,
        "current_version": request.current_version,
        "current_content_identity": request.current_content_identity,
        "current_parameters_sha256": hashlib.sha256(request.current_parameters).hexdigest(),
        "current_outer_state": request.outer_state.identity_facts(),
        "selection_identity": request.selection_identity,
        "ordered_proposal_identities": [item.proposal_id for item in request.contributions],
        "ordered_proposal_content_identities": [
            item.content_identity for item in request.contributions
        ],
        "ordered_base_versions": [item.base_version for item in request.contributions],
        "ordered_base_content_identities": [
            item.base_content_identity for item in request.contributions
        ],
        "ordered_processed_tokens": [
            item.processed_tokens for item in request.contributions
        ],
        "ordered_staleness": [item.staleness for item in request.contributions],
        "ordered_float32_weights": [item.f32_weight for item in request.contributions],
        "ordered_local_parameters_sha256": [
            item.parameters_sha256 for item in request.contributions
        ],
        "merge_policy": dict(merge_policy.identity_facts()),
        "outer_optimizer_policy": outer_policy.name,
        "outer_hyperparameters": {
            "learning_rate": outer_policy.f32_learning_rate,
            "momentum": outer_policy.f32_momentum,
            "nesterov": outer_policy.nesterov,
        },
        "accumulation_dtype": "float32",
        "fragment_map_identity": request.fragment_map_identity,
        "fragment_identity": request.descriptor.identity,
        "fragment_index": request.descriptor.index,
    }


def execute_streaming_fragment_update(
    request: FragmentMergeRequest,
    contribution_source: StreamingContributionSource,
    outer_policy: OuterSGDPolicy,
    *,
    base_source: StreamingBaseSource | None = None,
    merge_policy: MergePolicy | None = None,
    measure_process_rss: bool = False,
) -> FragmentUpdateResult:
    """Stream one selected fragment update through CPU float32 tensors.

    The function never asks for a model or proposal collection.  Each local
    payload lease ends before the next begins, while a noncurrent retained base
    is resolved only for the contribution that declares it.
    """

    import torch

    if not isinstance(request, FragmentMergeRequest):
        raise MergeError("request must be FragmentMergeRequest")
    if not isinstance(outer_policy, OuterSGDPolicy):
        raise MergeError("outer_policy must be OuterSGDPolicy")
    policy = DirectWeightedAverage() if merge_policy is None else merge_policy
    if not isinstance(policy, MergePolicy):
        raise MergeError("merge_policy must implement MergePolicy")
    if outer_policy.f32_momentum == 0 and request.outer_state.momentum_buffer is not None:
        raise MergeError("momentum-free SGD state must not contain a buffer")

    fragment_bytes = len(request.current_parameters)
    ledger = _TensorLedger()
    rss_at_entry: int | None = None
    maximum_rss: int | None = None

    def observe_rss() -> None:
        nonlocal maximum_rss
        if measure_process_rss:
            current_rss, _ = linux_process_memory_bytes()
            maximum_rss = current_rss if maximum_rss is None else max(maximum_rss, current_rss)

    if measure_process_rss:
        rss_at_entry, _ = linux_process_memory_bytes()
        maximum_rss = rss_at_entry
    current = _tensor_from_payload(request.current_parameters, "current_parameters")
    ledger.add(fragment_bytes)
    accumulator = torch.zeros_like(current)
    ledger.add(fragment_bytes)
    observe_rss()
    local_bytes_read = 0
    base_bytes_read = 0
    base_reads = 0
    active_local = 0
    maximum_active_local = 0
    active_base = 0
    maximum_active_base = 0

    for contribution in request.contributions:
        with contribution_source.open_payload(contribution) as local_payload:
            active_local += 1
            maximum_active_local = max(maximum_active_local, active_local)
            try:
                observe_rss()
                local_payload = _require_fp32_payload(local_payload, "local payload")
                if len(local_payload) != fragment_bytes:
                    raise MergeError("local contribution shape differs from current fragment")
                if hashlib.sha256(local_payload).hexdigest() != contribution.parameters_sha256:
                    raise MergeError("local contribution checksum mismatch")
                local_bytes_read += len(local_payload)
                local = _tensor_from_payload(local_payload, "local payload")
                ledger.add(fragment_bytes)
                observe_rss()
                try:
                    if contribution.base_version == request.current_version:
                        if contribution.base_content_identity != request.current_content_identity:
                            raise MergeError("current-base contribution identity mismatch")
                        base = current
                        policy.accumulate(
                            accumulator, base, local, contribution.f32_weight
                        )
                    else:
                        if base_source is None:
                            raise MergeError("noncurrent contribution requires a retained base source")
                        with base_source.open_base(contribution) as resolved:
                            active_base += 1
                            maximum_active_base = max(maximum_active_base, active_base)
                            try:
                                if (
                                    resolved.version != contribution.base_version
                                    or resolved.content_identity
                                    != contribution.base_content_identity
                                ):
                                    raise MergeError("resolved base identity differs from proposal")
                                if len(resolved.parameters) != fragment_bytes:
                                    raise MergeError("resolved base shape differs from current fragment")
                                base_bytes_read += len(resolved.parameters)
                                base_reads += 1
                                base = _tensor_from_payload(
                                    resolved.parameters, "resolved base parameters"
                                )
                                ledger.add(fragment_bytes)
                                observe_rss()
                                try:
                                    policy.accumulate(
                                        accumulator,
                                        base,
                                        local,
                                        contribution.f32_weight,
                                    )
                                finally:
                                    ledger.remove(fragment_bytes)
                                    del base
                            finally:
                                active_base -= 1
                finally:
                    ledger.remove(fragment_bytes)
                    del local
            finally:
                active_local -= 1

    if active_local or active_base:
        raise AssertionError("streaming payload lease leaked")
    if not bool(torch.isfinite(accumulator).all()):
        raise MergeError("merged gradient contains NaN or Inf")
    merged_gradient = _payload_from_tensor(accumulator, "merged gradient")

    momentum = outer_policy.f32_momentum
    learning_rate = outer_policy.f32_learning_rate
    buffer = None
    if momentum == 0:
        direction = accumulator
    else:
        if request.outer_state.momentum_buffer is None:
            buffer = accumulator.clone()
        else:
            buffer = _tensor_from_payload(
                request.outer_state.momentum_buffer, "momentum_buffer"
            )
            buffer.mul_(momentum).add_(accumulator)
        ledger.add(fragment_bytes)
        if outer_policy.nesterov:
            direction = accumulator.add(buffer, alpha=momentum)
            ledger.add(fragment_bytes)
        else:
            direction = buffer
    successor = current.add(direction, alpha=-learning_rate)
    ledger.add(fragment_bytes)
    observe_rss()
    successor_payload = _payload_from_tensor(successor, "successor parameters")
    next_buffer_payload = (
        None if buffer is None else _payload_from_tensor(buffer, "momentum_buffer")
    )

    update_facts = build_update_facts(request, policy, outer_policy)
    update_identity = canonical_digest(update_facts)
    result = FragmentUpdateResult(
        parameters=successor_payload,
        outer_state=FragmentOuterState(
            update_count=request.outer_state.update_count + 1,
            momentum_buffer=next_buffer_payload,
        ),
        merged_gradient=merged_gradient,
        update_identity=update_identity,
        update_facts=update_facts,
        byte_accounting=ByteAccounting(
            fragment_index=request.descriptor.index,
            fragment_bytes=fragment_bytes,
            current_input_bytes=fragment_bytes,
            local_payload_reads=len(request.contributions),
            local_payload_bytes=local_bytes_read,
            retained_base_reads=base_reads,
            retained_base_bytes=base_bytes_read,
            successor_output_bytes=fragment_bytes,
        ),
        memory_accounting=MemoryAccounting(
            maximum_active_local_payloads=maximum_active_local,
            maximum_active_base_payloads=maximum_active_base,
            maximum_live_tensor_bytes=ledger.maximum,
            fragment_bytes=fragment_bytes,
            process_rss_at_entry_bytes=rss_at_entry,
            maximum_observed_process_rss_bytes=maximum_rss,
            peak_process_rss_bytes=(
                None
                if rss_at_entry is None or maximum_rss is None
                else maximum_rss - rss_at_entry
            ),
        ),
    )
    return result


def linux_process_memory_bytes() -> tuple[int, int]:
    """Return current RSS and high-water RSS from Linux procfs."""

    status = Path(f"/proc/{os.getpid()}/status").read_text(encoding="utf-8")
    values: dict[str, int] = {}
    for line in status.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, raw, unit = line.split()
            if unit != "kB":
                raise MergeError(f"unexpected procfs memory unit: {unit}")
            values[name.removesuffix(":")] = int(raw) * 1024
    try:
        return values["VmRSS"], values["VmHWM"]
    except KeyError as error:
        raise MergeError("Linux procfs did not expose VmRSS and VmHWM") from error
