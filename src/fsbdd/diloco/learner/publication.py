from __future__ import annotations

import dataclasses
import hashlib
import io
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

from fsbdd.diloco.model.fragment_map import FragmentMap, validate_fragment_map
from fsbdd.diloco.protocol.global_state import (
    FragmentStateDescriptor,
    GlobalStateIdentities,
)
from fsbdd.diloco.common.identity import canonical_digest
from fsbdd.diloco.learner.runtime import LearnerProgress, SafeBoundaryEvent
from fsbdd.diloco.common.logging import StructuredLogger
from fsbdd.diloco.model.model_registry import LogicalLayerRegistry
from fsbdd.diloco.protocol.proposal import Proposal, ProposalStore


class SnapshotPublishError(RuntimeError):
    pass


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SnapshotPublishError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SnapshotPublishError(f"{name} must be a non-negative integer")
    return value


def _hex_identity(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SnapshotPublishError(f"{name} must be a lowercase SHA-256 identity")
    return value


def _positive_float(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SnapshotPublishError(f"{name} must be finite and positive")
    return float(value)


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentPublicationSchedule:
    """Deterministic fragment due schedule and exact byte-rate accounting."""

    fragment_bytes: tuple[int, ...]
    intervals: tuple[int, ...]
    offsets: tuple[int, ...]
    offset_algorithm: str
    learner_phase_offset: int = 0

    def __post_init__(self) -> None:
        if not self.fragment_bytes:
            raise SnapshotPublishError("publication schedule requires fragments")
        if not (len(self.fragment_bytes) == len(self.intervals) == len(self.offsets)):
            raise SnapshotPublishError(
                "fragment bytes intervals and offsets must have equal lengths"
            )
        if any(
            _positive_int(value, "fragment bytes") <= 0 for value in self.fragment_bytes
        ):
            raise AssertionError("unreachable")
        for interval, offset in zip(self.intervals, self.offsets, strict=True):
            _positive_int(interval, "publication interval")
            _nonnegative_int(offset, "publication offset")
            if offset >= interval:
                raise SnapshotPublishError(
                    "publication offset must be below its interval"
                )
        if not isinstance(self.offset_algorithm, str) or not self.offset_algorithm:
            raise SnapshotPublishError("offset_algorithm must be non-empty")
        _nonnegative_int(self.learner_phase_offset, "learner_phase_offset")

    @classmethod
    def unified(
        cls,
        fragment_bytes: Sequence[int],
        h: int,
        *,
        learner_phase_offset: int = 0,
    ) -> FragmentPublicationSchedule:
        """Place byte-weighted midpoints on the nearest circular free offsets.

        A heavier fragment receives more empty time around its midpoint.  When
        H is at least F, nearest-free assignment also guarantees that no two
        fragments share a due step.  Every tie is resolved by the lowest
        integer offset, making the result independent of hash/random state.
        """

        h = _positive_int(h, "H")
        values = tuple(
            _positive_int(value, "fragment bytes") for value in fragment_bytes
        )
        if not values:
            raise SnapshotPublishError("publication schedule requires fragments")
        phase = _nonnegative_int(learner_phase_offset, "learner_phase_offset")
        if phase >= h:
            raise SnapshotPublishError("learner phase offset must be below H")
        total = sum(values)
        denominator = 2 * total
        circle = denominator * h
        cumulative = 0
        offsets: list[int] = []
        used: set[int] = set()
        require_unique = h >= len(values)
        for size in values:
            target_numerator = (2 * cumulative + size) * h
            candidates = [
                candidate
                for candidate in range(h)
                if not require_unique or candidate not in used
            ]

            def objective(candidate: int) -> tuple[int, int]:
                direct = abs(candidate * denominator - target_numerator)
                return min(direct, circle - direct), candidate

            selected = min(candidates, key=objective)
            offsets.append((selected + phase) % h)
            used.add(selected)
            cumulative += size
        return cls(
            fragment_bytes=values,
            intervals=(h,) * len(values),
            offsets=tuple(offsets),
            offset_algorithm="byte_weighted_midpoint_nearest_free_v1",
            learner_phase_offset=phase,
        )

    @property
    def fragment_count(self) -> int:
        return len(self.fragment_bytes)

    @property
    def unified_h(self) -> int | None:
        return self.intervals[0] if len(set(self.intervals)) == 1 else None

    def is_due(self, local_optimizer_step: int, fragment_index: int) -> bool:
        step = _positive_int(local_optimizer_step, "local optimizer step")
        index = _nonnegative_int(fragment_index, "fragment index")
        try:
            interval = self.intervals[index]
            offset = self.offsets[index]
        except IndexError as error:
            raise SnapshotPublishError(f"unknown fragment index: {index}") from error
        return (step - offset) % interval == 0

    def due_fragments(self, local_optimizer_step: int) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(self.fragment_count)
            if self.is_due(local_optimizer_step, index)
        )

    def frequency_budget(self) -> dict[str, Any]:
        terms = tuple(
            Fraction(size, interval)
            for size, interval in zip(self.fragment_bytes, self.intervals, strict=True)
        )
        total = sum(terms, start=Fraction(0, 1))
        value: dict[str, Any] = {
            "formula": "sum_f(fragment_bytes_f / H_f)",
            "bytes_per_local_step": float(total),
            "exact_numerator": total.numerator,
            "exact_denominator": total.denominator,
            "terms": [
                {
                    "fragment_index": index,
                    "fragment_bytes": size,
                    "interval": interval,
                    "exact_numerator": term.numerator,
                    "exact_denominator": term.denominator,
                }
                for index, (size, interval, term) in enumerate(
                    zip(self.fragment_bytes, self.intervals, terms, strict=True)
                )
            ],
        }
        h = self.unified_h
        if h is not None:
            due_bytes = [0] * h
            for size, offset in zip(self.fragment_bytes, self.offsets, strict=True):
                due_step = h if offset == 0 else offset
                due_bytes[due_step - 1] += size
            value.update(
                {
                    "unified_h": h,
                    "unified_formula": "sum(fragment_bytes) / H",
                    "due_bytes_by_step": due_bytes,
                    "maximum_due_bytes": max(due_bytes),
                }
            )
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_bytes": list(self.fragment_bytes),
            "intervals": list(self.intervals),
            "offsets": list(self.offsets),
            "offset_algorithm": self.offset_algorithm,
            "learner_phase_offset": self.learner_phase_offset,
            "frequency_budget": self.frequency_budget(),
        }


def build_fragment_descriptors(
    fragment_map: FragmentMap,
) -> tuple[FragmentStateDescriptor, ...]:
    if not isinstance(fragment_map, FragmentMap):
        raise SnapshotPublishError("fragment_map must be a FragmentMap")
    if fragment_map.digest != canonical_digest(fragment_map._body()):
        raise SnapshotPublishError("fragment map digest mismatch")
    return tuple(
        FragmentStateDescriptor(
            index=fragment.index,
            identity=canonical_digest(
                {
                    "schema_version": 1,
                    "fragment_map_identity": fragment_map.digest,
                    "fragment_index": fragment.index,
                    "parameter_identities": list(fragment.parameter_identities),
                }
            ),
            dtype="float32",
            shape=(fragment.parameter_count,),
            parameter_identities=fragment.parameter_identities,
        )
        for fragment in fragment_map.fragments
    )


def build_fragment_parameter_groups(
    model: Any,
    registry: LogicalLayerRegistry,
    fragment_map: FragmentMap,
) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(registry, LogicalLayerRegistry):
        raise SnapshotPublishError("registry must be a LogicalLayerRegistry")
    if not isinstance(fragment_map, FragmentMap):
        raise SnapshotPublishError("fragment_map must be a FragmentMap")
    try:
        validate_fragment_map(fragment_map, registry)
    except ValueError as error:
        raise SnapshotPublishError(
            "fragment map and registry identities differ"
        ) from error
    try:
        parameters = dict(
            model.named_parameters(recurse=True, remove_duplicate=False)
        )
    except (AttributeError, TypeError) as error:
        raise SnapshotPublishError(
            "model must support named_parameters(remove_duplicate=False)"
        ) from error
    records = {record.identity: record for record in registry.parameters}
    groups: list[tuple[Any, ...]] = []
    for fragment in fragment_map.fragments:
        group = []
        for identity in fragment.parameter_identities:
            try:
                record = records[identity]
                group.append(parameters[record.owner_name])
            except KeyError as error:
                raise SnapshotPublishError(
                    f"registry parameter is absent from model: {identity}"
                ) from error
        groups.append(tuple(group))
    return tuple(groups)


@dataclasses.dataclass(frozen=True, slots=True)
class AdoptedFragmentBase:
    version: int
    content_identity: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.version, "base version")
        _hex_identity(self.content_identity, "base content identity")


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentSnapshot:
    proposal: Proposal
    safe_boundary_monotonic_ns: int
    staging_started_monotonic_ns: int
    staging_completed_monotonic_ns: int
    gpu_to_cpu_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, Proposal):
            raise SnapshotPublishError("snapshot proposal must be a Proposal")
        for name in (
            "safe_boundary_monotonic_ns",
            "staging_started_monotonic_ns",
            "staging_completed_monotonic_ns",
        ):
            _nonnegative_int(getattr(self, name), name)
        if self.staging_completed_monotonic_ns < self.staging_started_monotonic_ns:
            raise SnapshotPublishError("snapshot staging interval is reversed")
        if (
            not isinstance(self.gpu_to_cpu_seconds, (int, float))
            or not math.isfinite(self.gpu_to_cpu_seconds)
            or self.gpu_to_cpu_seconds < 0
        ):
            raise SnapshotPublishError(
                "gpu_to_cpu_seconds must be finite and nonnegative"
            )

    @property
    def identities(self) -> GlobalStateIdentities:
        return self.proposal.identities

    @property
    def learner_id(self) -> str:
        return self.proposal.learner_id

    @property
    def descriptor(self) -> FragmentStateDescriptor:
        return self.proposal.descriptor

    @property
    def sequence(self) -> int:
        return self.proposal.sequence

    @property
    def proposal_id(self) -> str:
        return self.proposal.proposal_id

    @property
    def base_version(self) -> int:
        return self.proposal.base_version

    @property
    def base_content_identity(self) -> str:
        return self.proposal.base_content_identity

    @property
    def local_steps(self) -> int:
        return self.proposal.local_steps

    @property
    def processed_tokens(self) -> int:
        return self.proposal.processed_tokens

    @property
    def snapshot_local_step(self) -> int:
        return self.proposal.snapshot_local_step

    @property
    def parameters(self) -> bytes:
        return self.proposal.parameters


@dataclasses.dataclass(frozen=True, slots=True)
class StagedFragmentSnapshot:
    """Immutable CPU snapshot whose content hashes are built by the publisher."""

    identities: GlobalStateIdentities
    learner_id: str
    descriptor: FragmentStateDescriptor
    sequence: int
    base_version: int
    base_content_identity: str
    local_steps: int
    processed_tokens: int
    snapshot_local_step: int
    parameters: bytes
    safe_boundary_monotonic_ns: int
    staging_started_monotonic_ns: int
    staging_completed_monotonic_ns: int
    gpu_to_cpu_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.identities, GlobalStateIdentities):
            raise SnapshotPublishError("staged snapshot identities are invalid")
        if not isinstance(self.learner_id, str) or not self.learner_id:
            raise SnapshotPublishError("staged snapshot learner_id is invalid")
        if not isinstance(self.descriptor, FragmentStateDescriptor):
            raise SnapshotPublishError("staged snapshot descriptor is invalid")
        if self.descriptor.dtype != "float32" or len(self.descriptor.shape) != 1:
            raise SnapshotPublishError(
                "staged snapshot descriptor must be flat float32"
            )
        _nonnegative_int(self.sequence, "snapshot sequence")
        _nonnegative_int(self.base_version, "snapshot base version")
        _hex_identity(self.base_content_identity, "snapshot base content identity")
        _nonnegative_int(self.local_steps, "snapshot local steps")
        _nonnegative_int(self.processed_tokens, "snapshot processed tokens")
        _nonnegative_int(self.snapshot_local_step, "snapshot local step")
        if not isinstance(self.parameters, bytes):
            raise SnapshotPublishError("staged snapshot parameters must be bytes")
        if len(self.parameters) != self.descriptor.shape[0] * 4:
            raise SnapshotPublishError(
                "staged snapshot payload differs from its descriptor"
            )
        for name in (
            "safe_boundary_monotonic_ns",
            "staging_started_monotonic_ns",
            "staging_completed_monotonic_ns",
        ):
            _nonnegative_int(getattr(self, name), name)
        if self.staging_completed_monotonic_ns < self.staging_started_monotonic_ns:
            raise SnapshotPublishError("snapshot staging interval is reversed")
        if self.staging_started_monotonic_ns < self.safe_boundary_monotonic_ns:
            raise SnapshotPublishError("snapshot staging precedes its safe boundary")
        if (
            not isinstance(self.gpu_to_cpu_seconds, (int, float))
            or not math.isfinite(self.gpu_to_cpu_seconds)
            or self.gpu_to_cpu_seconds < 0
        ):
            raise SnapshotPublishError(
                "gpu_to_cpu_seconds must be finite and nonnegative"
            )

    @property
    def proposal_id(self) -> str:
        return (
            f"{self.learner_id}-fragment-{self.descriptor.index:06d}-"
            f"sequence-{self.sequence:012d}"
        )

    def materialize_proposal(self) -> Proposal:
        return Proposal.create(
            proposal_id=self.proposal_id,
            identities=self.identities,
            learner_id=self.learner_id,
            descriptor=self.descriptor,
            sequence=self.sequence,
            base_version=self.base_version,
            base_content_identity=self.base_content_identity,
            local_steps=self.local_steps,
            processed_tokens=self.processed_tokens,
            snapshot_local_step=self.snapshot_local_step,
            parameters=self.parameters,
        )


SnapshotValue = FragmentSnapshot | StagedFragmentSnapshot


@dataclasses.dataclass(slots=True)
class _PublishSlot:
    pending: SnapshotValue | None = None
    pending_trace: dict[str, Any] | None = None
    in_flight: SnapshotValue | None = None
    in_flight_trace: dict[str, Any] | None = None
    failed: bool = False


BeforePublish = Callable[[Proposal], None]
TerminalTraceSink = Callable[[Mapping[str, Any]], None]


class BoundedProposalPublisher:
    """One worker and a latest-wins pending slot per learner/fragment."""

    def __init__(
        self,
        store: ProposalStore,
        *,
        learner_id: str,
        before_publish: BeforePublish | None = None,
        trace_sink: TerminalTraceSink | None = None,
        logger: StructuredLogger | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(store, ProposalStore):
            raise SnapshotPublishError("publisher requires a ProposalStore")
        if learner_id not in store.learner_ids:
            raise SnapshotPublishError("publisher learner is outside the frozen store")
        if before_publish is not None and not callable(before_publish):
            raise SnapshotPublishError("before_publish must be callable")
        if trace_sink is not None and not callable(trace_sink):
            raise SnapshotPublishError("trace_sink must be callable")
        if not callable(clock_ns):
            raise SnapshotPublishError("publisher clock_ns must be callable")
        self.store = store
        self.learner_id = learner_id
        self.before_publish = before_publish
        self.trace_sink = trace_sink
        self.logger = logger
        self.clock_ns = clock_ns
        self._condition = threading.Condition()
        self._sink_lock = threading.Lock()
        self._slots = [_PublishSlot() for _ in store.descriptors]
        self._errors: list[BaseException] = []
        self._terminal_emits_in_progress = 0
        self._closing = False
        self._highest_submitted_sequence = [-1] * len(self._slots)
        self._skip_count = 0
        self._replacement_count = 0
        self._captured_count = 0
        self._published_count = 0
        self._published_payload_bytes = 0
        self._gpu_to_cpu_seconds = 0.0
        self._proposal_materialization_seconds = 0.0
        self._cpu_to_fs_seconds = 0.0
        self._outcome_counts: dict[str, int] = {}
        self._latest_terminal: list[dict[str, Any] | None] = [None for _ in self._slots]
        self._maximum_pending = [0] * len(self._slots)
        self._maximum_in_flight = [0] * len(self._slots)
        self._threads = tuple(
            threading.Thread(
                target=self._worker,
                args=(index,),
                name=f"proposal-publisher-{learner_id}-{index}",
                daemon=True,
            )
            for index in range(len(self._slots))
        )
        for thread in self._threads:
            thread.start()

    def _new_trace(self, snapshot: SnapshotValue, outcome: str) -> dict[str, Any]:
        proposal = snapshot.proposal if isinstance(snapshot, FragmentSnapshot) else None
        return {
            "fragment_index": snapshot.descriptor.index,
            "fragment_identity": snapshot.descriptor.identity,
            "sequence": snapshot.sequence,
            "proposal_id": (
                proposal.proposal_id if proposal is not None else snapshot.proposal_id
            ),
            "proposal_content_identity": (
                None if proposal is None else proposal.content_identity
            ),
            "base_version": (
                proposal.base_version if proposal is not None else snapshot.base_version
            ),
            "base_content_identity": (
                proposal.base_content_identity
                if proposal is not None
                else snapshot.base_content_identity
            ),
            "local_steps": (
                proposal.local_steps if proposal is not None else snapshot.local_steps
            ),
            "processed_tokens": (
                proposal.processed_tokens
                if proposal is not None
                else snapshot.processed_tokens
            ),
            "snapshot_local_step": (
                proposal.snapshot_local_step
                if proposal is not None
                else snapshot.snapshot_local_step
            ),
            "payload_bytes": (
                proposal.payload_bytes
                if proposal is not None
                else len(snapshot.parameters)
            ),
            "parameters_sha256": (
                None if proposal is None else proposal.parameters_sha256
            ),
            "safe_boundary_monotonic_ns": snapshot.safe_boundary_monotonic_ns,
            "staging_started_monotonic_ns": snapshot.staging_started_monotonic_ns,
            "staging_completed_monotonic_ns": snapshot.staging_completed_monotonic_ns,
            "gpu_to_cpu_seconds": snapshot.gpu_to_cpu_seconds,
            "proposal_materialization_started_monotonic_ns": None,
            "proposal_materialization_completed_monotonic_ns": None,
            "proposal_materialization_seconds": None,
            "publication_started_monotonic_ns": None,
            "publication_completed_monotonic_ns": None,
            "cpu_to_fs_seconds": None,
            "outcome": outcome,
        }

    def _materialize_proposal(
        self, snapshot: SnapshotValue, trace: dict[str, Any]
    ) -> Proposal:
        if isinstance(snapshot, FragmentSnapshot):
            return snapshot.proposal
        started = self.clock_ns()
        with self._condition:
            trace["proposal_materialization_started_monotonic_ns"] = started
        proposal = snapshot.materialize_proposal()
        completed = self.clock_ns()
        elapsed = max(0.0, (completed - started) / 1e9)
        with self._condition:
            trace["proposal_materialization_completed_monotonic_ns"] = completed
            trace["proposal_materialization_seconds"] = elapsed
            trace["proposal_content_identity"] = proposal.content_identity
            trace["parameters_sha256"] = proposal.parameters_sha256
        return proposal

    def _record_terminal_locked(self, trace: dict[str, Any]) -> dict[str, Any]:
        terminal = dict(trace)
        index = int(terminal["fragment_index"])
        outcome = str(terminal["outcome"])
        self._latest_terminal[index] = terminal
        self._outcome_counts[outcome] = self._outcome_counts.get(outcome, 0) + 1
        if outcome == "published":
            self._published_count += 1
            self._published_payload_bytes += int(terminal["payload_bytes"])
            self._cpu_to_fs_seconds += float(terminal["cpu_to_fs_seconds"])
            materialization_seconds = terminal["proposal_materialization_seconds"]
            if materialization_seconds is not None:
                self._proposal_materialization_seconds += float(materialization_seconds)
        return terminal

    def _emit_terminal(self, trace: Mapping[str, Any]) -> None:
        try:
            with self._sink_lock:
                if self.trace_sink is not None:
                    self.trace_sink(dict(trace))
                if self.logger is not None:
                    self.logger.emit("proposal_publication_terminal", **dict(trace))
        except BaseException as error:
            with self._condition:
                if not self._errors:
                    self._errors.append(error)

    def submit(self, snapshot: SnapshotValue) -> str:
        if not isinstance(snapshot, (FragmentSnapshot, StagedFragmentSnapshot)):
            raise SnapshotPublishError("publisher accepts immutable snapshot values")
        index = snapshot.descriptor.index
        if (
            snapshot.learner_id != self.learner_id
            or snapshot.identities != self.store.identities
            or index >= len(self._slots)
            or snapshot.descriptor != self.store.descriptors[index]
        ):
            raise SnapshotPublishError("snapshot does not match publisher identities")
        terminal: dict[str, Any] | None = None
        with self._condition:
            self._raise_if_failed_locked()
            if self._closing:
                raise SnapshotPublishError("publisher is closing")
            slot = self._slots[index]
            self._captured_count += 1
            self._gpu_to_cpu_seconds += snapshot.gpu_to_cpu_seconds
            if snapshot.sequence <= self._highest_submitted_sequence[index]:
                self._skip_count += 1
                skipped_trace = self._new_trace(snapshot, "skipped_nonmonotonic")
                skipped_trace["publication_completed_monotonic_ns"] = self.clock_ns()
                terminal = self._record_terminal_locked(skipped_trace)
                result = "skipped_nonmonotonic"
            else:
                self._highest_submitted_sequence[index] = snapshot.sequence
                if slot.pending is not None:
                    if slot.pending_trace is None:  # pragma: no cover - state defense
                        raise AssertionError("pending snapshot lacks trace")
                    slot.pending_trace["outcome"] = "replaced_before_publish"
                    slot.pending_trace["publication_completed_monotonic_ns"] = (
                        self.clock_ns()
                    )
                    terminal = self._record_terminal_locked(slot.pending_trace)
                    self._replacement_count += 1
                slot.pending = snapshot
                slot.pending_trace = self._new_trace(snapshot, "pending")
                self._maximum_pending[index] = max(self._maximum_pending[index], 1)
                self._condition.notify_all()
                result = "pending"
        if terminal is not None:
            self._emit_terminal(terminal)
        return result

    def _worker(self, fragment_index: int) -> None:
        while True:
            with self._condition:
                slot = self._slots[fragment_index]
                self._condition.wait_for(
                    lambda: slot.pending is not None or self._closing or slot.failed
                )
                if slot.failed or (self._closing and slot.pending is None):
                    return
                snapshot = slot.pending
                trace = slot.pending_trace
                if (
                    snapshot is None or trace is None
                ):  # pragma: no cover - state defense
                    continue
                slot.pending = None
                slot.pending_trace = None
                slot.in_flight = snapshot
                slot.in_flight_trace = trace
                self._maximum_in_flight[fragment_index] = max(
                    self._maximum_in_flight[fragment_index], 1
                )
                trace["outcome"] = "publishing"
            try:
                proposal = self._materialize_proposal(snapshot, trace)
                started = self.clock_ns()
                trace["publication_started_monotonic_ns"] = started
                if self.before_publish is not None:
                    self.before_publish(proposal)
                self.store.publish(proposal)
            except BaseException as error:  # worker must surface every failure
                completed = self.clock_ns()
                abandoned: dict[str, Any] | None = None
                with self._condition:
                    trace["publication_completed_monotonic_ns"] = completed
                    publication_started = trace["publication_started_monotonic_ns"]
                    trace["cpu_to_fs_seconds"] = (
                        None
                        if publication_started is None
                        else max(0.0, (completed - int(publication_started)) / 1e9)
                    )
                    trace["outcome"] = "failed"
                    slot.in_flight = None
                    slot.in_flight_trace = None
                    slot.failed = True
                    if slot.pending is not None:
                        pending_trace = slot.pending_trace
                        if pending_trace is None:  # pragma: no cover - state defense
                            raise AssertionError("pending snapshot lacks trace")
                        pending_trace["outcome"] = "abandoned_after_failure"
                        pending_trace["publication_completed_monotonic_ns"] = completed
                        slot.pending = None
                        slot.pending_trace = None
                        abandoned = self._record_terminal_locked(pending_trace)
                    if not self._errors:
                        self._errors.append(error)
                    failed = self._record_terminal_locked(trace)
                    terminal_values = [failed]
                    if abandoned is not None:
                        terminal_values.append(abandoned)
                    self._terminal_emits_in_progress += len(terminal_values)
                for terminal in terminal_values:
                    try:
                        self._emit_terminal(terminal)
                    finally:
                        with self._condition:
                            self._terminal_emits_in_progress -= 1
                            self._condition.notify_all()
                return
            completed = self.clock_ns()
            with self._condition:
                trace["publication_completed_monotonic_ns"] = completed
                trace["cpu_to_fs_seconds"] = max(0.0, (completed - started) / 1e9)
                trace["outcome"] = "published"
                slot.in_flight = None
                slot.in_flight_trace = None
                published = self._record_terminal_locked(trace)
                self._terminal_emits_in_progress += 1
            try:
                self._emit_terminal(published)
            finally:
                with self._condition:
                    self._terminal_emits_in_progress -= 1
                    self._condition.notify_all()

    def _raise_if_failed_locked(self) -> None:
        if self._errors:
            error = self._errors[0]
            raise SnapshotPublishError(
                f"background proposal publication failed: {type(error).__name__}: {error}"
            ) from error

    def raise_if_failed(self) -> None:
        with self._condition:
            self._raise_if_failed_locked()

    def drain(self, timeout_seconds: float = 60.0) -> None:
        timeout = _positive_float(timeout_seconds, "drain timeout")
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                any(
                    slot.pending is not None or slot.in_flight is not None
                    for slot in self._slots
                )
                or self._terminal_emits_in_progress
            ):
                if self._terminal_emits_in_progress == 0:
                    self._raise_if_failed_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SnapshotPublishError(
                        "timed out draining proposal publications"
                    )
                self._condition.wait(timeout=remaining)
            self._raise_if_failed_locked()

    def close(self, timeout_seconds: float = 60.0) -> None:
        timeout = _positive_float(timeout_seconds, "close timeout")
        failure: BaseException | None = None
        try:
            self.drain(timeout)
        except BaseException as error:
            failure = error
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self._threads):
            raise SnapshotPublishError("publisher workers did not stop")
        if failure is not None:
            raise failure

    def summary(self) -> dict[str, Any]:
        with self._condition:
            pending = [int(slot.pending is not None) for slot in self._slots]
            in_flight = [int(slot.in_flight is not None) for slot in self._slots]
            active_pending = [
                None if slot.pending_trace is None else dict(slot.pending_trace)
                for slot in self._slots
            ]
            active_in_flight = [
                None if slot.in_flight_trace is None else dict(slot.in_flight_trace)
                for slot in self._slots
            ]
            latest_terminal = [
                None if trace is None else dict(trace)
                for trace in self._latest_terminal
            ]
            return {
                "learner_id": self.learner_id,
                "per_fragment_pending": pending,
                "per_fragment_in_flight": in_flight,
                "pending_upload_count": sum(pending),
                "in_flight_publication_count": sum(in_flight),
                "maximum_pending_per_fragment": list(self._maximum_pending),
                "maximum_in_flight_per_fragment": list(self._maximum_in_flight),
                "snapshot_skip_count": self._skip_count,
                "snapshot_replacement_count": self._replacement_count,
                "captured_snapshot_count": self._captured_count,
                "published_snapshot_count": self._published_count,
                "published_payload_bytes": self._published_payload_bytes,
                "gpu_to_cpu_seconds": self._gpu_to_cpu_seconds,
                "proposal_materialization_seconds": self._proposal_materialization_seconds,
                "cpu_to_fs_seconds": self._cpu_to_fs_seconds,
                "terminal_outcome_counts": dict(sorted(self._outcome_counts.items())),
                "errors": [
                    f"{type(error).__name__}: {error}" for error in self._errors
                ],
                "active_pending": active_pending,
                "active_in_flight": active_in_flight,
                "latest_terminal_per_fragment": latest_terminal,
                "resident_trace_records": sum(
                    trace is not None
                    for trace in (*active_pending, *active_in_flight, *latest_terminal)
                ),
                "resident_trace_record_bound": 3 * len(self._slots),
                "terminal_emits_in_progress": self._terminal_emits_in_progress,
            }


class FragmentSnapshotCoordinator:
    """Safe-boundary target-fragment staging plus bounded publication."""

    def __init__(
        self,
        *,
        identities: GlobalStateIdentities,
        progress: LearnerProgress,
        descriptors: tuple[FragmentStateDescriptor, ...],
        fragment_parameters: Sequence[Sequence[Any]],
        adopted_bases: Sequence[AdoptedFragmentBase],
        schedule: FragmentPublicationSchedule,
        store: ProposalStore,
        publisher: BoundedProposalPublisher | None = None,
        before_publish: BeforePublish | None = None,
        trace_sink: TerminalTraceSink | None = None,
        logger: StructuredLogger | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(identities, GlobalStateIdentities):
            raise SnapshotPublishError("coordinator identities are invalid")
        if not isinstance(progress, LearnerProgress):
            raise SnapshotPublishError("coordinator progress is invalid")
        if not isinstance(schedule, FragmentPublicationSchedule):
            raise SnapshotPublishError("coordinator schedule is invalid")
        if (
            not isinstance(descriptors, tuple)
            or descriptors != store.descriptors
            or identities != store.identities
            or progress.learner_id not in store.learner_ids
        ):
            raise SnapshotPublishError(
                "coordinator and proposal store identities differ"
            )
        if not (
            len(descriptors)
            == len(fragment_parameters)
            == len(adopted_bases)
            == len(progress.fragments)
            == schedule.fragment_count
        ):
            raise SnapshotPublishError("coordinator fragment vectors differ in length")
        groups = tuple(tuple(group) for group in fragment_parameters)
        flattened = [parameter for group in groups for parameter in group]
        if any(not group for group in groups) or len(flattened) != len(
            {id(parameter) for parameter in flattened}
        ):
            raise SnapshotPublishError(
                "fragment parameter groups must be nonempty and non-overlapping"
            )
        for descriptor, group, declared_bytes in zip(
            descriptors, groups, schedule.fragment_bytes, strict=True
        ):
            if descriptor.dtype != "float32" or len(descriptor.shape) != 1:
                raise SnapshotPublishError("snapshot descriptor must be flat float32")
            numel = sum(int(parameter.numel()) for parameter in group)
            if numel != descriptor.shape[0] or declared_bytes != numel * 4:
                raise SnapshotPublishError(
                    "fragment parameters descriptor and byte schedule differ"
                )
        bases = list(adopted_bases)
        if any(
            base.version != fragment.global_version
            for base, fragment in zip(bases, progress.fragments, strict=True)
        ):
            raise SnapshotPublishError(
                "adopted base versions differ from learner progress"
            )
        if publisher is not None and (
            before_publish is not None or trace_sink is not None
        ):
            raise SnapshotPublishError(
                "publisher hooks cannot be supplied with an existing publisher"
            )
        if not callable(clock_ns):
            raise SnapshotPublishError("coordinator clock_ns must be callable")
        self.identities = identities
        self.progress = progress
        self.descriptors = descriptors
        self.fragment_parameters = groups
        self.schedule = schedule
        self.logger = logger
        self.clock_ns = clock_ns
        self._bases = bases
        self._capture_lock = threading.Lock()
        self._last_boundary_step = progress.local_optimizer_steps
        self.publisher = publisher or BoundedProposalPublisher(
            store,
            learner_id=progress.learner_id,
            before_publish=before_publish,
            trace_sink=trace_sink,
            logger=logger,
            clock_ns=clock_ns,
        )

    def _validate_safe_boundary(self, event: SafeBoundaryEvent) -> None:
        if not isinstance(event, SafeBoundaryEvent):
            raise SnapshotPublishError("snapshot callback requires SafeBoundaryEvent")
        if event.learner_id != self.progress.learner_id:
            raise SnapshotPublishError("safe-boundary learner identity mismatch")
        if (
            event.local_optimizer_step != self.progress.local_optimizer_steps
            or event.local_optimizer_step <= self._last_boundary_step
        ):
            raise SnapshotPublishError("safe-boundary step is stale or inconsistent")
        if (
            event.fragment_global_versions
            != tuple(fragment.global_version for fragment in self.progress.fragments)
            or event.fragment_local_steps
            != tuple(
                fragment.local_steps_since_adoption
                for fragment in self.progress.fragments
            )
            or event.fragment_processed_input_tokens
            != tuple(
                fragment.processed_input_tokens_since_adoption
                for fragment in self.progress.fragments
            )
            or event.processed_input_tokens_total
            != self.progress.processed_input_tokens
        ):
            raise SnapshotPublishError("safe-boundary progress vector is inconsistent")
        if any(
            parameter.grad is not None
            for group in self.fragment_parameters
            for parameter in group
        ):
            raise SnapshotPublishError(
                "snapshot rejected while optimizer gradients remain active"
            )

    def on_safe_boundary(self, event: SafeBoundaryEvent) -> Mapping[str, Any]:
        with self._capture_lock:
            self._validate_safe_boundary(event)
            due = self.schedule.due_fragments(event.local_optimizer_step)
            staged: list[StagedFragmentSnapshot] = []
            for index in due:
                fragment_progress = self.progress.fragments[index]
                base = self._bases[index]
                if base.version != fragment_progress.global_version:
                    raise SnapshotPublishError(
                        "snapshot base version differs from fragment progress"
                    )
                # Reservation precedes staging.  A failure or crash may leave a
                # gap, but restoring LearnerProgress can never reuse a sequence.
                fragment_progress.proposal_sequence += 1
                sequence = fragment_progress.proposal_sequence
                started = self.clock_ns()
                payload = serialize_fragment_parameters(
                    self.fragment_parameters[index],
                    self.schedule.fragment_bytes[index],
                )
                completed = self.clock_ns()
                snapshot = StagedFragmentSnapshot(
                    identities=self.identities,
                    learner_id=self.progress.learner_id,
                    descriptor=self.descriptors[index],
                    sequence=sequence,
                    base_version=base.version,
                    base_content_identity=base.content_identity,
                    local_steps=fragment_progress.local_steps_since_adoption,
                    processed_tokens=(
                        fragment_progress.processed_input_tokens_since_adoption
                    ),
                    snapshot_local_step=event.local_optimizer_step,
                    parameters=payload,
                    safe_boundary_monotonic_ns=event.safe_boundary_monotonic_ns,
                    staging_started_monotonic_ns=started,
                    staging_completed_monotonic_ns=completed,
                    gpu_to_cpu_seconds=max(0.0, (completed - started) / 1e9),
                )
                staged.append(snapshot)
                self.publisher.submit(snapshot)
                if self.logger is not None:
                    self.logger.emit(
                        "fragment_snapshot_staged",
                        fragment_index=index,
                        sequence=sequence,
                        base_version=base.version,
                        base_content_identity=base.content_identity,
                        local_steps=snapshot.local_steps,
                        processed_tokens=snapshot.processed_tokens,
                        snapshot_local_step=snapshot.snapshot_local_step,
                        payload_bytes=len(snapshot.parameters),
                        parameters_sha256=None,
                        proposal_content_identity=None,
                        identity_materialization="background_publisher",
                        gpu_to_cpu_seconds=snapshot.gpu_to_cpu_seconds,
                    )
            self._last_boundary_step = event.local_optimizer_step
        publication = self.publisher.summary()
        return {
            "snapshot_due_fragments": list(due),
            "snapshot_staged_count": len(staged),
            "gpu_to_cpu_seconds": sum(item.gpu_to_cpu_seconds for item in staged),
            "cpu_to_fs_seconds": publication["cpu_to_fs_seconds"],
            "pending_upload_count": publication["pending_upload_count"],
            "in_flight_publication_count": publication["in_flight_publication_count"],
            "snapshot_skip_count": publication["snapshot_skip_count"],
            "snapshot_replacement_count": publication["snapshot_replacement_count"],
        }

    def update_adopted_base(
        self,
        fragment_index: int,
        *,
        version: int,
        content_identity: str,
    ) -> None:
        index = _nonnegative_int(fragment_index, "fragment index")
        base = AdoptedFragmentBase(version, content_identity)
        with self._capture_lock:
            try:
                current = self._bases[index]
                fragment = self.progress.fragments[index]
            except IndexError as error:
                raise SnapshotPublishError(
                    f"unknown fragment index: {index}"
                ) from error
            if base.version <= current.version:
                raise SnapshotPublishError("adopted base version must increase")
            self._bases[index] = base
            fragment.global_version = base.version
            fragment.local_steps_since_adoption = 0
            fragment.processed_input_tokens_since_adoption = 0

    def update_adopted_base_context(
        self,
        fragment_index: int,
        version: int,
        content_identity: str,
    ) -> None:
        """Advance snapshot metadata after an adoption-owned counter reset."""

        index = _nonnegative_int(fragment_index, "fragment index")
        base = AdoptedFragmentBase(version, content_identity)
        with self._capture_lock:
            try:
                current = self._bases[index]
                fragment = self.progress.fragments[index]
            except IndexError as error:
                raise SnapshotPublishError(
                    f"unknown fragment index: {index}"
                ) from error
            if base.version <= current.version:
                raise SnapshotPublishError("adopted base version must increase")
            if (
                fragment.global_version != base.version
                or fragment.local_steps_since_adoption != 0
                or fragment.processed_input_tokens_since_adoption != 0
            ):
                raise SnapshotPublishError(
                    "adoption must update target progress before snapshot base context"
                )
            self._bases[index] = base

    def drain(self, timeout_seconds: float = 60.0) -> None:
        self.publisher.drain(timeout_seconds)

    def close(self, timeout_seconds: float = 60.0) -> None:
        self.publisher.close(timeout_seconds)

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "learner_id": self.progress.learner_id,
            "identities": self.identities.to_dict(),
            "schedule": self.schedule.to_dict(),
            "adopted_bases": [dataclasses.asdict(base) for base in self._bases],
            "progress": self.progress.to_dict(),
            "publication": self.publisher.summary(),
        }


def fragment_payload_sha256(parameters: Sequence[Any]) -> str:
    """Independent exact fp32 target-fragment digest used by evidence analyzers."""

    expected = sum(int(parameter.numel()) for parameter in parameters) * 4
    return hashlib.sha256(
        serialize_fragment_parameters(parameters, expected)
    ).hexdigest()


def serialize_fragment_parameters(
    parameters: Sequence[Any], expected_bytes: int
) -> bytes:
    """Copy one ordered fragment to immutable little-endian fp32 bytes."""

    import torch

    expected = _positive_int(expected_bytes, "expected fragment bytes")
    stream = io.BytesIO()
    for parameter in parameters:
        value = parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(value).all().item()):
            raise SnapshotPublishError("snapshot parameters contain NaN or Inf")
        array = np.asarray(value.numpy(), dtype="<f4")
        chunk = array.tobytes(order="C")
        if stream.write(chunk) != len(chunk):  # pragma: no cover - BytesIO contract
            raise SnapshotPublishError("snapshot byte buffer accepted a partial write")
    payload = stream.getvalue()
    if len(payload) != expected:
        raise SnapshotPublishError("target-fragment snapshot byte count mismatch")
    return payload
