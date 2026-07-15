from __future__ import annotations

import dataclasses
import hashlib
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .global_state import FragmentGlobalState, GlobalStateStore
from .identity import canonical_digest
from .learner import LearnerProgress, SafeBoundaryEvent
from .learner_publish import fragment_payload_sha256
from .logging import StructuredLogger


class AdoptionError(RuntimeError):
    pass


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AdoptionError(f"{name} must be a non-negative integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise AdoptionError(f"{name} must be finite and positive")
    return float(value)


def _hex_identity(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AdoptionError(f"{name} must be a lowercase SHA-256 identity")
    return value


def _tensor_bytes(value: Any) -> bytes:
    import torch

    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.numel() == 0:
        return b""
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _optimizer_value_identity(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        payload = _tensor_bytes(value)
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, Mapping):
        rows = [
            {
                "key_type": type(key).__name__,
                "key": repr(key),
                "value": _optimizer_value_identity(item),
            }
            for key, item in value.items()
        ]
        rows.sort(key=lambda row: (row["key_type"], row["key"]))
        return {"kind": "mapping", "items": rows}
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [_optimizer_value_identity(item) for item in value],
        }
    if value is None or isinstance(value, (bool, int, str)):
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AdoptionError("optimizer state contains NaN or Inf")
        return {"kind": "float", "value": value}
    raise AdoptionError(f"unsupported optimizer state value: {type(value).__name__}")


def optimizer_fragment_state_sha256(
    optimizer: Any, fragment_parameters: Sequence[Any]
) -> str:
    """Canonical exact identity for the ordered optimizer state of a fragment."""

    states = []
    for position, parameter in enumerate(fragment_parameters):
        states.append(
            {
                "parameter_position": position,
                "state": _optimizer_value_identity(optimizer.state.get(parameter, {})),
            }
        )
    return canonical_digest({"schema_version": 1, "parameters": states})


@dataclasses.dataclass(frozen=True, slots=True)
class PendingFragmentAdoption:
    state: FragmentGlobalState
    discovered_completed_step: int
    fetch_started_monotonic_ns: int
    fetch_completed_monotonic_ns: int
    discovered_unix_ns: int
    fs_to_cpu_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.state, FragmentGlobalState):
            raise AdoptionError("pending adoption requires FragmentGlobalState")
        for name in (
            "discovered_completed_step",
            "fetch_started_monotonic_ns",
            "fetch_completed_monotonic_ns",
            "discovered_unix_ns",
        ):
            _nonnegative_int(getattr(self, name), name)
        if self.fetch_completed_monotonic_ns < self.fetch_started_monotonic_ns:
            raise AdoptionError("fragment fetch interval is reversed")
        if (
            not isinstance(self.fs_to_cpu_seconds, (int, float))
            or not math.isfinite(self.fs_to_cpu_seconds)
            or self.fs_to_cpu_seconds < 0
        ):
            raise AdoptionError("fs_to_cpu_seconds must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_index": self.state.descriptor.index,
            "fragment_identity": self.state.descriptor.identity,
            "version": self.state.version,
            "content_identity": self.state.content_identity,
            "parameters_sha256": hashlib.sha256(self.state.parameters).hexdigest(),
            "discovered_completed_step": self.discovered_completed_step,
            "fetch_started_monotonic_ns": self.fetch_started_monotonic_ns,
            "fetch_completed_monotonic_ns": self.fetch_completed_monotonic_ns,
            "discovered_unix_ns": self.discovered_unix_ns,
            "fs_to_cpu_seconds": self.fs_to_cpu_seconds,
        }


DiscoveryTraceSink = Callable[[Mapping[str, Any]], None]


class LatestFragmentPoller:
    """Fixed-slot verifier with one latest-wins pending state per fragment."""

    def __init__(
        self,
        store: GlobalStateStore,
        *,
        adopted_versions: Sequence[int],
        completed_step: Callable[[], int],
        poll_interval_seconds: float = 0.01,
        trace_sink: DiscoveryTraceSink | None = None,
        logger: StructuredLogger | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        autostart: bool = True,
    ) -> None:
        if not isinstance(store, GlobalStateStore):
            raise AdoptionError("latest poller requires a GlobalStateStore")
        versions = tuple(
            _nonnegative_int(value, "adopted version") for value in adopted_versions
        )
        if len(versions) != len(store.descriptors):
            raise AdoptionError("adopted versions must match the fragment store")
        if not callable(completed_step):
            raise AdoptionError("completed_step must be callable")
        if trace_sink is not None and not callable(trace_sink):
            raise AdoptionError("trace_sink must be callable")
        self.store = store
        self.completed_step = completed_step
        self.poll_interval_seconds = _positive_float(
            poll_interval_seconds, "poll interval"
        )
        self.trace_sink = trace_sink
        self.logger = logger
        self.clock_ns = clock_ns
        self.wall_clock_ns = wall_clock_ns
        self._condition = threading.Condition()
        self._poll_lock = threading.Lock()
        self._stop = threading.Event()
        self._adopted_versions = list(versions)
        self._highest_observed_versions = list(versions)
        self._pending: list[PendingFragmentAdoption | None] = [
            None for _ in versions
        ]
        self._maximum_pending = [0 for _ in versions]
        self._poll_cycles = 0
        self._read_count = 0
        self._discovery_count = 0
        self._pending_replacement_count = 0
        self._repeated_current_count = 0
        self._ignored_stale_count = 0
        self._errors: list[BaseException] = []
        self._started = False
        self._closed = False
        self._thread: threading.Thread | None = None
        if autostart:
            self.start()

    def start(self) -> None:
        with self._condition:
            self._raise_if_failed_locked()
            if self._closed:
                raise AdoptionError("latest poller is closed")
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._worker,
                name="global-fragment-latest-poller",
                daemon=True,
            )
            self._thread.start()

    def _record_observation(
        self, observation: PendingFragmentAdoption
    ) -> dict[str, Any] | None:
        state = observation.state
        index = state.descriptor.index
        with self._condition:
            highest = self._highest_observed_versions[index]
            if state.version < highest:
                raise AdoptionError(
                    f"global current slot regressed for fragment {index}: "
                    f"{state.version} < {highest}"
                )
            current = self._adopted_versions[index]
            pending = self._pending[index]
            if state.version <= current:
                if state.version == current:
                    self._repeated_current_count += 1
                else:
                    self._ignored_stale_count += 1
                return None
            if pending is not None and state.version <= pending.state.version:
                if state.version == pending.state.version:
                    self._repeated_current_count += 1
                else:
                    self._ignored_stale_count += 1
                return None
            replaced_version = None
            if pending is not None:
                replaced_version = pending.state.version
                self._pending_replacement_count += 1
            self._highest_observed_versions[index] = state.version
            self._pending[index] = observation
            self._maximum_pending[index] = 1
            self._discovery_count += 1
            self._condition.notify_all()
        trace = observation.to_dict()
        trace.update(
            {
                "event": "global_fragment_discovered",
                "adopted_version_at_discovery": current,
                "replaced_pending_version": replaced_version,
            }
        )
        return trace

    def poll_once(self) -> tuple[Mapping[str, Any], ...]:
        traces: list[Mapping[str, Any]] = []
        with self._poll_lock:
            for index in range(len(self.store.descriptors)):
                started = self.clock_ns()
                state = self.store.load_fragment(index, timeout_seconds=0)
                completed = self.clock_ns()
                step = _nonnegative_int(self.completed_step(), "completed step")
                observation = PendingFragmentAdoption(
                    state=state,
                    discovered_completed_step=step,
                    fetch_started_monotonic_ns=started,
                    fetch_completed_monotonic_ns=completed,
                    discovered_unix_ns=_nonnegative_int(
                        self.wall_clock_ns(), "discovery wall clock"
                    ),
                    fs_to_cpu_seconds=max(0.0, (completed - started) / 1e9),
                )
                trace = self._record_observation(observation)
                with self._condition:
                    self._read_count += 1
                if trace is not None:
                    traces.append(trace)
                    if self.trace_sink is not None:
                        self.trace_sink(dict(trace))
                    if self.logger is not None:
                        self.logger.emit(
                            "global_fragment_discovered",
                            **{key: value for key, value in trace.items() if key != "event"},
                        )
            with self._condition:
                self._poll_cycles += 1
        return tuple(traces)

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except BaseException as error:
                with self._condition:
                    if not self._errors:
                        self._errors.append(error)
                    self._condition.notify_all()
                self._stop.set()
                return
            self._stop.wait(self.poll_interval_seconds)

    def _raise_if_failed_locked(self) -> None:
        if self._errors:
            error = self._errors[0]
            raise AdoptionError(
                f"background global fragment polling failed: "
                f"{type(error).__name__}: {error}"
            ) from error

    def raise_if_failed(self) -> None:
        with self._condition:
            self._raise_if_failed_locked()

    def take_pending(self) -> tuple[PendingFragmentAdoption, ...]:
        with self._condition:
            self._raise_if_failed_locked()
            result = tuple(item for item in self._pending if item is not None)
            self._pending = [None for _ in self._pending]
            return result

    def mark_adopted(self, fragment_index: int, version: int) -> None:
        index = _nonnegative_int(fragment_index, "fragment index")
        value = _nonnegative_int(version, "adopted version")
        with self._condition:
            try:
                current = self._adopted_versions[index]
            except IndexError as error:
                raise AdoptionError(f"unknown fragment index: {index}") from error
            if value <= current:
                raise AdoptionError("adopted version must strictly increase")
            self._adopted_versions[index] = value
            pending = self._pending[index]
            if pending is not None and pending.state.version <= value:
                self._pending[index] = None

    def summary(self) -> dict[str, Any]:
        with self._condition:
            pending = [
                None if item is None else item.to_dict() for item in self._pending
            ]
            return {
                "schema_version": 1,
                "adopted_versions": list(self._adopted_versions),
                "highest_observed_versions": list(self._highest_observed_versions),
                "pending": pending,
                "pending_count": sum(item is not None for item in self._pending),
                "maximum_pending_per_fragment": list(self._maximum_pending),
                "resident_pending_bound": len(self._pending),
                "poll_cycles": self._poll_cycles,
                "fixed_slot_read_count": self._read_count,
                "discovery_count": self._discovery_count,
                "pending_replacement_count": self._pending_replacement_count,
                "repeated_current_count": self._repeated_current_count,
                "ignored_stale_count": self._ignored_stale_count,
                "errors": [
                    f"{type(error).__name__}: {error}" for error in self._errors
                ],
                "started": self._started,
                "closed": self._closed,
            }

    def close(self, timeout_seconds: float = 60.0) -> None:
        timeout = _positive_float(timeout_seconds, "close timeout")
        with self._condition:
            if self._closed:
                self._raise_if_failed_locked()
                return
            self._closed = True
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise AdoptionError("latest poller did not stop")
        self.raise_if_failed()


BaseContextSink = Callable[[int, int, str], None]
AdoptionTraceSink = Callable[[Mapping[str, Any]], None]


class FragmentAdoptionCoordinator:
    """Apply independently discovered complete fragments at optimizer boundaries."""

    def __init__(
        self,
        *,
        store: GlobalStateStore,
        progress: LearnerProgress,
        initial_states: Sequence[FragmentGlobalState],
        fragment_parameters: Sequence[Sequence[Any]],
        optimizer: Any | None = None,
        identity_audit: bool = False,
        base_context_sink: BaseContextSink | None = None,
        trace_sink: AdoptionTraceSink | None = None,
        logger: StructuredLogger | None = None,
        poll_interval_seconds: float = 0.01,
        autostart: bool = True,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        wall_clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not isinstance(store, GlobalStateStore):
            raise AdoptionError("adoption coordinator requires GlobalStateStore")
        if not isinstance(progress, LearnerProgress):
            raise AdoptionError("adoption coordinator requires LearnerProgress")
        states = tuple(initial_states)
        groups = tuple(tuple(group) for group in fragment_parameters)
        if not (
            len(states)
            == len(groups)
            == len(progress.fragments)
            == len(store.descriptors)
        ):
            raise AdoptionError("adoption fragment vectors differ in length")
        if any(not group for group in groups):
            raise AdoptionError("adoption fragment groups must be non-empty")
        flattened = [parameter for group in groups for parameter in group]
        if len(flattened) != len({id(parameter) for parameter in flattened}):
            raise AdoptionError("adoption fragment groups must not overlap")
        for state, descriptor, fragment, group in zip(
            states, store.descriptors, progress.fragments, groups, strict=True
        ):
            if (
                state.identities != store.identities
                or state.descriptor != descriptor
                or state.version != fragment.global_version
            ):
                raise AdoptionError("initial global state identity or version mismatch")
            if descriptor.dtype != "float32" or len(descriptor.shape) != 1:
                raise AdoptionError("adoption descriptors must be flat float32")
            if sum(int(parameter.numel()) for parameter in group) != descriptor.shape[0]:
                raise AdoptionError("adoption parameter group differs from descriptor")
        if identity_audit and optimizer is None:
            raise AdoptionError("identity audit requires an optimizer")
        if base_context_sink is not None and not callable(base_context_sink):
            raise AdoptionError("base_context_sink must be callable")
        if trace_sink is not None and not callable(trace_sink):
            raise AdoptionError("trace_sink must be callable")
        self.store = store
        self.progress = progress
        self.initial_states = states
        self.fragment_parameters = groups
        self.optimizer = optimizer
        self.identity_audit = bool(identity_audit)
        self.base_context_sink = base_context_sink
        self.trace_sink = trace_sink
        self.logger = logger
        self.clock_ns = clock_ns
        self.wall_clock_ns = wall_clock_ns
        self._lock = threading.Lock()
        self._last_boundary_step = progress.local_optimizer_steps
        self._content_identities = [state.content_identity for state in states]
        self._latest_trace: list[dict[str, Any] | None] = [None for _ in states]
        self._adoption_count = 0
        self._skipped_intermediate_versions = 0
        self._fs_to_cpu_seconds = 0.0
        self._cpu_to_gpu_seconds = 0.0
        self.poller = LatestFragmentPoller(
            store,
            adopted_versions=tuple(fragment.global_version for fragment in progress.fragments),
            completed_step=lambda: self.progress.local_optimizer_steps,
            poll_interval_seconds=poll_interval_seconds,
            logger=logger,
            clock_ns=clock_ns,
            wall_clock_ns=wall_clock_ns,
            autostart=autostart,
        )

    def start(self) -> None:
        self.poller.start()

    def poll_once(self) -> tuple[Mapping[str, Any], ...]:
        return self.poller.poll_once()

    def _validate_safe_boundary(self, event: SafeBoundaryEvent) -> None:
        if not isinstance(event, SafeBoundaryEvent):
            raise AdoptionError("adoption callback requires SafeBoundaryEvent")
        if event.learner_id != self.progress.learner_id:
            raise AdoptionError("safe-boundary learner identity mismatch")
        if (
            event.local_optimizer_step != self.progress.local_optimizer_steps
            or event.local_optimizer_step <= self._last_boundary_step
        ):
            raise AdoptionError("safe-boundary step is stale or inconsistent")
        if (
            event.fragment_global_versions
            != tuple(fragment.global_version for fragment in self.progress.fragments)
            or event.fragment_local_steps
            != tuple(
                fragment.local_steps_since_adoption for fragment in self.progress.fragments
            )
            or event.fragment_processed_input_tokens
            != tuple(
                fragment.processed_input_tokens_since_adoption
                for fragment in self.progress.fragments
            )
            or event.processed_input_tokens_total
            != self.progress.processed_input_tokens
        ):
            raise AdoptionError("safe-boundary progress vector is inconsistent")
        if any(
            parameter.grad is not None
            for group in self.fragment_parameters
            for parameter in group
        ):
            raise AdoptionError(
                "adoption rejected while optimizer gradients remain active"
            )

    def _decode_sources(
        self, observation: PendingFragmentAdoption
    ) -> tuple[Any, ...]:
        import torch

        state = observation.state
        descriptor = self.store.descriptors[state.descriptor.index]
        expected_bytes = descriptor.shape[0] * 4
        if len(state.parameters) != expected_bytes:
            raise AdoptionError(
                f"fragment {descriptor.index} payload byte count differs from descriptor"
            )
        array = np.frombuffer(state.parameters, dtype="<f4")
        if array.size != descriptor.shape[0] or not bool(np.isfinite(array).all()):
            raise AdoptionError("global fragment parameters contain NaN Inf or wrong shape")
        sources = []
        offset = 0
        for parameter in self.fragment_parameters[descriptor.index]:
            count = int(parameter.numel())
            source = torch.from_numpy(array[offset : offset + count].copy()).reshape(
                parameter.shape
            )
            sources.append(source)
            offset += count
        if offset != array.size:
            raise AdoptionError("global fragment parameters do not cover the target")
        return tuple(sources)

    def _audit_hashes(self) -> tuple[list[str], list[str]]:
        if not self.identity_audit or self.optimizer is None:
            return [], []
        parameters = [fragment_payload_sha256(group) for group in self.fragment_parameters]
        moments = [
            optimizer_fragment_state_sha256(self.optimizer, group)
            for group in self.fragment_parameters
        ]
        return parameters, moments

    def _apply_one(
        self, event: SafeBoundaryEvent, observation: PendingFragmentAdoption
    ) -> dict[str, Any] | None:
        import torch

        state = observation.state
        index = state.descriptor.index
        fragment = self.progress.fragments[index]
        if state.version <= fragment.global_version:
            return None
        sources = self._decode_sources(observation)
        parameter_hashes_before, moment_hashes_before = self._audit_hashes()
        counters_before = [
            {
                "global_version": item.global_version,
                "local_steps": item.local_steps_since_adoption,
                "processed_input_tokens": item.processed_input_tokens_since_adoption,
                "proposal_sequence": item.proposal_sequence,
            }
            for item in self.progress.fragments
        ]
        version_before = fragment.global_version
        started = self.clock_ns()
        with torch.no_grad():
            for parameter, source in zip(
                self.fragment_parameters[index], sources, strict=True
            ):
                parameter.copy_(
                    source.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                        non_blocking=False,
                    )
                )
        cuda_devices = {
            parameter.device
            for parameter in self.fragment_parameters[index]
            if parameter.device.type == "cuda"
        }
        for device in cuda_devices:
            torch.cuda.synchronize(device)
        completed = self.clock_ns()
        cpu_to_gpu_seconds = max(0.0, (completed - started) / 1e9)

        fragment.global_version = state.version
        fragment.local_steps_since_adoption = 0
        fragment.processed_input_tokens_since_adoption = 0
        self._content_identities[index] = state.content_identity
        if self.base_context_sink is not None:
            self.base_context_sink(index, state.version, state.content_identity)
        self.poller.mark_adopted(index, state.version)

        parameter_hashes_after, moment_hashes_after = self._audit_hashes()
        target_payload_sha256 = hashlib.sha256(state.parameters).hexdigest()
        if self.identity_audit:
            if any(
                before != after
                for position, (before, after) in enumerate(
                    zip(parameter_hashes_before, parameter_hashes_after, strict=True)
                )
                if position != index
            ):
                raise AdoptionError("adoption changed a non-target fragment")
            if all(
                parameter.dtype == torch.float32
                for parameter in self.fragment_parameters[index]
            ) and parameter_hashes_after[index] != target_payload_sha256:
                raise AdoptionError("adopted target differs from verified global payload")
            if moment_hashes_before != moment_hashes_after:
                raise AdoptionError("adoption changed inner optimizer state")
        counters_after = [
            {
                "global_version": item.global_version,
                "local_steps": item.local_steps_since_adoption,
                "processed_input_tokens": item.processed_input_tokens_since_adoption,
                "proposal_sequence": item.proposal_sequence,
            }
            for item in self.progress.fragments
        ]
        jump = state.version - version_before
        adoption_unix_ns = _nonnegative_int(
            self.wall_clock_ns(), "adoption wall clock"
        )
        trace: dict[str, Any] = {
            "event": "global_fragment_adopted",
            "learner_id": self.progress.learner_id,
            "fragment_index": index,
            "fragment_identity": state.descriptor.identity,
            "content_identity": state.content_identity,
            "from_version": version_before,
            "to_version": state.version,
            "version_jump": jump,
            "skipped_intermediate_versions": max(0, jump - 1),
            "safe_boundary_local_step": event.local_optimizer_step,
            "safe_boundary_monotonic_ns": event.safe_boundary_monotonic_ns,
            "adoption_completed_monotonic_ns": completed,
            "adoption_unix_ns": adoption_unix_ns,
            "discovered_completed_step": observation.discovered_completed_step,
            "discovered_unix_ns": observation.discovered_unix_ns,
            "extra_local_steps_before_adoption": max(
                0, event.local_optimizer_step - observation.discovered_completed_step
            ),
            "discovery_to_adoption_seconds": max(
                0.0,
                (completed - observation.fetch_completed_monotonic_ns) / 1e9,
            ),
            "fs_to_cpu_seconds": observation.fs_to_cpu_seconds,
            "cpu_to_gpu_seconds": cpu_to_gpu_seconds,
            "target_payload_bytes": len(state.parameters),
            "target_payload_sha256": target_payload_sha256,
            "parameter_hashes_before": parameter_hashes_before,
            "parameter_hashes_after": parameter_hashes_after,
            "optimizer_state_hashes_before": moment_hashes_before,
            "optimizer_state_hashes_after": moment_hashes_after,
            "counters_before": counters_before,
            "counters_after": counters_after,
            "version_vector_after": [
                item.global_version for item in self.progress.fragments
            ],
            "identity_audit": self.identity_audit,
        }
        self._adoption_count += 1
        self._skipped_intermediate_versions += max(0, jump - 1)
        self._fs_to_cpu_seconds += observation.fs_to_cpu_seconds
        self._cpu_to_gpu_seconds += cpu_to_gpu_seconds
        self._latest_trace[index] = trace
        if self.trace_sink is not None:
            self.trace_sink(dict(trace))
        if self.logger is not None:
            self.logger.emit(
                "global_fragment_adopted",
                **{key: value for key, value in trace.items() if key != "event"},
            )
        return trace

    def on_safe_boundary(self, event: SafeBoundaryEvent) -> Mapping[str, Any]:
        with self._lock:
            self.poller.raise_if_failed()
            self._validate_safe_boundary(event)
            observations = self.poller.take_pending()
            traces = []
            for observation in observations:
                trace = self._apply_one(event, observation)
                if trace is not None:
                    traces.append(trace)
            self._last_boundary_step = event.local_optimizer_step
            poller = self.poller.summary()
            version_lag = [
                max(0, observed - fragment.global_version)
                for observed, fragment in zip(
                    poller["highest_observed_versions"],
                    self.progress.fragments,
                    strict=True,
                )
            ]
        extra_steps: list[int | None] = [None for _ in self.progress.fragments]
        for trace in traces:
            extra_steps[int(trace["fragment_index"])] = int(
                trace["extra_local_steps_before_adoption"]
            )
        return {
            "adoption_applied_count": len(traces),
            "adoption_applied_fragments": [
                int(trace["fragment_index"]) for trace in traces
            ],
            "adoption_version_vector_after": [
                fragment.global_version for fragment in self.progress.fragments
            ],
            "adoption_traces": traces,
            "fs_to_cpu_seconds": sum(
                float(trace["fs_to_cpu_seconds"]) for trace in traces
            ),
            "cpu_to_gpu_seconds": sum(
                float(trace["cpu_to_gpu_seconds"]) for trace in traces
            ),
            "fragment_version_lag": version_lag,
            "extra_local_steps_before_adoption": extra_steps,
            "pending_adoption_count": int(poller["pending_count"]),
        }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "learner_id": self.progress.learner_id,
                "identities": self.store.identities.to_dict(),
                "progress": self.progress.to_dict(),
                "content_identities": list(self._content_identities),
                "identity_audit": self.identity_audit,
                "adoption_count": self._adoption_count,
                "skipped_intermediate_versions": self._skipped_intermediate_versions,
                "fs_to_cpu_seconds": self._fs_to_cpu_seconds,
                "cpu_to_gpu_seconds": self._cpu_to_gpu_seconds,
                "latest_trace_per_fragment": [
                    None if trace is None else dict(trace)
                    for trace in self._latest_trace
                ],
                "resident_trace_records": sum(
                    trace is not None for trace in self._latest_trace
                ),
                "resident_trace_record_bound": len(self._latest_trace),
                "poller": self.poller.summary(),
            }

    def close(self, timeout_seconds: float = 60.0) -> None:
        self.poller.close(timeout_seconds)
