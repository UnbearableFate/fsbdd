from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fsbdd.diloco.protocol.global_commit import (
    AtomicCommitRequest,
    AtomicFragmentAuthority,
    AtomicGlobalCommitStore,
)
from fsbdd.diloco.protocol.global_state import (
    BootstrapFragment,
    FragmentStateDescriptor,
    GlobalStateIdentities,
    GlobalStateStore,
)
from fsbdd.diloco.common.identity import canonical_bytes, canonical_digest, file_digest
from fsbdd.auxiliary.contracts.manifest import build_manifest
from fsbdd.diloco.protocol.proposal import (
    EligibilityPolicy,
    Proposal,
    ProposalStore,
    RetainedBaseIdentity,
    compute_candidate_weights,
    select_candidates,
)
from fsbdd.diloco.protocol.storage import PosixStorageBackend, PublicationInterrupted
from fsbdd.diloco.syncer.readiness import FrozenSelection


class GlobalCommitStressError(RuntimeError):
    pass


_HEX = frozenset("0123456789abcdef")


def _identity(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_hex(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise GlobalCommitStressError(f"{field} must be a SHA-256 identity")
    return value


def _write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _wait_json(path: Path, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"coordination record did not appear: {path}")
            time.sleep(0.01)
            continue
        if not isinstance(value, dict) or value.get("complete") is not True:
            raise GlobalCommitStressError(f"coordination record is malformed: {path}")
        return value


def _load_config(path: Path, expected_identity: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_identity:
        raise GlobalCommitStressError("global commit config identity mismatch")
    value = json.loads(raw)
    expected = {
        "schema_version",
        "learner_count",
        "fragment_count",
        "s_max",
        "selection",
        "policy",
        "formal_workload",
        "publication_fault_points",
        "coordination_timeout_seconds",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise GlobalCommitStressError("global commit config schema mismatch")
    selection = value["selection"]
    policy = value["policy"]
    workload = value["formal_workload"]
    if (
        value["schema_version"] != 1
        or value["learner_count"] != 4
        or value["fragment_count"] != 2
        or value["s_max"] != 0
        or selection != {"q": 4, "q_fresh": 4, "max_contributors": 4, "lambda_s": 1.0}
        or set(policy)
        != {
            "merge",
            "outer_optimizer",
            "learning_rate",
            "momentum",
            "nesterov",
            "accumulation_dtype",
        }
        or policy["merge"] != "direct_weighted_average"
        or policy["outer_optimizer"] != "sgd"
        or policy["accumulation_dtype"] != "float32"
        or set(workload)
        != {
            "successful_updates",
            "fault_replays_per_point",
            "concurrent_reader_threads",
            "minimum_reader_observations",
        }
        or workload["successful_updates"] != 20
        or workload["fault_replays_per_point"] != 4
        or workload["concurrent_reader_threads"] != 8
        or workload["minimum_reader_observations"] < 100
        or value["publication_fault_points"]
        != [
            "before_payload_write",
            "after_payload_write",
            "before_record_replace",
            "after_record_replace",
        ]
    ):
        raise GlobalCommitStressError(
            "global commit profile differs from frozen Stage 1 values"
        )
    return value


def _rank() -> int:
    for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"):
        if name in os.environ:
            return int(os.environ[name])
    raise GlobalCommitStressError("role command requires an MPI launcher rank")


def _size() -> int:
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"):
        if name in os.environ:
            return int(os.environ[name])
    raise GlobalCommitStressError("role command requires an MPI launcher size")


def _descriptors(count: int) -> tuple[FragmentStateDescriptor, ...]:
    return tuple(
        FragmentStateDescriptor(
            index=index,
            identity=_identity(f"s1-11-fragment-{index}"),
            dtype="uint8",
            shape=(32,),
            parameter_identities=(f"formal-parameter-{index}",),
        )
        for index in range(count)
    )


def _learner_ids(count: int) -> tuple[str, ...]:
    return tuple(f"learner-{index:02d}" for index in range(count))


def _identities(run_id: str, config_identity: str) -> GlobalStateIdentities:
    return GlobalStateIdentities(
        run_identity=run_id,
        config_identity=config_identity,
        model_identity=_identity("s1-11-deterministic-byte-model"),
        fragment_map_identity=_identity("s1-11-two-fragment-map"),
    )


def _policy_identity(config: Mapping[str, Any]) -> str:
    return canonical_digest(config["policy"])


def _stores(
    root: Path,
    *,
    run_id: str,
    config_identity: str,
    config: Mapping[str, Any],
) -> tuple[AtomicGlobalCommitStore, ProposalStore]:
    identities = _identities(run_id, config_identity)
    descriptors = _descriptors(int(config["fragment_count"]))
    learners = _learner_ids(int(config["learner_count"]))
    atomic = AtomicGlobalCommitStore(
        GlobalStateStore(
            PosixStorageBackend(root / "global"),
            identities=identities,
            descriptors=descriptors,
            s_max=0,
        ),
        learner_ids=learners,
        policy_identity=_policy_identity(config),
    )
    proposals = ProposalStore(
        PosixStorageBackend(root / "proposals"),
        identities=identities,
        descriptors=descriptors,
        learner_ids=learners,
        maximum_local_steps=1_000_000,
        maximum_processed_tokens=1_000_000_000,
    )
    return atomic, proposals


def _bootstrap(atomic: AtomicGlobalCommitStore) -> None:
    atomic.bootstrap(
        tuple(
            BootstrapFragment(
                descriptor=descriptor,
                parameters=hashlib.sha256(
                    f"bootstrap-parameters-{descriptor.index}".encode()
                ).digest(),
                outer_state=canonical_bytes(
                    {"update_count": 0, "fragment_index": descriptor.index}
                ),
            )
            for descriptor in atomic.store.descriptors
        )
    )


def _proposal(
    authority: AtomicFragmentAuthority,
    learner_id: str,
    *,
    sequence: int,
    tokens: int,
    suffix: str = "main",
) -> Proposal:
    return Proposal.create(
        proposal_id=(
            f"{suffix}-{learner_id}-fragment-{authority.state.descriptor.index}"
            f"-base-{authority.version}-sequence-{sequence}"
        ),
        identities=authority.state.identities,
        learner_id=learner_id,
        descriptor=authority.state.descriptor,
        sequence=sequence,
        base_version=authority.version,
        base_content_identity=authority.content_identity,
        local_steps=sequence + 1,
        processed_tokens=tokens,
        snapshot_local_step=sequence + 1,
        parameters=hashlib.sha256(
            f"proposal:{suffix}:{learner_id}:{authority.state.descriptor.index}:"
            f"{authority.version}:{sequence}".encode()
        ).digest(),
    )


def _selection(
    authority: AtomicFragmentAuthority,
    proposals: tuple[Proposal, ...],
    *,
    generation: int,
) -> FrozenSelection:
    weights = compute_candidate_weights(
        proposals, current_version=authority.version, lambda_s=1.0
    )
    semantic = {
        "schema_version": 1,
        "logical_syncer_id": "formal-syncer-0",
        "fragment_index": authority.state.descriptor.index,
        "current_version": authority.version,
        "authority_identity": authority.authority_identity,
        "proposal_content_identities": [item.content_identity for item in proposals],
        "weights": [item.to_dict() for item in weights],
    }
    return FrozenSelection(
        fragment_index=authority.state.descriptor.index,
        current_version=authority.version,
        authority_identity=authority.authority_identity,
        generation=generation,
        grace_started_ns=generation,
        frozen_observed_ns=generation,
        proposals=proposals,
        weights=weights,
        selection_identity=canonical_digest(semantic),
    )


def _request(
    authority: AtomicFragmentAuthority,
    proposals: tuple[Proposal, ...],
    *,
    generation: int,
    suffix: str,
    policy_identity: str,
) -> AtomicCommitRequest:
    selection = _selection(authority, proposals, generation=generation)
    parameters = hashlib.sha256(
        f"successor:{suffix}:{authority.state.descriptor.index}:{authority.version + 1}".encode()
    ).digest()
    outer = canonical_bytes(
        {
            "update_count": authority.version + 1,
            "fragment_index": authority.state.descriptor.index,
            "selection_identity": selection.selection_identity,
        }
    )
    update_identity = canonical_digest(
        {
            "schema_version": 1,
            "current_content_identity": authority.content_identity,
            "next_version": authority.version + 1,
            "parameters_sha256": hashlib.sha256(parameters).hexdigest(),
            "outer_optimizer_state_sha256": hashlib.sha256(outer).hexdigest(),
            "selection_identity": selection.selection_identity,
            "policy_identity": policy_identity,
        }
    )
    return AtomicCommitRequest(
        fragment_index=authority.state.descriptor.index,
        expected_current_version=authority.version,
        expected_current_content_identity=authority.content_identity,
        next_version=authority.version + 1,
        parameters=parameters,
        outer_optimizer_state=outer,
        selection=selection,
        policy_identity=policy_identity,
        update_identity=update_identity,
    )


def _frontier_rows(authority: AtomicFragmentAuthority) -> list[dict[str, Any]]:
    return [
        {
            "learner_id": learner,
            "last_sequence": frontier.last_sequence,
            "last_base_version": frontier.last_base_version,
        }
        for learner, frontier in zip(
            authority.frontiers.learner_ids,
            authority.frontiers.entries,
            strict=True,
        )
    ]


def _authority_summary(authority: AtomicFragmentAuthority) -> dict[str, Any]:
    value = {
        "fragment_index": authority.state.descriptor.index,
        "version": authority.version,
        "outer_update_count": authority.state.outer_update_count,
        "authority_identity": authority.authority_identity,
        "content_identity": authority.content_identity,
        "base_content_identity": authority.state.base_content_identity,
        "parameters_sha256": hashlib.sha256(authority.parameters).hexdigest(),
        "outer_optimizer_state_sha256": hashlib.sha256(
            authority.outer_optimizer_state
        ).hexdigest(),
        "policy_identity": authority.envelope.policy_identity,
        "previous_authority_identity": authority.envelope.previous_authority_identity,
        "selection_identity": authority.envelope.selection_identity,
        "update_identity": authority.envelope.update_identity,
        "envelope_identity": authority.envelope.envelope_identity,
        "frontiers": _frontier_rows(authority),
        "selected": [item.to_dict() for item in authority.envelope.selected],
    }
    return {**value, "authority_fingerprint": canonical_digest(value)}


def _request_summary(request: AtomicCommitRequest) -> dict[str, Any]:
    return {
        "fragment_index": request.fragment_index,
        "expected_current_version": request.expected_current_version,
        "expected_current_content_identity": request.expected_current_content_identity,
        "next_version": request.next_version,
        "parameters_sha256": hashlib.sha256(request.parameters).hexdigest(),
        "outer_optimizer_state_sha256": hashlib.sha256(
            request.outer_optimizer_state
        ).hexdigest(),
        "policy_identity": request.policy_identity,
        "selection_identity": request.selection.selection_identity,
        "update_identity": request.update_identity,
        "selected": [
            {
                "proposal_id": proposal.proposal_id,
                "content_identity": proposal.content_identity,
                "learner_id": proposal.learner_id,
                "sequence": proposal.sequence,
                "base_version": proposal.base_version,
                "base_content_identity": proposal.base_content_identity,
                "processed_tokens": proposal.processed_tokens,
                "staleness": weight.staleness,
                "normalized_weight": weight.normalized_weight,
                "parameters_sha256": proposal.parameters_sha256,
                "payload_bytes": proposal.payload_bytes,
            }
            for proposal, weight in zip(
                request.selection.proposals, request.selection.weights, strict=True
            )
        ],
    }


def _eligible(authority: AtomicFragmentAuthority, proposal: Proposal) -> bool:
    result = select_candidates(
        (proposal,),
        authority.frontiers,
        EligibilityPolicy(
            identities=authority.state.identities,
            descriptor=authority.state.descriptor,
            learner_ids=authority.frontiers.learner_ids,
            current_version=authority.version,
            retained_bases=(
                RetainedBaseIdentity(authority.version, authority.content_identity),
            ),
            s_max=0,
            q=1,
            q_fresh=1,
            max_contributors=1,
            maximum_local_steps=1_000_000,
            maximum_processed_tokens=1_000_000_000,
        ),
    )
    return result.ready


def _fault_matrix(
    root: Path,
    *,
    run_id: str,
    config_identity: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    traces = []
    repeats = int(config["formal_workload"]["fault_replays_per_point"])
    for crash_at in config["publication_fault_points"]:
        for replay in range(repeats):
            case_root = root / f"{crash_at}-{replay:02d}"
            atomic, _ = _stores(
                case_root,
                run_id=f"{run_id}-fault-{crash_at}-{replay}",
                config_identity=config_identity,
                config=config,
            )
            _bootstrap(atomic)
            before = atomic.load_fragment(0)
            proposals = tuple(
                _proposal(
                    before,
                    learner,
                    sequence=0,
                    tokens=100 + index,
                    suffix=f"fault-{crash_at}-{replay}",
                )
                for index, learner in enumerate(atomic.learner_ids)
            )
            request = _request(
                before,
                proposals,
                generation=1,
                suffix=f"fault-{crash_at}-{replay}",
                policy_identity=atomic.policy_identity,
            )
            try:
                atomic.commit(request, crash_at=crash_at)
            except PublicationInterrupted as error:
                interruption = str(error)
            else:
                raise GlobalCommitStressError(
                    "fault injection did not interrupt publication"
                )
            observed = atomic.load_fragment(0)
            eligible = all(_eligible(observed, item) for item in proposals)
            replay_result = atomic.commit(request)
            final = atomic.load_fragment(0)
            traces.append(
                {
                    "crash_at": crash_at,
                    "replay": replay,
                    "interruption": interruption,
                    "before": _authority_summary(before),
                    "observed": _authority_summary(observed),
                    "selected_eligible_after_interruption": eligible,
                    "replay_published": replay_result.published,
                    "replay_duplicate": replay_result.duplicate_retry,
                    "final": _authority_summary(final),
                    "request": _request_summary(request),
                }
            )
    return traces


def _late_arrival_trace(
    root: Path,
    *,
    run_id: str,
    config_identity: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    atomic, _ = _stores(
        root,
        run_id=f"{run_id}-late",
        config_identity=config_identity,
        config=config,
    )
    _bootstrap(atomic)
    before = atomic.load_fragment(0)
    selected = tuple(
        _proposal(before, learner, sequence=0, tokens=100 + index, suffix="frozen")
        for index, learner in enumerate(atomic.learner_ids[:2])
    )
    request = _request(
        before,
        selected,
        generation=1,
        suffix="late-first",
        policy_identity=atomic.policy_identity,
    )
    late: list[Proposal] = []

    def arrive(_successor) -> None:
        late.append(
            _proposal(
                before,
                atomic.learner_ids[2],
                sequence=0,
                tokens=999,
                suffix="late-old-base",
            )
        )

    first = atomic.commit(request, before_visibility=arrive).authority
    if len(late) != 1:
        raise GlobalCommitStressError("late-arrival hook did not execute exactly once")
    refreshed = _proposal(
        first,
        atomic.learner_ids[2],
        sequence=1,
        tokens=999,
        suffix="late-refreshed",
    )
    next_request = _request(
        first,
        (refreshed,),
        generation=2,
        suffix="late-second",
        policy_identity=atomic.policy_identity,
    )
    second = atomic.commit(next_request).authority
    return {
        "before": _authority_summary(before),
        "first_request": _request_summary(request),
        "first": _authority_summary(first),
        "second_request": _request_summary(next_request),
        "second": _authority_summary(second),
        "frozen_proposal_ids": [item.proposal_id for item in selected],
        "late_old_base_proposal_id": late[0].proposal_id,
        "late_old_base_eligible_after_first": _eligible(first, late[0]),
        "refreshed_proposal_id": refreshed.proposal_id,
        "refreshed_selected_in_second": refreshed.proposal_id
        in {item.proposal_id for item in second.envelope.selected},
    }


def _writer_reader_role(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    config_identity: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    timeout = float(config["coordination_timeout_seconds"])
    coordination = root / "coordination"
    _wait_json(coordination / "ready.json", timeout)
    atomic, proposal_store = _stores(
        root / "main",
        run_id=run_id,
        config_identity=config_identity,
        config=config,
    )
    thread_count = int(config["formal_workload"]["concurrent_reader_threads"])
    minimum = int(config["formal_workload"]["minimum_reader_observations"])
    maximum = max(minimum, 1_000)
    cycles = int(config["formal_workload"]["successful_updates"])
    fragment_count = len(atomic.store.descriptors)
    stop = threading.Event()
    reader_errors: list[str] = []
    observations: list[dict[str, Any]] = []
    counts = [0] * thread_count
    coverage: list[set[tuple[int, int]]] = [set() for _ in range(thread_count)]
    initial_ready = [threading.Event() for _ in range(thread_count)]
    final_ready = [threading.Event() for _ in range(thread_count)]
    lock = threading.Lock()

    def reader(reader_index: int) -> None:
        local: list[dict[str, Any]] = []
        try:
            cursor = reader_index % fragment_count
            while not stop.is_set() or len(local) < minimum:
                if len(local) >= maximum:
                    stop.wait(0.01)
                authority = atomic.load_fragment(cursor)
                summary = _authority_summary(authority)
                local.append(
                    {
                        "reader_index": reader_index,
                        "fragment_index": cursor,
                        "version": summary["version"],
                        "authority_fingerprint": summary["authority_fingerprint"],
                    }
                )
                with lock:
                    coverage[reader_index].add((cursor, authority.version))
                    if all(
                        (fragment, 0) in coverage[reader_index]
                        for fragment in range(fragment_count)
                    ):
                        initial_ready[reader_index].set()
                    if all(
                        (fragment, cycles) in coverage[reader_index]
                        for fragment in range(fragment_count)
                    ):
                        final_ready[reader_index].set()
                cursor = (cursor + 1) % fragment_count
        except Exception as error:  # surfaced in the formal role record
            reader_errors.append(f"{type(error).__name__}: {error}")
        finally:
            with lock:
                counts[reader_index] = len(local)
                observations.extend(local)

    threads = [
        threading.Thread(target=reader, args=(index,), name=f"reader-{index}")
        for index in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    published = 0
    try:
        for ready in initial_ready:
            if not ready.wait(timeout):
                raise TimeoutError("reader did not observe every bootstrap authority")
        for cycle in range(cycles):
            for fragment_index in range(fragment_count):
                authority = atomic.load_fragment(fragment_index)
                if authority.version != cycle:
                    raise GlobalCommitStressError(
                        "writer observed an unexpected cycle version"
                    )
                for learner_index, learner_id in enumerate(proposal_store.learner_ids):
                    proposal_store.publish(
                        _proposal(
                            authority,
                            learner_id,
                            sequence=cycle,
                            tokens=(learner_index + 1) * 100 + cycle,
                            suffix="main",
                        )
                    )
                    published += 1
            _replace_json(
                coordination / f"proposals-{cycle:03d}.json",
                {"complete": True, "cycle": cycle, "published": published},
            )
            _wait_json(coordination / f"committed-{cycle:03d}.json", timeout)
        _wait_json(coordination / "updates-complete.json", timeout)
        for ready in final_ready:
            if not ready.wait(timeout):
                raise TimeoutError("reader did not span every successful replacement")
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=timeout)
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("concurrent authority reader did not stop")
    _replace_json(coordination / "writer-done.json", {"complete": True})
    _wait_json(coordination / "main-complete.json", timeout)
    result = {
        "schema_version": 1,
        "complete": True,
        "role": "proposal_writer_reader",
        "rank": 1,
        "size": 2,
        "hostname": socket.gethostname().split(".")[0],
        "pid": os.getpid(),
        "run_id": run_id,
        "config_identity": config_identity,
        "application_coordination": "filesystem_only",
        "mpi_usage": "launcher_only",
        "torch_imported_before": "torch" in sys.modules,
        "torch_imported_after": "torch" in sys.modules,
        "published_proposals": published,
        "reader_thread_count": thread_count,
        "reader_observation_counts": counts,
        "reader_errors": reader_errors,
        "observations": observations,
        "final_versions": list(atomic.load_snapshot().version_vector),
    }
    _write_json_new(result_root / "proposal_writer_reader.json", result)
    return result


def _committer_role(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    config_identity: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    timeout = float(config["coordination_timeout_seconds"])
    coordination = root / "coordination"
    atomic, proposal_store = _stores(
        root / "main",
        run_id=run_id,
        config_identity=config_identity,
        config=config,
    )
    _bootstrap(atomic)
    initial = [_authority_summary(item) for item in atomic.load_snapshot().authorities]
    _replace_json(coordination / "ready.json", {"complete": True})
    traces: list[dict[str, Any]] = []
    last_request: AtomicCommitRequest | None = None
    cycles = int(config["formal_workload"]["successful_updates"])
    for cycle in range(cycles):
        _wait_json(coordination / f"proposals-{cycle:03d}.json", timeout)
        latest = proposal_store.discover_latest(timeout_seconds=0)
        for fragment_index in range(len(atomic.store.descriptors)):
            before = atomic.load_fragment(fragment_index)
            candidates = tuple(
                item
                for item in latest
                if item.descriptor.index == fragment_index
                and item.base_version == before.version
            )
            if len(candidates) != len(atomic.learner_ids):
                raise GlobalCommitStressError(
                    "committer did not observe complete proposal quorum"
                )
            ordered = tuple(sorted(candidates, key=lambda item: item.learner_id))
            request = _request(
                before,
                ordered,
                generation=cycle + 1,
                suffix="main",
                policy_identity=atomic.policy_identity,
            )
            result = atomic.commit(request)
            after = result.authority
            traces.append(
                {
                    "cycle": cycle,
                    "fragment_index": fragment_index,
                    "before": _authority_summary(before),
                    "request": _request_summary(request),
                    "after": _authority_summary(after),
                    "published": result.published,
                    "duplicate_retry": result.duplicate_retry,
                }
            )
            last_request = request
        _replace_json(
            coordination / f"committed-{cycle:03d}.json",
            {"complete": True, "cycle": cycle},
        )
    if last_request is None:
        raise GlobalCommitStressError("formal workload performed no commit")
    duplicate = atomic.commit(last_request)
    retry_trace = {
        "request": _request_summary(last_request),
        "authority": _authority_summary(duplicate.authority),
        "published": duplicate.published,
        "duplicate_retry": duplicate.duplicate_retry,
    }
    _replace_json(coordination / "updates-complete.json", {"complete": True})
    _wait_json(coordination / "writer-done.json", timeout)
    faults = _fault_matrix(
        root / "faults",
        run_id=run_id,
        config_identity=config_identity,
        config=config,
    )
    late = _late_arrival_trace(
        root / "late",
        run_id=run_id,
        config_identity=config_identity,
        config=config,
    )
    final = atomic.load_snapshot()
    visibility = sorted(
        path.name for path in (root / "main" / "global" / "visibility").iterdir()
    )
    _replace_json(coordination / "main-complete.json", {"complete": True})
    result = {
        "schema_version": 1,
        "complete": True,
        "role": "atomic_committer",
        "rank": 0,
        "size": 2,
        "hostname": socket.gethostname().split(".")[0],
        "pid": os.getpid(),
        "run_id": run_id,
        "config_identity": config_identity,
        "application_coordination": "filesystem_only",
        "mpi_usage": "launcher_only",
        "torch_imported_before": "torch" in sys.modules,
        "torch_imported_after": "torch" in sys.modules,
        "initial_authorities": initial,
        "commit_traces": traces,
        "duplicate_retry": retry_trace,
        "fault_traces": faults,
        "late_arrival": late,
        "final_versions": list(final.version_vector),
        "global_visibility_records": visibility,
        "shared_global_head_present": any(
            "head" in item or "latest" in item for item in visibility
        ),
    }
    _write_json_new(result_root / "atomic_committer.json", result)
    return result


def run_role(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    config_path: Path,
    config_identity: str,
) -> dict[str, Any]:
    config = _load_config(config_path, config_identity)
    rank = _rank()
    if _size() != 2 or rank not in (0, 1):
        raise GlobalCommitStressError("formal stress requires exactly two ranks")
    result_root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        return _committer_role(
            root=root,
            result_root=result_root,
            run_id=run_id,
            config_identity=config_identity,
            config=config,
        )
    return _writer_reader_role(
        root=root,
        result_root=result_root,
        run_id=run_id,
        config_identity=config_identity,
        config=config,
    )


def build_analyzer_fixture(
    root: Path,
    config: Mapping[str, Any],
    config_identity: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a complete, deterministic two-role fixture without an MPI launcher."""

    run_id = "s1-11-analyzer-fixture"
    atomic, _ = _stores(
        root / "main",
        run_id=run_id,
        config_identity=config_identity,
        config=config,
    )
    _bootstrap(atomic)
    initial = [_authority_summary(item) for item in atomic.load_snapshot().authorities]
    traces: list[dict[str, Any]] = []
    allowed = list(initial)
    last_request: AtomicCommitRequest | None = None
    cycles = int(config["formal_workload"]["successful_updates"])
    for cycle in range(cycles):
        for fragment_index in range(len(atomic.store.descriptors)):
            before = atomic.load_fragment(fragment_index)
            proposals = tuple(
                _proposal(
                    before,
                    learner,
                    sequence=cycle,
                    tokens=(index + 1) * 100 + cycle,
                    suffix="fixture",
                )
                for index, learner in enumerate(atomic.learner_ids)
            )
            request = _request(
                before,
                proposals,
                generation=cycle + 1,
                suffix="fixture",
                policy_identity=atomic.policy_identity,
            )
            result = atomic.commit(request)
            after = _authority_summary(result.authority)
            allowed.append(after)
            traces.append(
                {
                    "cycle": cycle,
                    "fragment_index": fragment_index,
                    "before": _authority_summary(before),
                    "request": _request_summary(request),
                    "after": after,
                    "published": result.published,
                    "duplicate_retry": result.duplicate_retry,
                }
            )
            last_request = request
    if last_request is None:
        raise GlobalCommitStressError("analyzer fixture performed no commit")
    duplicate = atomic.commit(last_request)
    minimum = int(config["formal_workload"]["minimum_reader_observations"])
    thread_count = int(config["formal_workload"]["concurrent_reader_threads"])
    observations = [
        {
            "reader_index": reader,
            "fragment_index": allowed[index % len(allowed)]["fragment_index"],
            "version": allowed[index % len(allowed)]["version"],
            "authority_fingerprint": allowed[index % len(allowed)][
                "authority_fingerprint"
            ],
        }
        for reader in range(thread_count)
        for index in range(minimum)
    ]
    writer = {
        "schema_version": 1,
        "complete": True,
        "role": "proposal_writer_reader",
        "rank": 1,
        "size": 2,
        "hostname": "writer-host",
        "pid": 101,
        "run_id": run_id,
        "config_identity": config_identity,
        "application_coordination": "filesystem_only",
        "mpi_usage": "launcher_only",
        "torch_imported_before": False,
        "torch_imported_after": False,
        "published_proposals": cycles
        * len(atomic.store.descriptors)
        * len(atomic.learner_ids),
        "reader_thread_count": thread_count,
        "reader_observation_counts": [minimum] * thread_count,
        "reader_errors": [],
        "observations": observations,
        "final_versions": [cycles] * len(atomic.store.descriptors),
    }
    visibility = sorted(
        path.name for path in (root / "main" / "global" / "visibility").iterdir()
    )
    committer = {
        "schema_version": 1,
        "complete": True,
        "role": "atomic_committer",
        "rank": 0,
        "size": 2,
        "hostname": "committer-host",
        "pid": 100,
        "run_id": run_id,
        "config_identity": config_identity,
        "application_coordination": "filesystem_only",
        "mpi_usage": "launcher_only",
        "torch_imported_before": False,
        "torch_imported_after": False,
        "initial_authorities": initial,
        "commit_traces": traces,
        "duplicate_retry": {
            "request": _request_summary(last_request),
            "authority": _authority_summary(duplicate.authority),
            "published": duplicate.published,
            "duplicate_retry": duplicate.duplicate_retry,
        },
        "fault_traces": _fault_matrix(
            root / "faults",
            run_id=run_id,
            config_identity=config_identity,
            config=config,
        ),
        "late_arrival": _late_arrival_trace(
            root / "late",
            run_id=run_id,
            config_identity=config_identity,
            config=config,
        ),
        "final_versions": [cycles] * len(atomic.store.descriptors),
        "global_visibility_records": visibility,
        "shared_global_head_present": False,
    }
    return writer, committer


def _validate_authority_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GlobalCommitStressError("authority summary must be an object")
    fingerprint = value.get("authority_fingerprint")
    semantic = {
        key: item for key, item in value.items() if key != "authority_fingerprint"
    }
    if canonical_digest(semantic) != fingerprint:
        raise GlobalCommitStressError("authority summary fingerprint mismatch")
    for field in (
        "content_identity",
        "authority_identity",
        "base_content_identity",
        "parameters_sha256",
        "outer_optimizer_state_sha256",
        "policy_identity",
        "previous_authority_identity",
        "selection_identity",
        "update_identity",
        "envelope_identity",
        "authority_fingerprint",
    ):
        _require_hex(value.get(field), field)
    if value["version"] != value["outer_update_count"]:
        raise GlobalCommitStressError("outer update count differs from version")
    frontiers = value.get("frontiers")
    if not isinstance(frontiers, list) or len(frontiers) != 4:
        raise GlobalCommitStressError("authority frontier shape is not fixed")
    return value


def _validate_request_transition(
    before: dict[str, Any],
    request: Any,
    after: dict[str, Any],
    *,
    policy_identity: str,
    selected_count: int | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(request, dict):
        raise GlobalCommitStressError("commit request trace is malformed")
    if (
        request.get("fragment_index") != before["fragment_index"]
        or after["fragment_index"] != before["fragment_index"]
        or request.get("expected_current_version") != before["version"]
        or request.get("expected_current_content_identity")
        != before["content_identity"]
        or request.get("next_version") != before["version"] + 1
        or after["version"] != request.get("next_version")
        or after["base_content_identity"] != before["content_identity"]
        or after["previous_authority_identity"] != before["authority_identity"]
        or request.get("parameters_sha256") != after["parameters_sha256"]
        or request.get("outer_optimizer_state_sha256")
        != after["outer_optimizer_state_sha256"]
        or request.get("policy_identity") != policy_identity
        or after["policy_identity"] != policy_identity
        or request.get("selection_identity") != after["selection_identity"]
        or request.get("update_identity") != after["update_identity"]
    ):
        raise GlobalCommitStressError("request-to-authority transition differs")
    selected = request.get("selected")
    if (
        not isinstance(selected, list)
        or not selected
        or (selected_count is not None and len(selected) != selected_count)
        or any(not isinstance(item, dict) for item in selected)
        or len({item.get("proposal_id") for item in selected}) != len(selected)
        or len({item.get("learner_id") for item in selected}) != len(selected)
    ):
        raise GlobalCommitStressError("request selected facts are malformed")
    if abs(sum(float(item["normalized_weight"]) for item in selected) - 1.0) > 2e-6:
        raise GlobalCommitStressError("request selected weights do not sum to one")
    if any(
        item.get("base_version") != before["version"]
        or item.get("base_content_identity") != before["content_identity"]
        or item.get("staleness") != 0
        for item in selected
    ):
        raise GlobalCommitStressError("request selected base facts differ")
    if after.get("selected") != selected:
        raise GlobalCommitStressError("visible selected facts differ from request")
    expected_frontiers = {
        item["learner_id"]: (item["last_sequence"], item["last_base_version"])
        for item in before["frontiers"]
    }
    for item in selected:
        if item["learner_id"] not in expected_frontiers:
            raise GlobalCommitStressError("selected learner is absent from frontier")
        expected_frontiers[item["learner_id"]] = (
            item["sequence"],
            item["base_version"],
        )
    actual_frontiers = {
        item["learner_id"]: (item["last_sequence"], item["last_base_version"])
        for item in after["frontiers"]
    }
    if actual_frontiers != expected_frontiers:
        raise GlobalCommitStressError("visible consumption frontier differs")
    return selected


def analyze_roles(
    writer: Mapping[str, Any],
    committer: Mapping[str, Any],
    config: Mapping[str, Any],
    config_identity: str,
) -> dict[str, Any]:
    if (
        writer.get("role") != "proposal_writer_reader"
        or committer.get("role") != "atomic_committer"
    ):
        raise GlobalCommitStressError("formal role names differ")
    for role, rank in ((writer, 1), (committer, 0)):
        if (
            role.get("schema_version") != 1
            or role.get("complete") is not True
            or role.get("rank") != rank
            or role.get("size") != 2
            or role.get("config_identity") != config_identity
            or role.get("application_coordination") != "filesystem_only"
            or role.get("mpi_usage") != "launcher_only"
            or role.get("torch_imported_before") is not False
            or role.get("torch_imported_after") is not False
        ):
            raise GlobalCommitStressError(
                "formal role identity or CPU contract differs"
            )
    if writer.get("run_id") != committer.get("run_id"):
        raise GlobalCommitStressError("formal role run identities differ")
    if writer.get("hostname") == committer.get("hostname"):
        raise GlobalCommitStressError("formal roles must execute on distinct hosts")
    cycles = int(config["formal_workload"]["successful_updates"])
    fragments = int(config["fragment_count"])
    traces = committer.get("commit_traces")
    if not isinstance(traces, list) or len(traces) != cycles * fragments:
        raise GlobalCommitStressError("successful commit trace count differs")
    allowed: dict[str, tuple[int, int]] = {}
    initial = committer.get("initial_authorities")
    if not isinstance(initial, list) or len(initial) != fragments:
        raise GlobalCommitStressError("initial authority trace count differs")
    for authority in initial:
        authority = _validate_authority_summary(authority)
        if authority["version"] != 0 or authority["selected"]:
            raise GlobalCommitStressError(
                "bootstrap authority is not empty version zero"
            )
        allowed[authority["authority_fingerprint"]] = (
            authority["fragment_index"],
            authority["version"],
        )
    per_fragment = {index: [] for index in range(fragments)}
    policy_identity = _policy_identity(config)
    for trace in traces:
        if not isinstance(trace, dict):
            raise GlobalCommitStressError("commit trace is malformed")
        before = _validate_authority_summary(trace.get("before"))
        after = _validate_authority_summary(trace.get("after"))
        request = trace.get("request")
        if not isinstance(request, dict):
            raise GlobalCommitStressError("commit request trace is malformed")
        fragment = trace.get("fragment_index")
        cycle = trace.get("cycle")
        if fragment not in per_fragment or cycle != before["version"]:
            raise GlobalCommitStressError("commit trace cycle or fragment differs")
        if (
            trace.get("published") is not True
            or trace.get("duplicate_retry") is not False
        ):
            raise GlobalCommitStressError("successful commit result flags differ")
        selected = _validate_request_transition(
            before,
            request,
            after,
            policy_identity=policy_identity,
            selected_count=4,
        )
        if any(item["sequence"] != cycle for item in selected):
            raise GlobalCommitStressError(
                "selected proposal sequence differs from cycle"
            )
        allowed[after["authority_fingerprint"]] = (
            after["fragment_index"],
            after["version"],
        )
        per_fragment[fragment].append((before, after))
    for fragment, rows in per_fragment.items():
        if [row[1]["version"] for row in rows] != list(range(1, cycles + 1)):
            raise GlobalCommitStressError(
                f"fragment {fragment} version trace is not exact +1"
            )
        for previous, current in zip(rows, rows[1:], strict=False):
            if (
                previous[1]["authority_fingerprint"]
                != current[0]["authority_fingerprint"]
            ):
                raise GlobalCommitStressError(
                    "successive authority chain is discontinuous"
                )
    retry = committer.get("duplicate_retry")
    if (
        not isinstance(retry, dict)
        or retry.get("published") is not False
        or retry.get("duplicate_retry") is not True
    ):
        raise GlobalCommitStressError("exact duplicate retry contract differs")
    retry_authority = _validate_authority_summary(retry.get("authority"))
    last_before = _validate_authority_summary(traces[-1].get("before"))
    _validate_request_transition(
        last_before,
        retry.get("request"),
        retry_authority,
        policy_identity=policy_identity,
        selected_count=4,
    )
    if (
        retry_authority["authority_fingerprint"]
        != traces[-1]["after"]["authority_fingerprint"]
    ):
        raise GlobalCommitStressError("duplicate retry changed visible authority")
    fault_traces = committer.get("fault_traces")
    expected_faults = len(config["publication_fault_points"]) * int(
        config["formal_workload"]["fault_replays_per_point"]
    )
    if not isinstance(fault_traces, list) or len(fault_traces) != expected_faults:
        raise GlobalCommitStressError("fault trace count differs")
    fault_counts = {point: 0 for point in config["publication_fault_points"]}
    for trace in fault_traces:
        point = trace.get("crash_at")
        if point not in fault_counts:
            raise GlobalCommitStressError("unknown fault injection point")
        fault_counts[point] += 1
        before = _validate_authority_summary(trace.get("before"))
        observed = _validate_authority_summary(trace.get("observed"))
        final = _validate_authority_summary(trace.get("final"))
        _validate_request_transition(
            before,
            trace.get("request"),
            final,
            policy_identity=policy_identity,
            selected_count=4,
        )
        if point not in str(trace.get("interruption")):
            raise GlobalCommitStressError("fault interruption identity differs")
        if point == "after_record_replace":
            if (
                observed["authority_fingerprint"] != final["authority_fingerprint"]
                or trace.get("selected_eligible_after_interruption") is not False
                or trace.get("replay_published") is not False
                or trace.get("replay_duplicate") is not True
            ):
                raise GlobalCommitStressError(
                    "post-replace replay classification differs"
                )
        elif (
            observed["authority_fingerprint"] != before["authority_fingerprint"]
            or trace.get("selected_eligible_after_interruption") is not True
            or trace.get("replay_published") is not True
            or trace.get("replay_duplicate") is not False
        ):
            raise GlobalCommitStressError("pre-replace replay classification differs")
    if any(
        count != int(config["formal_workload"]["fault_replays_per_point"])
        for count in fault_counts.values()
    ):
        raise GlobalCommitStressError("fault replay distribution differs")
    late = committer.get("late_arrival")
    if not isinstance(late, dict):
        raise GlobalCommitStressError("late-arrival frozen selection contract differs")
    late_before = _validate_authority_summary(late.get("before"))
    late_first = _validate_authority_summary(late.get("first"))
    late_second = _validate_authority_summary(late.get("second"))
    first_selected = _validate_request_transition(
        late_before,
        late.get("first_request"),
        late_first,
        policy_identity=policy_identity,
        selected_count=2,
    )
    second_selected = _validate_request_transition(
        late_first,
        late.get("second_request"),
        late_second,
        policy_identity=policy_identity,
        selected_count=1,
    )
    frozen_ids = [item["proposal_id"] for item in first_selected]
    if (
        late.get("frozen_proposal_ids") != frozen_ids
        or late.get("late_old_base_proposal_id") in frozen_ids
        or late.get("late_old_base_eligible_after_first") is not False
        or late.get("refreshed_proposal_id") != second_selected[0]["proposal_id"]
        or late.get("refreshed_selected_in_second") is not True
    ):
        raise GlobalCommitStressError("late-arrival frozen selection contract differs")
    observations = writer.get("observations")
    counts = writer.get("reader_observation_counts")
    minimum = int(config["formal_workload"]["minimum_reader_observations"])
    observations_valid = isinstance(observations, list)
    coverage: dict[int, dict[int, set[int]]] = {
        reader: {fragment: set() for fragment in range(fragments)}
        for reader in range(int(config["formal_workload"]["concurrent_reader_threads"]))
    }
    if observations_valid:
        for item in observations:
            if not isinstance(item, dict):
                observations_valid = False
                break
            reader = item.get("reader_index")
            fragment = item.get("fragment_index")
            version = item.get("version")
            fingerprint = item.get("authority_fingerprint")
            if (
                reader not in coverage
                or fragment not in coverage[reader]
                or allowed.get(fingerprint) != (fragment, version)
            ):
                observations_valid = False
                break
            coverage[reader][fragment].add(version)
    spans_replacements = observations_valid and all(
        0 in coverage[reader][fragment]
        and cycles in coverage[reader][fragment]
        and len(coverage[reader][fragment]) >= 2
        for reader in coverage
        for fragment in coverage[reader]
    )
    if (
        writer.get("reader_thread_count")
        != int(config["formal_workload"]["concurrent_reader_threads"])
        or not isinstance(counts, list)
        or len(counts) != writer["reader_thread_count"]
        or any(count < minimum for count in counts)
        or writer.get("reader_errors") != []
        or not observations_valid
        or len(observations) != sum(counts)
        or not spans_replacements
    ):
        raise GlobalCommitStressError(
            "concurrent reader observations include mixed authority"
        )
    if (
        writer.get("final_versions") != [cycles] * fragments
        or committer.get("final_versions") != [cycles] * fragments
    ):
        raise GlobalCommitStressError("formal role final versions differ")
    expected_visibility = [
        f"global-current-{index:06d}.json" for index in range(fragments)
    ]
    if (
        committer.get("global_visibility_records") != expected_visibility
        or committer.get("shared_global_head_present") is not False
    ):
        raise GlobalCommitStressError(
            "fixed per-fragment current record contract differs"
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "run_id": writer["run_id"],
        "hosts": [writer["hostname"], committer["hostname"]],
        "successful_commits": len(traces),
        "final_versions": writer["final_versions"],
        "fault_replays": len(fault_traces),
        "fault_counts": fault_counts,
        "reader_threads": writer["reader_thread_count"],
        "reader_observations": len(observations),
        "minimum_reader_observations_per_thread": min(counts),
        "every_reader_spanned_bootstrap_to_final": True,
        "duplicate_retry": "same-authority-no-increment",
        "late_arrival": "frozen-then-refreshed-next-round",
        "visibility_records": expected_visibility,
        "application_coordination": "filesystem_only",
    }


def summarize(
    *, result_root: Path, config_path: Path, config_identity: str, output: Path
) -> dict[str, Any]:
    config = _load_config(config_path, config_identity)
    writer = json.loads(
        (result_root / "proposal_writer_reader.json").read_text(encoding="utf-8")
    )
    committer = json.loads(
        (result_root / "atomic_committer.json").read_text(encoding="utf-8")
    )
    summary = analyze_roles(writer, committer, config, config_identity)
    _write_json_new(output, summary)
    return summary


def manifest_command(args: argparse.Namespace) -> dict[str, Any]:
    writer = json.loads(
        (args.result_root / "proposal_writer_reader.json").read_text(encoding="utf-8")
    )
    committer = json.loads(
        (args.result_root / "atomic_committer.json").read_text(encoding="utf-8")
    )
    declared = {
        "proposal_writer_reader": [writer["hostname"]],
        "atomic_committer": [committer["hostname"]],
    }
    modules = [
        line.strip()
        for line in args.modules_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
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
        "skill": {
            "repository": args.skill_repository,
            "commit": args.skill_commit,
        },
        "execution": {
            "identity": args.run_id,
            "initial_hostname": args.initial_hostname,
            "compute_hostname": socket.gethostname().split(".")[0],
            "workflow": "two-node-filesystem-only-atomic-global-commit",
        },
        "roles": {"declared": declared, "actual": declared},
        "paths": {
            "project_root": str(args.project_root),
            "evidence_root": str(args.evidence_root),
        },
    }
    scheduler = {
        "job_id": args.job_id,
        "qtime_utc": args.qtime_utc,
        "queue": args.queue,
        "group": args.group,
        "nodefile_sha256": file_digest(args.nodefile),
        "modules": modules,
    }
    manifest = build_manifest(
        "S1-11", "L2", args.qtime_utc, "pbs_qtime", identities, scheduler=scheduler
    )
    _write_json_new(args.output, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S1-11 atomic global commit stress")
    sub = parser.add_subparsers(dest="command", required=True)
    role = sub.add_parser("role")
    role.add_argument("--root", type=Path, required=True)
    role.add_argument("--result-root", type=Path, required=True)
    role.add_argument("--run-id", required=True)
    role.add_argument("--config-path", type=Path, required=True)
    role.add_argument("--config-identity", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--result-root", type=Path, required=True)
    summary.add_argument("--config-path", type=Path, required=True)
    summary.add_argument("--config-identity", required=True)
    summary.add_argument("--output", type=Path, required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--repository", required=True)
    manifest.add_argument("--branch", required=True)
    manifest.add_argument("--commit", required=True)
    manifest.add_argument("--run-id", required=True)
    manifest.add_argument("--config-sha256", required=True)
    manifest.add_argument("--research-sha256", required=True)
    manifest.add_argument("--spec-sha256", required=True)
    manifest.add_argument("--skill-repository", required=True)
    manifest.add_argument("--skill-commit", required=True)
    manifest.add_argument("--initial-hostname", required=True)
    manifest.add_argument("--project-root", type=Path, required=True)
    manifest.add_argument("--evidence-root", type=Path, required=True)
    manifest.add_argument("--job-id", required=True)
    manifest.add_argument("--qtime-utc", required=True)
    manifest.add_argument("--queue", required=True)
    manifest.add_argument("--group", required=True)
    manifest.add_argument("--nodefile", type=Path, required=True)
    manifest.add_argument("--modules-file", type=Path, required=True)
    manifest.add_argument("--result-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "role":
        result = run_role(
            root=args.root,
            result_root=args.result_root,
            run_id=args.run_id,
            config_path=args.config_path,
            config_identity=args.config_identity,
        )
    elif args.command == "summarize":
        result = summarize(
            result_root=args.result_root,
            config_path=args.config_path,
            config_identity=args.config_identity,
            output=args.output,
        )
    else:
        result = manifest_command(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
