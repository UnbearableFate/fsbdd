from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import itertools
import struct
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from fsbdd.global_state import FragmentStateDescriptor
from fsbdd.identity import canonical_digest
from fsbdd.syncer_merge import (
    ContributionFact,
    DirectWeightedAverage,
    FragmentMergeRequest,
    FragmentOuterState,
    MergeError,
    MergePolicy,
    OuterSGDPolicy,
    ResolvedBase,
    build_update_facts,
    execute_numpy_streaming_fragment_update,
    execute_streaming_fragment_update,
)
from fsbdd_stage0.oracle import OuterSGDState, outer_sgd_step, weighted_direct_merge


MAP_ID = hashlib.sha256(b"s1-10-map").hexdigest()
CURRENT_ID = hashlib.sha256(b"s1-10-current").hexdigest()
OLD_ID = hashlib.sha256(b"s1-10-old").hexdigest()
SELECTION_ID = hashlib.sha256(b"s1-10-selection").hexdigest()
DESCRIPTOR = FragmentStateDescriptor(
    index=0,
    identity=hashlib.sha256(b"s1-10-fragment-0").hexdigest(),
    dtype="float32",
    shape=(2,),
    parameter_identities=("parameter-0", "parameter-1"),
)


def payload(values: Sequence[float]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def values(raw: bytes) -> list[float]:
    return np.frombuffer(raw, dtype="<f4").astype(np.float64).tolist()


class MemorySource:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.active = 0
        self.maximum_active = 0
        self.opens: list[str] = []
        self.full_model_reads = 0

    @contextlib.contextmanager
    def open_payload(self, contribution: ContributionFact) -> Iterator[bytes]:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.opens.append(contribution.proposal_id)
        try:
            yield self.payloads[contribution.proposal_id]
        finally:
            self.active -= 1

    def load_full_model(self) -> None:
        self.full_model_reads += 1
        raise AssertionError("steady merge requested a full model")


class MemoryBaseSource:
    def __init__(self, bases: dict[str, ResolvedBase]) -> None:
        self.bases = bases
        self.active = 0
        self.maximum_active = 0

    @contextlib.contextmanager
    def open_base(self, contribution: ContributionFact) -> Iterator[ResolvedBase]:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            yield self.bases[contribution.base_content_identity]
        finally:
            self.active -= 1


def make_fact(
    proposal_id: str,
    local: bytes,
    *,
    learner_id: str,
    weight: float,
    base_version: int = 0,
    base_identity: str = CURRENT_ID,
    current_version: int = 0,
    tokens: int = 1,
) -> ContributionFact:
    return ContributionFact(
        proposal_id=proposal_id,
        content_identity=hashlib.sha256(f"content:{proposal_id}".encode()).hexdigest(),
        learner_id=learner_id,
        base_version=base_version,
        base_content_identity=base_identity,
        processed_tokens=tokens,
        staleness=current_version - base_version,
        normalized_weight=weight,
        parameters_sha256=hashlib.sha256(local).hexdigest(),
        payload_bytes=len(local),
    )


def make_request(
    current: bytes,
    contributions: tuple[ContributionFact, ...],
    *,
    version: int = 0,
    current_identity: str = CURRENT_ID,
    outer_state: FragmentOuterState | None = None,
    descriptor: FragmentStateDescriptor | None = None,
) -> FragmentMergeRequest:
    resolved_descriptor = (
        dataclasses.replace(DESCRIPTOR, shape=(len(current) // 4,))
        if descriptor is None
        else descriptor
    )
    return FragmentMergeRequest(
        descriptor=resolved_descriptor,
        fragment_map_identity=MAP_ID,
        current_version=version,
        current_content_identity=current_identity,
        current_parameters=current,
        outer_state=(
            FragmentOuterState(update_count=version)
            if outer_state is None
            else outer_state
        ),
        selection_identity=SELECTION_ID,
        contributions=contributions,
    )


def assert_close(actual: bytes, expected: Sequence[float], tolerance: float = 1e-6) -> None:
    assert np.allclose(values(actual), expected, rtol=0.0, atol=tolerance)


def relative_l2(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return float(
        np.linalg.norm(left_array - right_array)
        / max(float(np.linalg.norm(right_array)), 1e-12)
    )


def test_direct_average_control_matches_stage0_oracle() -> None:
    current = payload([10.0, 0.0])
    local_a = payload([8.0, 4.0])
    local_b = payload([6.0, 2.0])
    facts = (
        make_fact("a", local_a, learner_id="learner-a", weight=0.25, tokens=1),
        make_fact("b", local_b, learner_id="learner-b", weight=0.75, tokens=3),
    )
    source = MemorySource({"a": local_a, "b": local_b})
    request = make_request(current, facts)
    policy = OuterSGDPolicy(learning_rate=1.0, momentum=0.0, nesterov=False)

    result = execute_streaming_fragment_update(request, source, policy)
    merged = weighted_direct_merge(
        current=[10.0, 0.0],
        bases=[[10.0, 0.0], [10.0, 0.0]],
        locals_=[[8.0, 4.0], [6.0, 2.0]],
        weights=[0.25, 0.75],
    )
    expected, expected_state = outer_sgd_step(
        [10.0, 0.0],
        merged,
        OuterSGDState(None),
        learning_rate=1.0,
        momentum=0.0,
        nesterov=False,
    )
    assert_close(result.merged_gradient, merged)
    assert_close(result.parameters, expected)
    assert expected_state.momentum_buffer is None
    assert result.outer_state == FragmentOuterState(update_count=1)
    assert source.opens == ["a", "b"]
    assert source.maximum_active == 1 and source.active == 0
    assert source.full_model_reads == 0
    assert result.byte_accounting.local_payload_reads == 2
    assert result.byte_accounting.local_payload_bytes == 2 * len(current)
    assert result.byte_accounting.full_model_operations == 0
    assert result.memory_accounting.maximum_active_local_payloads == 1
    assert result.memory_accounting.maximum_tensor_fragment_multiples <= 5


def test_cpu_only_numpy_profile_a_kernel_matches_torch_transition() -> None:
    current = payload([10.0, -2.0, 4.0, 8.0])
    locals_ = {
        "a": payload([9.0, -1.0, 5.0, 7.0]),
        "b": payload([8.0, -4.0, 3.0, 6.0]),
        "c": payload([11.0, -3.0, 2.0, 9.0]),
        "d": payload([7.0, 0.0, 6.0, 5.0]),
    }
    facts = tuple(
        make_fact(
            name,
            local,
            learner_id=f"learner-{name}",
            weight=0.25,
            tokens=10,
        )
        for name, local in locals_.items()
    )
    request = make_request(
        current,
        facts,
        outer_state=FragmentOuterState(0, payload([0.5, -0.25, 0.75, -0.5])),
    )
    policy = OuterSGDPolicy(learning_rate=0.1, momentum=0.9, nesterov=True)
    torch_result = execute_streaming_fragment_update(
        request, MemorySource(locals_), policy
    )
    numpy_result = execute_numpy_streaming_fragment_update(
        request, MemorySource(locals_), policy
    )
    assert_close(numpy_result.parameters, values(torch_result.parameters))
    assert_close(numpy_result.merged_gradient, values(torch_result.merged_gradient))
    assert numpy_result.outer_state.update_count == torch_result.outer_state.update_count
    assert_close(
        numpy_result.outer_state.momentum_buffer,
        values(torch_result.outer_state.momentum_buffer),
    )
    assert numpy_result.update_identity == torch_result.update_identity
    assert numpy_result.byte_accounting == torch_result.byte_accounting


def test_declared_old_base_gradient_is_applied_to_current_state() -> None:
    current = payload([12.0, -2.0])
    stale_local = payload([1.0, 1.0])
    fresh_local = payload([8.0, -4.0])
    facts = (
        make_fact(
            "stale",
            stale_local,
            learner_id="learner-stale",
            weight=0.5,
            base_version=0,
            base_identity=OLD_ID,
            current_version=1,
            tokens=4,
        ),
        make_fact(
            "fresh",
            fresh_local,
            learner_id="learner-fresh",
            weight=0.5,
            base_version=1,
            base_identity=CURRENT_ID,
            current_version=1,
            tokens=4,
        ),
    )
    source = MemorySource({"stale": stale_local, "fresh": fresh_local})
    bases = MemoryBaseSource(
        {
            OLD_ID: ResolvedBase(
                0,
                OLD_ID,
                payload([3.0, 5.0]),
                hashlib.sha256(payload([3.0, 5.0])).hexdigest(),
            )
        }
    )
    request = make_request(current, facts, version=1)
    result = execute_streaming_fragment_update(
        request,
        source,
        OuterSGDPolicy(1.0, 0.0, False),
        base_source=bases,
    )
    assert_close(result.merged_gradient, [3.0, 3.0])
    assert_close(result.parameters, [9.0, -5.0])
    assert values(result.parameters) != [1.5, 2.5]
    assert result.byte_accounting.retained_base_reads == 1
    assert result.byte_accounting.retained_base_bytes == len(current)
    assert bases.maximum_active == 1 and bases.active == 0


def test_nesterov_uses_updated_buffer_and_matches_stage0_two_steps() -> None:
    parameters = [10.0, -5.0]
    oracle_state = OuterSGDState(None)
    production_state = FragmentOuterState(0)
    policy = OuterSGDPolicy(0.1, 0.9, True)
    expected_rows = [[9.62, -4.24], [9.648, -4.486]]
    for version, gradient in enumerate(([2.0, -4.0], [-1.0, 3.0])):
        current = payload(parameters)
        current_identity = hashlib.sha256(f"current-{version}".encode()).hexdigest()
        local = payload([left - right for left, right in zip(parameters, gradient, strict=True)])
        fact = make_fact(
            f"proposal-{version}",
            local,
            learner_id="learner-a",
            weight=1.0,
            base_version=version,
            base_identity=current_identity,
            current_version=version,
        )
        request = make_request(
            current,
            (fact,),
            version=version,
            current_identity=current_identity,
            outer_state=production_state,
        )
        result = execute_streaming_fragment_update(
            request, MemorySource({fact.proposal_id: local}), policy
        )
        parameters, oracle_state = outer_sgd_step(
            parameters,
            gradient,
            oracle_state,
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True,
        )
        assert_close(result.parameters, parameters)
        assert_close(result.parameters, expected_rows[version])
        assert result.outer_state.momentum_buffer is not None
        assert_close(result.outer_state.momentum_buffer, oracle_state.momentum_buffer)
        production_state = result.outer_state


def test_eight_contributors_stream_one_at_a_time_with_exact_bytes() -> None:
    current = payload([10.0, -2.0, 5.0, 1.0])
    locals_by_id = {
        f"p-{index}": payload([9.0 - index, -1.0, 4.0, float(index)])
        for index in range(8)
    }
    weights = [struct.unpack("!f", struct.pack("!f", 1.0 / 8.0))[0]] * 8
    facts = tuple(
        make_fact(
            proposal_id,
            local,
            learner_id=f"learner-{index}",
            weight=weights[index],
            tokens=index + 1,
        )
        for index, (proposal_id, local) in enumerate(locals_by_id.items())
    )
    source = MemorySource(locals_by_id)
    result = execute_streaming_fragment_update(
        make_request(current, facts), source, OuterSGDPolicy(1.0, 0.0, False)
    )
    assert source.maximum_active == 1 and source.active == 0
    assert result.memory_accounting.maximum_active_local_payloads == 1
    assert result.memory_accounting.maximum_live_tensor_bytes <= 5 * len(current)
    assert result.byte_accounting.local_payload_bytes == 8 * len(current)
    assert result.byte_accounting.current_input_bytes == len(current)
    assert result.byte_accounting.successor_output_bytes == len(current)


def test_contribution_order_is_within_float32_tolerance() -> None:
    current_values = [1.0, -4.0, 3.0, 0.5]
    current = payload(current_values)
    locals_by_id = {
        "a": payload([0.0, -3.0, 1.0, 0.0]),
        "b": payload([2.0, -8.0, 4.0, 1.0]),
        "c": payload([-1.0, 2.0, 2.0, -0.5]),
    }
    weights = {"a": 0.2, "b": 0.3, "c": 0.5}
    outputs = []
    for order in itertools.permutations(locals_by_id):
        facts = tuple(
            make_fact(
                key,
                locals_by_id[key],
                learner_id=f"learner-{key}",
                weight=weights[key],
            )
            for key in order
        )
        result = execute_streaming_fragment_update(
            make_request(current, facts),
            MemorySource(locals_by_id),
            OuterSGDPolicy(1.0, 0.0, False),
        )
        outputs.append(values(result.parameters))
    canonical = outputs[0]
    assert max(relative_l2(item, canonical) for item in outputs) <= 1e-6


def test_fragment_outer_states_are_independent() -> None:
    descriptor_one = dataclasses.replace(
        DESCRIPTOR,
        index=1,
        identity=hashlib.sha256(b"s1-10-fragment-1").hexdigest(),
    )
    initial = FragmentOuterState(0, payload([4.0, -2.0]))
    local = payload([0.0, 0.0])
    fact = make_fact("p", local, learner_id="learner", weight=1.0)
    result = execute_streaming_fragment_update(
        make_request(
            payload([1.0, 1.0]),
            (fact,),
            outer_state=initial,
            descriptor=DESCRIPTOR,
        ),
        MemorySource({"p": local}),
        OuterSGDPolicy(0.1, 0.9, True),
    )
    untouched = FragmentOuterState(0, payload([-9.0, 7.0]))
    assert result.outer_state != initial
    assert untouched == FragmentOuterState(0, payload([-9.0, 7.0]))
    assert descriptor_one != DESCRIPTOR


def test_update_identity_binds_normative_facts() -> None:
    current = payload([2.0, 1.0])
    local_a = payload([1.0, 0.0])
    local_b = payload([0.0, 2.0])
    facts = (
        make_fact("a", local_a, learner_id="learner-a", weight=0.25, tokens=1),
        make_fact("b", local_b, learner_id="learner-b", weight=0.75, tokens=3),
    )
    request = make_request(current, facts)
    merge = DirectWeightedAverage()
    outer = OuterSGDPolicy(0.1, 0.9, True)
    original = canonical_digest(build_update_facts(request, merge, outer))
    advanced_facts = tuple(
        dataclasses.replace(item, staleness=1) for item in request.contributions
    )
    mutations = [
        dataclasses.replace(
            request,
            current_version=1,
            outer_state=FragmentOuterState(1),
            contributions=advanced_facts,
        ),
        dataclasses.replace(
            request,
            current_content_identity=hashlib.sha256(b"different-current").hexdigest(),
        ),
        dataclasses.replace(
            request,
            selection_identity=hashlib.sha256(b"different-selection").hexdigest(),
        ),
        dataclasses.replace(
            request,
            fragment_map_identity=hashlib.sha256(b"different-map").hexdigest(),
        ),
    ]
    for mutation in mutations:
        assert canonical_digest(build_update_facts(mutation, merge, outer)) != original
    changed_weights = (
        dataclasses.replace(facts[0], normalized_weight=0.5),
        dataclasses.replace(facts[1], normalized_weight=0.5),
    )
    assert canonical_digest(
        build_update_facts(dataclasses.replace(request, contributions=changed_weights), merge, outer)
    ) != original
    assert canonical_digest(build_update_facts(request, merge, OuterSGDPolicy(0.2, 0.9, True))) != original


class RecordingDirectPolicy(MergePolicy):
    name = "recording_direct"

    def __init__(self) -> None:
        self.calls = 0

    def identity_facts(self):
        return {"name": self.name, "formula": "direct"}

    def accumulate(self, accumulator, base, local, weight):
        self.calls += 1
        local.neg_().add_(base).mul_(weight)
        accumulator.add_(local)


def test_merge_policy_is_explicitly_pluggable_and_direct_is_default() -> None:
    current = payload([1.0, 2.0])
    local = payload([0.0, 1.0])
    fact = make_fact("p", local, learner_id="learner", weight=1.0)
    request = make_request(current, (fact,))
    direct = execute_streaming_fragment_update(
        request, MemorySource({"p": local}), OuterSGDPolicy(1.0, 0.0, False)
    )
    plugin = RecordingDirectPolicy()
    custom = execute_streaming_fragment_update(
        request,
        MemorySource({"p": local}),
        OuterSGDPolicy(1.0, 0.0, False),
        merge_policy=plugin,
    )
    assert direct.parameters == custom.parameters == local
    assert plugin.calls == 1
    assert direct.update_facts["merge_policy"]["name"] == "direct_weighted_average"
    assert custom.update_facts["merge_policy"]["name"] == "recording_direct"


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"normalized_weight": float("nan")}, "finite"),
        ({"payload_bytes": 7}, "divisible by four"),
        ({"staleness": 1}, "staleness differs"),
    ],
)
def test_invalid_contribution_facts_fail_closed(mutation, message) -> None:
    local = payload([0.0, 1.0])
    fact = make_fact("p", local, learner_id="learner", weight=1.0)
    if "staleness" in mutation:
        changed = dataclasses.replace(fact, **mutation)
        with pytest.raises(MergeError, match=message):
            make_request(payload([1.0, 2.0]), (changed,))
    else:
        with pytest.raises(MergeError, match=message):
            dataclasses.replace(fact, **mutation)


def test_checksum_dtype_duplicate_and_outer_state_guards_fail_closed() -> None:
    current = payload([1.0, 2.0])
    local = payload([0.0, 1.0])
    fact = make_fact("p", local, learner_id="learner", weight=1.0)
    bad_source = MemorySource({"p": payload([9.0, 9.0])})
    with pytest.raises(MergeError, match="checksum mismatch"):
        execute_streaming_fragment_update(
            make_request(current, (fact,)), bad_source, OuterSGDPolicy(1.0, 0.0, False)
        )
    with pytest.raises(MergeError, match="duplicate learner"):
        make_request(
            current,
            (
                dataclasses.replace(fact, proposal_id="p-a"),
                dataclasses.replace(fact, proposal_id="p-b"),
            ),
        )
    with pytest.raises(MergeError, match="float32 fragment"):
        make_request(current, (fact,), descriptor=dataclasses.replace(DESCRIPTOR, dtype="safetensors"))
    with pytest.raises(MergeError, match="descriptor shape differs"):
        make_request(
            current,
            (fact,),
            descriptor=dataclasses.replace(DESCRIPTOR, shape=(3,)),
        )
    with pytest.raises(MergeError, match="update_count"):
        make_request(current, (fact,), outer_state=FragmentOuterState(1))
    with pytest.raises(MergeError, match="momentum-free"):
        execute_streaming_fragment_update(
            make_request(current, (fact,), outer_state=FragmentOuterState(0, payload([0.0, 0.0]))),
            MemorySource({"p": local}),
            OuterSGDPolicy(1.0, 0.0, False),
        )
