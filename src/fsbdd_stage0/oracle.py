"""Independent scalar/small-vector reference math for FS-Decoupled DiLoCo.

This module deliberately has no dependency on future learner, syncer, storage, or
training code.  Operations are rounded through IEEE-754 binary32 so the oracle
matches the accumulation dtype frozen by ORACLE-01/02.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass


class OracleInputError(ValueError):
    """Raised when an oracle input violates the normative eligibility domain."""


def _f32(value: float) -> float:
    try:
        rounded = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, TypeError, ValueError) as error:
        raise OracleInputError(f"value is not representable as float32: {value!r}") from error
    if not math.isfinite(rounded):
        raise OracleInputError(f"value must be finite: {value!r}")
    return rounded


def _vector(name: str, values: Sequence[float]) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise OracleInputError(f"{name} must be a numeric sequence")
    if not values:
        raise OracleInputError(f"{name} must not be empty")
    return [_f32(value) for value in values]


def pseudo_gradient(base: Sequence[float], local: Sequence[float]) -> list[float]:
    """Return the normative base-relative displacement ``G_base - L_local``."""

    base_values = _vector("base", base)
    local_values = _vector("local", local)
    if len(base_values) != len(local_values):
        raise OracleInputError("base and local shapes differ")
    return [_f32(left - right) for left, right in zip(base_values, local_values, strict=True)]


def validated_pseudo_gradient(
    *,
    base: Sequence[float],
    local: Sequence[float],
    base_version: int,
    current_version: int,
    s_max: int,
    base_identity_matches: bool,
) -> list[float]:
    """Validate base eligibility, then compute a base-relative displacement."""

    if not base_identity_matches:
        raise OracleInputError("base content identity does not match")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (base_version, current_version, s_max)):
        raise OracleInputError("versions and s_max must be integers")
    if base_version < 0 or current_version < 0 or s_max < 0:
        raise OracleInputError("versions and s_max must be non-negative")
    staleness = current_version - base_version
    if staleness < 0:
        raise OracleInputError("future base version")
    if staleness > s_max:
        raise OracleInputError("base is too stale")
    return pseudo_gradient(base, local)


def inverse_staleness_weights(
    tokens: Sequence[float], staleness: Sequence[int], lambda_s: float
) -> list[float]:
    """Return normalized ``tokens / (1 + lambda_s * staleness)`` weights."""

    if isinstance(tokens, (str, bytes)) or isinstance(staleness, (str, bytes)):
        raise OracleInputError("tokens and staleness must be sequences")
    if len(tokens) == 0 or len(tokens) != len(staleness):
        raise OracleInputError("tokens and staleness must be non-empty and shape-matched")
    lambda_value = _f32(lambda_s)
    if lambda_value < 0:
        raise OracleInputError("lambda_s must be non-negative")

    raw: list[float] = []
    for token_value, stale_value in zip(tokens, staleness, strict=True):
        token = _f32(token_value)
        if token <= 0:
            raise OracleInputError("tokens must be positive")
        if isinstance(stale_value, bool) or not isinstance(stale_value, int) or stale_value < 0:
            raise OracleInputError("staleness must contain non-negative integers")
        denominator = _f32(1.0 + _f32(lambda_value * stale_value))
        raw.append(_f32(token / denominator))

    total = _f32(0.0)
    for value in raw:
        total = _f32(total + value)
    if total <= 0:
        raise OracleInputError("weight mass must be positive")
    return [_f32(value / total) for value in raw]


@dataclass(frozen=True, slots=True)
class Proposal:
    """Serializable proposal facts used by oracle and simulator decisions."""

    proposal_id: str
    learner_id: str
    sequence: int
    base_version: int
    tokens: int
    local_steps: int
    base_identity_matches: bool

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.learner_id:
            raise OracleInputError("proposal and learner identities must be non-empty")
        for name, value in (
            ("sequence", self.sequence),
            ("base_version", self.base_version),
            ("tokens", self.tokens),
            ("local_steps", self.local_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise OracleInputError(f"{name} must be an integer")
        if self.sequence < 0 or self.base_version < 0:
            raise OracleInputError("sequence and base_version must be non-negative")
        if self.tokens <= 0 or self.local_steps <= 0:
            raise OracleInputError("tokens and local_steps must be positive")
        if not isinstance(self.base_identity_matches, bool):
            raise OracleInputError("base_identity_matches must be boolean")


@dataclass(frozen=True, slots=True)
class ConsumptionFrontier:
    """Fixed-size once-only state for one learner/fragment pair."""

    last_sequence: int = -1
    last_base_version: int = -1

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < -1
            for value in (self.last_sequence, self.last_base_version)
        ):
            raise OracleInputError("consumption frontier values must be integers >= -1")


@dataclass(frozen=True, slots=True)
class CandidatePolicy:
    """Frozen decision policy for a single fragment version."""

    current_version: int
    s_max: int
    q: int
    q_fresh: int
    max_contributors: int
    lambda_s: float = 1.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.current_version,
            self.s_max,
            self.q,
            self.q_fresh,
            self.max_contributors,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise OracleInputError("candidate-policy count/version fields must be integers")
        if self.current_version < 0 or self.s_max < 0:
            raise OracleInputError("current_version and s_max must be non-negative")
        if not 0 <= self.q_fresh <= self.q <= self.max_contributors:
            raise OracleInputError("require 0 <= q_fresh <= q <= max_contributors")
        if self.q == 0:
            raise OracleInputError("q must be positive")
        if not math.isfinite(self.lambda_s) or self.lambda_s < 0:
            raise OracleInputError("lambda_s must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SelectionResult:
    ready: bool
    selected: tuple[Proposal, ...]
    rejections: dict[str, str]


def _eligibility_reason(
    proposal: Proposal,
    frontier: ConsumptionFrontier,
    policy: CandidatePolicy,
) -> str | None:
    if proposal.sequence <= frontier.last_sequence:
        return "consumed_sequence"
    if proposal.base_version <= frontier.last_base_version:
        return "consumed_base"
    if not proposal.base_identity_matches:
        return "base_identity_mismatch"
    staleness = policy.current_version - proposal.base_version
    if staleness < 0:
        return "future_base"
    if staleness > policy.s_max:
        return "too_stale"
    return None


def _candidate_key(proposal: Proposal, policy: CandidatePolicy) -> tuple[float | str, ...]:
    staleness = policy.current_version - proposal.base_version
    effective_tokens = proposal.tokens / (1.0 + policy.lambda_s * staleness)
    return (
        staleness,
        -effective_tokens,
        -proposal.sequence,
        proposal.learner_id,
        proposal.proposal_id,
    )


def select_proposals(
    proposals: Sequence[Proposal],
    frontiers: dict[str, ConsumptionFrontier],
    policy: CandidatePolicy,
) -> SelectionResult:
    """Apply eligibility, one-per-learner, quorum, and deterministic ordering."""

    rejections: dict[str, str] = {}
    eligible_by_learner: dict[str, list[Proposal]] = {}
    seen_ids: set[str] = set()
    for proposal in proposals:
        if proposal.proposal_id in seen_ids:
            raise OracleInputError(f"duplicate proposal identity: {proposal.proposal_id}")
        seen_ids.add(proposal.proposal_id)
        frontier = frontiers.get(proposal.learner_id, ConsumptionFrontier())
        reason = _eligibility_reason(proposal, frontier, policy)
        if reason is not None:
            rejections[proposal.proposal_id] = reason
            continue
        eligible_by_learner.setdefault(proposal.learner_id, []).append(proposal)

    distinct: list[Proposal] = []
    for learner_id in sorted(eligible_by_learner):
        ordered = sorted(
            eligible_by_learner[learner_id], key=lambda item: _candidate_key(item, policy)
        )
        distinct.append(ordered[0])
        for duplicate in ordered[1:]:
            rejections[duplicate.proposal_id] = "duplicate_learner"

    ordered_distinct = sorted(distinct, key=lambda item: _candidate_key(item, policy))
    fresh_count = sum(
        proposal.base_version == policy.current_version for proposal in ordered_distinct
    )
    ready = len(ordered_distinct) >= policy.q and fresh_count >= policy.q_fresh
    selected = (
        tuple(ordered_distinct[: policy.max_contributors]) if ready else tuple()
    )
    return SelectionResult(ready=ready, selected=selected, rejections=rejections)


def commit_consumption(
    frontiers: dict[str, ConsumptionFrontier],
    selected: Sequence[Proposal],
    publication_succeeded: bool,
) -> dict[str, ConsumptionFrontier]:
    """Advance fixed-size frontiers iff the corresponding global publish succeeded."""

    committed = dict(frontiers)
    if not publication_succeeded:
        return committed
    seen_learners: set[str] = set()
    for proposal in selected:
        if proposal.learner_id in seen_learners:
            raise OracleInputError("selected set contains a duplicate learner")
        seen_learners.add(proposal.learner_id)
        previous = committed.get(proposal.learner_id, ConsumptionFrontier())
        if proposal.sequence <= previous.last_sequence:
            raise OracleInputError("cannot commit a consumed proposal sequence")
        if proposal.base_version <= previous.last_base_version:
            raise OracleInputError("cannot commit an already consumed base")
        committed[proposal.learner_id] = ConsumptionFrontier(
            last_sequence=proposal.sequence,
            last_base_version=proposal.base_version,
        )
    return committed


def weighted_direct_merge(
    *,
    current: Sequence[float],
    bases: Sequence[Sequence[float]],
    locals_: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> list[float]:
    """Return ``sum_i w_i * (G_base_i - L_i)`` using float32 accumulation."""

    current_values = _vector("current", current)
    if not bases or len(bases) != len(locals_) or len(bases) != len(weights):
        raise OracleInputError("bases, locals, and weights must have equal non-zero length")
    rounded_weights = [_f32(weight) for weight in weights]
    if any(weight < 0 for weight in rounded_weights):
        raise OracleInputError("merge weights must be non-negative")
    weight_sum = _f32(0.0)
    for weight in rounded_weights:
        weight_sum = _f32(weight_sum + weight)
    if not math.isclose(weight_sum, 1.0, abs_tol=2e-6):
        raise OracleInputError("merge weights must sum to one")

    accumulator = [_f32(0.0) for _ in current_values]
    for base, local, weight in zip(bases, locals_, rounded_weights, strict=True):
        gradient = pseudo_gradient(base, local)
        if len(gradient) != len(current_values):
            raise OracleInputError("contribution shape differs from current parameters")
        for index, value in enumerate(gradient):
            accumulator[index] = _f32(accumulator[index] + _f32(weight * value))
    return accumulator


@dataclass(frozen=True, slots=True)
class OuterSGDState:
    momentum_buffer: list[float] | None


def outer_sgd_step(
    parameters: Sequence[float],
    gradient: Sequence[float],
    state: OuterSGDState,
    *,
    learning_rate: float,
    momentum: float,
    nesterov: bool,
) -> tuple[list[float], OuterSGDState]:
    """Reference PyTorch-style SGD/momentum/Nesterov parameter transition."""

    parameter_values = _vector("parameters", parameters)
    gradient_values = _vector("gradient", gradient)
    if len(parameter_values) != len(gradient_values):
        raise OracleInputError("parameter and gradient shapes differ")
    lr = _f32(learning_rate)
    mu = _f32(momentum)
    if lr <= 0 or not 0 <= mu < 1:
        raise OracleInputError("learning_rate must be positive and momentum in [0, 1)")
    if nesterov and mu <= 0:
        raise OracleInputError("Nesterov requires positive momentum")

    if mu == 0:
        if state.momentum_buffer is not None:
            raise OracleInputError("momentum-free SGD state must not contain a buffer")
        direction = gradient_values
        next_state = OuterSGDState(momentum_buffer=None)
    else:
        if state.momentum_buffer is None:
            buffer = list(gradient_values)
        else:
            old_buffer = _vector("momentum_buffer", state.momentum_buffer)
            if len(old_buffer) != len(gradient_values):
                raise OracleInputError("momentum-buffer shape differs from gradient")
            buffer = [
                _f32(_f32(mu * old) + grad)
                for old, grad in zip(old_buffer, gradient_values, strict=True)
            ]
        direction = (
            [
                _f32(grad + _f32(mu * buffered))
                for grad, buffered in zip(gradient_values, buffer, strict=True)
            ]
            if nesterov
            else buffer
        )
        next_state = OuterSGDState(momentum_buffer=buffer)

    next_parameters = [
        _f32(value - _f32(lr * update))
        for value, update in zip(parameter_values, direction, strict=True)
    ]
    return next_parameters, next_state


def validate_requirement_evidence(
    rows: Sequence[dict[str, str]], required_ids: set[str]
) -> None:
    """Reject missing, duplicate, or structurally empty traceability entries."""

    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        requirement_id = row.get("requirement_id", "").strip()
        if not requirement_id:
            raise OracleInputError("traceability row has no requirement_id")
        if requirement_id in indexed:
            raise OracleInputError(f"duplicate traceability row: {requirement_id}")
        indexed[requirement_id] = row
    missing = sorted(required_ids - indexed.keys())
    if missing:
        raise OracleInputError(f"requirements have no evidence mapping: {missing}")
    for requirement_id in sorted(required_ids):
        row = indexed[requirement_id]
        for field in ("primary_loop", "evidence", "status"):
            if not row.get(field, "").strip():
                raise OracleInputError(
                    f"traceability row {requirement_id} has empty {field}"
                )
