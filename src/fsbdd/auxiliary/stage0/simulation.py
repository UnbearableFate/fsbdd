"""Bounded deterministic discrete-event simulator for Stage 0-A.

The simulator models protocol decisions and counters only.  It deliberately
imports the Stage 0-C candidate oracle so simulation and later runtime tests
share one decision semantic rather than two similar implementations.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

from .oracle import (
    CandidatePolicy,
    ConsumptionFrontier,
    OracleInputError,
    Proposal,
    commit_consumption,
    select_proposals,
)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    learners: int
    fragments: int
    q: int
    q_fresh: int
    s_max: int
    lambda_s: float
    h_steps: int
    grace_fraction_of_h: float
    upload_delay_seconds: float
    visibility_delay_seconds: float
    duration_seconds: float
    heterogeneity_ratio: float
    speed_model: str
    tokens_per_step: int
    seed: int
    max_trace_events: int

    def __post_init__(self) -> None:
        integer_fields = {
            "learners": self.learners,
            "fragments": self.fragments,
            "q": self.q,
            "q_fresh": self.q_fresh,
            "s_max": self.s_max,
            "h_steps": self.h_steps,
            "tokens_per_step": self.tokens_per_step,
            "seed": self.seed,
            "max_trace_events": self.max_trace_events,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise OracleInputError(f"{name} must be an integer")
        if self.learners <= 0 or self.fragments <= 0:
            raise OracleInputError("learners and fragments must be positive")
        if not 0 <= self.q_fresh <= self.q <= self.learners:
            raise OracleInputError("require 0 <= q_fresh <= q <= learners")
        if self.q == 0 or self.h_steps <= 0:
            raise OracleInputError("q and h_steps must be positive")
        if self.s_max not in {0, 1, 2}:
            raise OracleInputError("Stage 0 simulation supports s_max in {0, 1, 2}")
        if self.fragments > self.h_steps:
            raise OracleInputError(
                "fragments must not exceed H for unique staggered offsets"
            )
        if self.tokens_per_step <= 0 or self.max_trace_events < 0:
            raise OracleInputError(
                "tokens_per_step must be positive and trace cap non-negative"
            )
        numeric = {
            "lambda_s": self.lambda_s,
            "grace_fraction_of_h": self.grace_fraction_of_h,
            "upload_delay_seconds": self.upload_delay_seconds,
            "visibility_delay_seconds": self.visibility_delay_seconds,
            "duration_seconds": self.duration_seconds,
            "heterogeneity_ratio": self.heterogeneity_ratio,
        }
        if any(not math.isfinite(value) for value in numeric.values()):
            raise OracleInputError("simulation numeric configuration must be finite")
        if self.lambda_s < 0 or self.grace_fraction_of_h < 0:
            raise OracleInputError("lambda_s and grace must be non-negative")
        if self.upload_delay_seconds < 0 or self.visibility_delay_seconds < 0:
            raise OracleInputError("publication delays must be non-negative")
        if self.duration_seconds <= 0 or self.heterogeneity_ratio < 1:
            raise OracleInputError(
                "duration must be positive and heterogeneity_ratio >= 1"
            )
        if self.speed_model not in {"constant_ratio", "lognormal_jitter"}:
            raise OracleInputError("unsupported speed_model")

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "learners": self.learners,
            "fragments": self.fragments,
            "q": self.q,
            "q_fresh": self.q_fresh,
            "s_max": self.s_max,
            "lambda_s": self.lambda_s,
            "h_steps": self.h_steps,
            "grace_fraction_of_h": self.grace_fraction_of_h,
            "upload_delay_seconds": self.upload_delay_seconds,
            "visibility_delay_seconds": self.visibility_delay_seconds,
            "duration_seconds": self.duration_seconds,
            "heterogeneity_ratio": self.heterogeneity_ratio,
            "speed_model": self.speed_model,
            "tokens_per_step": self.tokens_per_step,
            "seed": self.seed,
            "max_trace_events": self.max_trace_events,
        }

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(
            (fragment * self.h_steps) // self.fragments
            for fragment in range(self.fragments)
        )

    @property
    def grace_seconds(self) -> float:
        return self.grace_fraction_of_h * self.h_steps

    @property
    def digest(self) -> str:
        canonical = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(order=True, slots=True)
class _Event:
    time: float
    priority: int
    identity: str
    kind: str = field(compare=False)
    payload: tuple[Any, ...] = field(compare=False)


@dataclass(slots=True)
class _IntervalDistribution:
    h_seconds: float
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = 0.0
    buckets: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        edges = (
            0.25 * self.h_seconds,
            0.5 * self.h_seconds,
            self.h_seconds,
            2.0 * self.h_seconds,
            4.0 * self.h_seconds,
        )
        index = next((i for i, edge in enumerate(edges) if value <= edge), 5)
        self.buckets[index] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.total / self.count if self.count else 0.0,
            "min": self.minimum if self.count else 0.0,
            "max": self.maximum if self.count else 0.0,
            "bucket_upper_h_multiples": [0.25, 0.5, 1.0, 2.0, 4.0, "inf"],
            "bucket_counts": list(self.buckets),
        }


@dataclass(frozen=True, slots=True)
class SimulationResult:
    config_digest: str
    processed_tokens: int
    token_opportunities: int
    accepted_tokens: int
    discarded_tokens: int
    accepted_token_efficiency: float
    token_weighted_discard_rate: float
    accepted_proposals: int
    stale_accepted_proposals: int
    stale_accepted_tokens: int
    stale_acceptance_rate: float
    rejection_counts: dict[str, int]
    update_counts: tuple[int, ...]
    global_cycle: int
    update_intervals: dict[str, Any]
    max_event_queue: int
    max_latest_slots: int
    max_frontier_slots: int
    max_rejection_tracking_slots: int
    max_accepted_tracking_slots: int
    trace: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_digest": self.config_digest,
            "processed_tokens": self.processed_tokens,
            "token_opportunities": self.token_opportunities,
            "accepted_tokens": self.accepted_tokens,
            "discarded_tokens": self.discarded_tokens,
            "accepted_token_efficiency": self.accepted_token_efficiency,
            "token_weighted_discard_rate": self.token_weighted_discard_rate,
            "accepted_proposals": self.accepted_proposals,
            "stale_accepted_proposals": self.stale_accepted_proposals,
            "stale_accepted_tokens": self.stale_accepted_tokens,
            "stale_acceptance_rate": self.stale_acceptance_rate,
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "update_counts": list(self.update_counts),
            "global_cycle": self.global_cycle,
            "update_intervals": self.update_intervals,
            "max_event_queue": self.max_event_queue,
            "max_latest_slots": self.max_latest_slots,
            "max_frontier_slots": self.max_frontier_slots,
            "max_rejection_tracking_slots": self.max_rejection_tracking_slots,
            "max_accepted_tracking_slots": self.max_accepted_tracking_slots,
            "trace": list(self.trace),
        }


def evaluate_decision_case(case: dict[str, Any], lambda_s: float) -> dict[str, Any]:
    """Evaluate one canonical Stage 0-C decision vector without reinterpretation."""

    proposals = [Proposal(**proposal) for proposal in case["proposals"]]
    frontiers = {
        learner: ConsumptionFrontier(**frontier)
        for learner, frontier in case["frontiers"].items()
    }
    policy = CandidatePolicy(
        current_version=case["current_version"],
        s_max=case["s_max"],
        q=case["q"],
        q_fresh=case["q_fresh"],
        max_contributors=case["max_contributors"],
        lambda_s=lambda_s,
    )
    result = select_proposals(proposals, frontiers, policy)
    return {
        "ready": result.ready,
        "selected": [proposal.proposal_id for proposal in result.selected],
        "rejections": dict(sorted(result.rejections.items())),
    }


class _Simulator:
    _PRIORITY = {"adopt_visible": 0, "step": 1, "proposal_visible": 2, "update": 3}

    def __init__(
        self, config: SimulationConfig, learner_order: tuple[int, ...] | None
    ) -> None:
        self.config = config
        order = learner_order or tuple(range(config.learners))
        if tuple(sorted(order)) != tuple(range(config.learners)):
            raise OracleInputError("learner_order must be a permutation of learner IDs")
        self.queue: list[_Event] = []
        self.rng = {
            learner: random.Random(config.seed * 1_000_003 + learner * 97)
            for learner in range(config.learners)
        }
        self.local_steps = [0] * config.learners
        self.adopted_versions = [[0] * config.fragments for _ in range(config.learners)]
        self.pending_adoptions = [
            [0] * config.fragments for _ in range(config.learners)
        ]
        self.tokens_since_adopt = [
            [0] * config.fragments for _ in range(config.learners)
        ]
        self.sequences = [[0] * config.fragments for _ in range(config.learners)]
        self.current_versions = [0] * config.fragments
        self.latest: dict[tuple[int, int], Proposal] = {}
        self.frontiers = [dict() for _ in range(config.fragments)]
        self.grace_deadlines: list[float | None] = [None] * config.fragments
        self.update_counts = [0] * config.fragments
        self.last_update_times: list[float | None] = [None] * config.fragments
        self.intervals = _IntervalDistribution(float(config.h_steps))
        self.accepted_tokens = 0
        self.discarded_tokens = 0
        self.accepted_proposals = 0
        self.stale_accepted_proposals = 0
        self.stale_accepted_tokens = 0
        self.last_accepted_by_slot: dict[tuple[int, int], str] = {}
        self.last_counted_rejection_by_slot: dict[tuple[int, int], str] = {}
        self.rejection_counts: dict[str, int] = {}
        self.trace: list[dict[str, Any]] = []
        self.max_event_queue = 0
        self.max_latest_slots = 0
        self.max_frontier_slots = 0
        for learner in order:
            first_time = self._step_interval(learner)
            self._push(first_time, "step", (learner,), f"l{learner:06d}")

    def _step_interval(self, learner: int) -> float:
        if self.config.learners == 1:
            slowdown = 1.0
        else:
            fraction = learner / (self.config.learners - 1)
            slowdown = 1.0 + (self.config.heterogeneity_ratio - 1.0) * fraction
        if self.config.speed_model == "constant_ratio":
            return slowdown
        sigma = 0.25
        jitter = self.rng[learner].lognormvariate(-0.5 * sigma * sigma, sigma)
        return slowdown * jitter

    def _push(
        self, time_value: float, kind: str, payload: tuple[Any, ...], identity: str
    ) -> None:
        if time_value > self.config.duration_seconds:
            return
        heapq.heappush(
            self.queue,
            _Event(time_value, self._PRIORITY[kind], identity, kind, payload),
        )
        self.max_event_queue = max(self.max_event_queue, len(self.queue))

    def _record(self, time_value: float, event: str, **fields: Any) -> None:
        if len(self.trace) >= self.config.max_trace_events:
            return
        self.trace.append({"time": round(time_value, 9), "event": event, **fields})

    def _publication_due(self, local_step: int, fragment: int) -> bool:
        offset = self.config.offsets[fragment]
        first = offset if offset > 0 else self.config.h_steps
        return local_step >= first and (local_step - offset) % self.config.h_steps == 0

    def _handle_step(self, time_value: float, learner: int) -> None:
        for fragment in range(self.config.fragments):
            pending = self.pending_adoptions[learner][fragment]
            if pending > self.adopted_versions[learner][fragment]:
                self.adopted_versions[learner][fragment] = pending
                self.tokens_since_adopt[learner][fragment] = 0
                self._record(
                    time_value,
                    "adopt",
                    learner=learner,
                    fragment=fragment,
                    version=pending,
                )

        self.local_steps[learner] += 1
        local_step = self.local_steps[learner]
        for fragment in range(self.config.fragments):
            self.tokens_since_adopt[learner][fragment] += self.config.tokens_per_step
            if not self._publication_due(local_step, fragment):
                continue
            self.sequences[learner][fragment] += 1
            sequence = self.sequences[learner][fragment]
            base_version = self.adopted_versions[learner][fragment]
            proposal = Proposal(
                proposal_id=f"l{learner}-f{fragment}-s{sequence}-b{base_version}",
                learner_id=str(learner),
                sequence=sequence,
                base_version=base_version,
                tokens=self.tokens_since_adopt[learner][fragment],
                local_steps=max(
                    1,
                    self.tokens_since_adopt[learner][fragment]
                    // self.config.tokens_per_step,
                ),
                base_identity_matches=True,
            )
            visible_time = (
                time_value
                + self.config.upload_delay_seconds
                + self.config.visibility_delay_seconds
            )
            identity = f"f{fragment:06d}:l{learner:06d}:s{sequence:012d}"
            self._push(
                visible_time,
                "proposal_visible",
                (learner, fragment, proposal),
                identity,
            )
            self._record(
                time_value,
                "snapshot",
                learner=learner,
                fragment=fragment,
                sequence=sequence,
                base_version=base_version,
            )

        next_time = time_value + self._step_interval(learner)
        self._push(next_time, "step", (learner,), f"l{learner:06d}")

    def _selection(self, fragment: int):
        proposals = [
            proposal
            for (learner, proposal_fragment), proposal in self.latest.items()
            if proposal_fragment == fragment
        ]
        policy = CandidatePolicy(
            current_version=self.current_versions[fragment],
            s_max=self.config.s_max,
            q=self.config.q,
            q_fresh=self.config.q_fresh,
            max_contributors=self.config.learners,
            lambda_s=self.config.lambda_s,
        )
        return select_proposals(proposals, self.frontiers[fragment], policy)

    def _count_rejections(self, result) -> None:
        visible_by_id = {
            proposal.proposal_id: (slot, proposal)
            for slot, proposal in self.latest.items()
        }
        for proposal_id, reason in result.rejections.items():
            visible = visible_by_id.get(proposal_id)
            if visible is None:
                continue
            slot, proposal = visible
            if self.last_counted_rejection_by_slot.get(slot) == proposal_id:
                continue
            self.last_counted_rejection_by_slot[slot] = proposal_id
            self.rejection_counts[reason] = self.rejection_counts.get(reason, 0) + 1
            if (
                self.last_accepted_by_slot.get(slot) != proposal_id
                and reason != "duplicate_learner"
            ):
                self.discarded_tokens += proposal.tokens

    def _evaluate_new_visibility(self, time_value: float, fragment: int) -> None:
        result = self._selection(fragment)
        self._count_rejections(result)
        if not result.ready or self.grace_deadlines[fragment] is not None:
            return
        if self.config.grace_seconds == 0:
            self._perform_update(time_value, fragment, result)
            return
        deadline = time_value + self.config.grace_seconds
        self.grace_deadlines[fragment] = deadline
        self._push(deadline, "update", (fragment, deadline), f"f{fragment:06d}")
        self._record(time_value, "grace_open", fragment=fragment, deadline=deadline)

    def _perform_update(self, time_value: float, fragment: int, result) -> None:
        current_version = self.current_versions[fragment]
        selected = result.selected
        self.frontiers[fragment] = commit_consumption(
            self.frontiers[fragment], selected, True
        )
        selected_tokens = sum(proposal.tokens for proposal in selected)
        stale = [
            proposal for proposal in selected if proposal.base_version < current_version
        ]
        self.accepted_tokens += selected_tokens
        self.accepted_proposals += len(selected)
        self.stale_accepted_proposals += len(stale)
        self.stale_accepted_tokens += sum(proposal.tokens for proposal in stale)
        selected_ids = {proposal.proposal_id for proposal in selected}
        for slot, proposal in self.latest.items():
            if slot[1] == fragment and proposal.proposal_id in selected_ids:
                self.last_accepted_by_slot[slot] = proposal.proposal_id
        self.current_versions[fragment] += 1
        self.update_counts[fragment] += 1
        previous_time = self.last_update_times[fragment]
        if previous_time is not None:
            self.intervals.add(time_value - previous_time)
        self.last_update_times[fragment] = time_value
        self.grace_deadlines[fragment] = None
        self._record(
            time_value,
            "update",
            fragment=fragment,
            old_version=current_version,
            new_version=self.current_versions[fragment],
            selected=[proposal.proposal_id for proposal in selected],
        )
        adoption_visible = time_value + self.config.visibility_delay_seconds
        new_version = self.current_versions[fragment]
        for learner in range(self.config.learners):
            self._push(
                adoption_visible,
                "adopt_visible",
                (learner, fragment, new_version),
                f"f{fragment:06d}:l{learner:06d}:v{new_version:012d}",
            )

    def _handle_proposal_visible(
        self, time_value: float, learner: int, fragment: int, proposal: Proposal
    ) -> None:
        slot = (learner, fragment)
        previous = self.latest.get(slot)
        if previous is not None and proposal.sequence <= previous.sequence:
            self._record(
                time_value,
                "old_completion_ignored",
                learner=learner,
                fragment=fragment,
                sequence=proposal.sequence,
            )
            return
        self.latest[slot] = proposal
        self.max_latest_slots = max(self.max_latest_slots, len(self.latest))
        self._record(
            time_value,
            "proposal_visible",
            learner=learner,
            fragment=fragment,
            sequence=proposal.sequence,
            base_version=proposal.base_version,
            current_version=self.current_versions[fragment],
        )
        self._evaluate_new_visibility(time_value, fragment)

    def run(self) -> SimulationResult:
        while self.queue:
            event = heapq.heappop(self.queue)
            if event.kind == "step":
                self._handle_step(event.time, event.payload[0])
            elif event.kind == "proposal_visible":
                self._handle_proposal_visible(event.time, *event.payload)
            elif event.kind == "adopt_visible":
                learner, fragment, version = event.payload
                self.pending_adoptions[learner][fragment] = max(
                    self.pending_adoptions[learner][fragment], version
                )
            elif event.kind == "update":
                fragment, deadline = event.payload
                if self.grace_deadlines[fragment] != deadline:
                    continue
                self.grace_deadlines[fragment] = None
                result = self._selection(fragment)
                self._count_rejections(result)
                if result.ready:
                    self._perform_update(event.time, fragment, result)
            else:
                raise AssertionError(f"unknown event kind: {event.kind}")
            self.max_frontier_slots = max(
                self.max_frontier_slots,
                sum(len(frontier) for frontier in self.frontiers),
            )

        processed_tokens = sum(self.local_steps) * self.config.tokens_per_step
        opportunities = processed_tokens * self.config.fragments
        accepted_efficiency = (
            self.accepted_tokens / opportunities if opportunities else 0.0
        )
        discard_mass = self.accepted_tokens + self.discarded_tokens
        discard_rate = self.discarded_tokens / discard_mass if discard_mass else 0.0
        stale_rate = (
            self.stale_accepted_tokens / self.accepted_tokens
            if self.accepted_tokens
            else 0.0
        )
        return SimulationResult(
            config_digest=self.config.digest,
            processed_tokens=processed_tokens,
            token_opportunities=opportunities,
            accepted_tokens=self.accepted_tokens,
            discarded_tokens=self.discarded_tokens,
            accepted_token_efficiency=accepted_efficiency,
            token_weighted_discard_rate=discard_rate,
            accepted_proposals=self.accepted_proposals,
            stale_accepted_proposals=self.stale_accepted_proposals,
            stale_accepted_tokens=self.stale_accepted_tokens,
            stale_acceptance_rate=stale_rate,
            rejection_counts=dict(sorted(self.rejection_counts.items())),
            update_counts=tuple(self.update_counts),
            global_cycle=min(self.update_counts),
            update_intervals=self.intervals.as_dict(),
            max_event_queue=self.max_event_queue,
            max_latest_slots=self.max_latest_slots,
            max_frontier_slots=self.max_frontier_slots,
            max_rejection_tracking_slots=len(self.last_counted_rejection_by_slot),
            max_accepted_tracking_slots=len(self.last_accepted_by_slot),
            trace=tuple(self.trace),
        )


def simulate(
    config: SimulationConfig, *, learner_order: tuple[int, ...] | None = None
) -> SimulationResult:
    """Run a finite deterministic simulation and return bounded summary state."""

    return _Simulator(config, learner_order).run()
