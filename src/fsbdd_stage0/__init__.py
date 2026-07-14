"""Reproducible Stage 0 research oracles, simulation, and storage probes."""

from .oracle import (
    CandidatePolicy,
    ConsumptionFrontier,
    OracleInputError,
    OuterSGDState,
    Proposal,
    commit_consumption,
    inverse_staleness_weights,
    outer_sgd_step,
    pseudo_gradient,
    select_proposals,
    validate_requirement_evidence,
    validated_pseudo_gradient,
    weighted_direct_merge,
)

__all__ = [
    "CandidatePolicy",
    "ConsumptionFrontier",
    "OracleInputError",
    "OuterSGDState",
    "Proposal",
    "commit_consumption",
    "inverse_staleness_weights",
    "outer_sgd_step",
    "pseudo_gradient",
    "select_proposals",
    "validate_requirement_evidence",
    "validated_pseudo_gradient",
    "weighted_direct_merge",
]
