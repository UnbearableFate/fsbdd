"""Independent scalar/small-vector reference math for FS-Decoupled DiLoCo.

This module deliberately has no dependency on future learner, syncer, storage, or
training code.  Operations are rounded through IEEE-754 binary32 so the oracle
matches the accumulation dtype frozen by ORACLE-01/02.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence


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
