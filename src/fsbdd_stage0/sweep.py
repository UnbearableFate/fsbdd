"""Resumable SIM-03 sweep runner and deterministic report generator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .simulation import SimulationConfig, simulate


AXIS_ORDER = (
    "learners",
    "quorum_fraction",
    "heterogeneity_ratio",
    "grace_fraction_of_h",
    "visibility_delay_seconds",
    "s_max",
    "speed_model",
    "seed",
)
EXPECTED_SIM03_AXES = {
    "learners": [4, 8, 16],
    "quorum_fraction": [0.5, 0.75, 1.0],
    "heterogeneity_ratio": [1.0, 1.5, 2.0, 3.0],
    "grace_fraction_of_h": [0.0, 0.1, 0.25],
    "visibility_delay_seconds": [0.0, 1.0, 5.0, 30.0],
    "s_max": [0, 1, 2],
    "speed_model": ["constant_ratio", "lognormal_jitter"],
}
METRIC_FIELDS = (
    "accepted_token_efficiency",
    "token_weighted_discard_rate",
    "stale_acceptance_rate",
    "update_interval_mean",
    "global_cycle",
)
CSV_FIELDS = (
    "schema_version",
    "matrix_index",
    "row_key",
    "config_digest",
    "generator_commit",
    "generator_digest",
    *AXIS_ORDER,
    "fragments",
    "q",
    "q_fresh",
    "h_steps",
    "lambda_s",
    "upload_delay_seconds",
    "nominal_fastest_step_seconds",
    "duration_seconds",
    "tokens_per_step",
    "visibility_delay_over_h",
    "processed_tokens",
    "token_opportunities",
    "accepted_tokens",
    "discarded_tokens",
    "accepted_token_efficiency",
    "token_weighted_discard_rate",
    "accepted_proposals",
    "stale_accepted_proposals",
    "stale_accepted_tokens",
    "stale_acceptance_rate",
    "global_cycle",
    "update_interval_count",
    "update_interval_mean",
    "update_interval_min",
    "update_interval_max",
    "update_interval_buckets_json",
    "rejection_counts_json",
    "update_counts_json",
    "max_event_queue",
    "max_latest_slots",
    "max_frontier_slots",
    "max_rejection_tracking_slots",
    "max_accepted_tracking_slots",
    "row_digest",
)


class SweepInputError(ValueError):
    """Raised when a frozen sweep input or generated artifact is incomplete."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_sweep_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_sweep_config(config)
    return config


def validate_sweep_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise SweepInputError("sweep schema_version must be 1")
    axes = config.get("axes")
    if not isinstance(axes, dict) or set(axes) != set(AXIS_ORDER):
        raise SweepInputError("sweep axes must exactly match the frozen axis schema")
    for axis, expected in EXPECTED_SIM03_AXES.items():
        if axes.get(axis) != expected:
            raise SweepInputError(f"incomplete or reordered SIM-03 axis: {axis}")
    seeds = axes.get("seed")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise SweepInputError("at least three unique integer seeds are required")
    fixed = config.get("fixed", {})
    if fixed.get("fragments") != 4 or fixed.get("h_steps") != 50:
        raise SweepInputError("resolved Stage 0 scan requires F=4 and H=50")
    if fixed.get("nominal_fastest_step_seconds") != 1.0:
        raise SweepInputError("delay/H requires the explicit nominal 1s fastest step")
    if fixed.get("max_trace_events") != 0:
        raise SweepInputError("full-matrix rows must not retain event history")
    if not fixed.get("one_per_learner") or not fixed.get("same_base_once_only"):
        raise SweepInputError("oracle consumption semantics cannot be disabled")
    expected_rows = math.prod(len(axes[name]) for name in AXIS_ORDER)
    execution = config.get("execution", {})
    if execution.get("expected_rows") != expected_rows:
        raise SweepInputError("expected_rows does not equal the Cartesian product")
    if execution.get("shard_count", 0) <= 0 or execution.get("checkpoint_every_rows", 0) <= 0:
        raise SweepInputError("shard and checkpoint counts must be positive")
    required_outputs = {
        "token_weighted_discard_rate",
        "accepted_token_efficiency",
        "fragment_update_interval_distribution",
        "stale_acceptance_rate",
        "recovery_gain_smax1_vs_smax0",
    }
    if set(config.get("outputs", [])) != required_outputs:
        raise SweepInputError("SIM-04 output schema is incomplete")


def matrix_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    axes = config["axes"]
    fixed = config["fixed"]
    specs: list[dict[str, Any]] = []
    for values in itertools.product(*(axes[name] for name in AXIS_ORDER)):
        axis = dict(zip(AXIS_ORDER, values, strict=True))
        learners = int(axis["learners"])
        q_float = learners * float(axis["quorum_fraction"])
        q = int(q_float)
        if q != q_float:
            raise SweepInputError("quorum fraction does not produce an integer Q")
        spec = {
            **axis,
            "fragments": fixed["fragments"],
            "q": q,
            "q_fresh": fixed["q_fresh"],
            "h_steps": fixed["h_steps"],
            "lambda_s": fixed["lambda_s"],
            "upload_delay_seconds": fixed["upload_delay_seconds"],
            "nominal_fastest_step_seconds": fixed["nominal_fastest_step_seconds"],
            "duration_seconds": fixed["duration_seconds"],
            "tokens_per_step": fixed["tokens_per_step"],
            "max_trace_events": fixed["max_trace_events"],
        }
        specs.append(spec)
    if len(specs) != config["execution"]["expected_rows"]:
        raise SweepInputError("generated matrix cardinality mismatch")
    return specs


def spec_key(spec: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(spec).encode("utf-8"))


def _simulation_config(spec: dict[str, Any], *, max_trace_events: int | None = None) -> SimulationConfig:
    return SimulationConfig(
        learners=int(spec["learners"]),
        fragments=int(spec["fragments"]),
        q=int(spec["q"]),
        q_fresh=int(spec["q_fresh"]),
        s_max=int(spec["s_max"]),
        lambda_s=float(spec["lambda_s"]),
        h_steps=int(spec["h_steps"]),
        grace_fraction_of_h=float(spec["grace_fraction_of_h"]),
        upload_delay_seconds=float(spec["upload_delay_seconds"]),
        visibility_delay_seconds=float(spec["visibility_delay_seconds"]),
        duration_seconds=float(spec["duration_seconds"]),
        heterogeneity_ratio=float(spec["heterogeneity_ratio"]),
        speed_model=str(spec["speed_model"]),
        tokens_per_step=int(spec["tokens_per_step"]),
        seed=int(spec["seed"]),
        max_trace_events=(
            int(spec["max_trace_events"])
            if max_trace_events is None
            else max_trace_events
        ),
    )


def _row_integrity_digest(row: dict[str, Any]) -> str:
    normalized = {
        field: str(row[field])
        for field in CSV_FIELDS
        if field != "row_digest"
    }
    return _sha256_bytes(_canonical_json(normalized).encode("utf-8"))


def run_matrix_row(
    matrix_index: int,
    spec: dict[str, Any],
    *,
    generator_commit: str = "direct-test",
    generator_digest: str = "direct-test",
) -> dict[str, Any]:
    result = simulate(_simulation_config(spec))
    intervals = result.update_intervals
    row = {
        "schema_version": 1,
        "matrix_index": matrix_index,
        "row_key": spec_key(spec),
        "config_digest": result.config_digest,
        "generator_commit": generator_commit,
        "generator_digest": generator_digest,
        **{name: spec[name] for name in AXIS_ORDER},
        "fragments": spec["fragments"],
        "q": spec["q"],
        "q_fresh": spec["q_fresh"],
        "h_steps": spec["h_steps"],
        "lambda_s": spec["lambda_s"],
        "upload_delay_seconds": spec["upload_delay_seconds"],
        "nominal_fastest_step_seconds": spec["nominal_fastest_step_seconds"],
        "duration_seconds": spec["duration_seconds"],
        "tokens_per_step": spec["tokens_per_step"],
        "visibility_delay_over_h": float(spec["visibility_delay_seconds"]) / (
            float(spec["h_steps"]) * float(spec["nominal_fastest_step_seconds"])
        ),
        "processed_tokens": result.processed_tokens,
        "token_opportunities": result.token_opportunities,
        "accepted_tokens": result.accepted_tokens,
        "discarded_tokens": result.discarded_tokens,
        "accepted_token_efficiency": result.accepted_token_efficiency,
        "token_weighted_discard_rate": result.token_weighted_discard_rate,
        "accepted_proposals": result.accepted_proposals,
        "stale_accepted_proposals": result.stale_accepted_proposals,
        "stale_accepted_tokens": result.stale_accepted_tokens,
        "stale_acceptance_rate": result.stale_acceptance_rate,
        "global_cycle": result.global_cycle,
        "update_interval_count": intervals["count"],
        "update_interval_mean": intervals["mean"],
        "update_interval_min": intervals["min"],
        "update_interval_max": intervals["max"],
        "update_interval_buckets_json": _canonical_json(intervals),
        "rejection_counts_json": _canonical_json(result.rejection_counts),
        "update_counts_json": _canonical_json(result.update_counts),
        "max_event_queue": result.max_event_queue,
        "max_latest_slots": result.max_latest_slots,
        "max_frontier_slots": result.max_frontier_slots,
        "max_rejection_tracking_slots": result.max_rejection_tracking_slots,
        "max_accepted_tracking_slots": result.max_accepted_tracking_slots,
    }
    row["row_digest"] = _row_integrity_digest(row)
    return row


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise SweepInputError(f"unexpected CSV schema: {path}")
        return list(reader)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_retained_row(
    row: dict[str, str],
    spec: dict[str, Any],
    matrix_index: int,
    generator_commit: str,
    generator_digest: str,
) -> None:
    if int(row["schema_version"]) != 1 or int(row["matrix_index"]) != matrix_index:
        raise SweepInputError("retained row schema/index mismatch")
    if row["row_key"] != spec_key(spec):
        raise SweepInputError("retained row does not match the frozen matrix")
    if row["config_digest"] != _simulation_config(spec).digest:
        raise SweepInputError("retained row simulation config digest mismatch")
    if (
        row["generator_commit"] != generator_commit
        or row["generator_digest"] != generator_digest
    ):
        raise SweepInputError("retained row was produced by a different generator")
    if row["row_digest"] != _row_integrity_digest(row):
        raise SweepInputError("retained row integrity digest mismatch")
    expected_spec_fields = (
        *AXIS_ORDER,
        "fragments",
        "q",
        "q_fresh",
        "h_steps",
        "lambda_s",
        "upload_delay_seconds",
        "nominal_fastest_step_seconds",
        "duration_seconds",
        "tokens_per_step",
    )
    for field in expected_spec_fields:
        if row[field] != str(spec[field]):
            raise SweepInputError(f"retained row field differs from frozen spec: {field}")
    numeric_fields = (
        "visibility_delay_over_h",
        "accepted_token_efficiency",
        "token_weighted_discard_rate",
        "stale_acceptance_rate",
        "update_interval_mean",
        "update_interval_min",
        "update_interval_max",
    )
    if any(not math.isfinite(float(row[field])) for field in numeric_fields):
        raise SweepInputError("retained row contains a non-finite metric")
    processed = int(row["processed_tokens"])
    opportunities = int(row["token_opportunities"])
    accepted = int(row["accepted_tokens"])
    discarded = int(row["discarded_tokens"])
    stale_tokens = int(row["stale_accepted_tokens"])
    if min(processed, opportunities, accepted, discarded, stale_tokens) < 0:
        raise SweepInputError("retained token counters must be non-negative")
    if opportunities != processed * int(spec["fragments"]):
        raise SweepInputError("retained token opportunity unit mismatch")
    expected_efficiency = accepted / opportunities if opportunities else 0.0
    discard_mass = accepted + discarded
    expected_discard = discarded / discard_mass if discard_mass else 0.0
    expected_stale = stale_tokens / accepted if accepted else 0.0
    comparisons = (
        (float(row["accepted_token_efficiency"]), expected_efficiency),
        (float(row["token_weighted_discard_rate"]), expected_discard),
        (float(row["stale_acceptance_rate"]), expected_stale),
        (
            float(row["visibility_delay_over_h"]),
            float(spec["visibility_delay_seconds"])
            / (
                float(spec["h_steps"])
                * float(spec["nominal_fastest_step_seconds"])
            ),
        ),
    )
    if any(not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15) for actual, expected in comparisons):
        raise SweepInputError("retained derived metric failed recomputation")
    intervals = json.loads(row["update_interval_buckets_json"])
    if (
        intervals["count"] != int(row["update_interval_count"])
        or not math.isclose(intervals["mean"], float(row["update_interval_mean"]), abs_tol=1e-15)
        or sum(intervals["bucket_counts"]) != intervals["count"]
    ):
        raise SweepInputError("retained interval distribution is inconsistent")
    update_counts = json.loads(row["update_counts_json"])
    if min(update_counts) != int(row["global_cycle"]):
        raise SweepInputError("retained global_cycle is not the fragment minimum")
    bound = int(spec["learners"]) * int(spec["fragments"])
    bounded_fields = (
        "max_latest_slots",
        "max_frontier_slots",
        "max_rejection_tracking_slots",
        "max_accepted_tracking_slots",
    )
    if any(int(row[field]) > bound for field in bounded_fields):
        raise SweepInputError("retained operational state exceeds M*F")


def run_shard(
    config_path: Path,
    output_path: Path,
    shard_index: int,
    shard_count: int,
    generator_commit: str,
    generator_digest: str,
    max_new_rows: int | None = None,
) -> dict[str, int | bool]:
    config = load_sweep_config(config_path)
    if shard_count != config["execution"]["shard_count"]:
        raise SweepInputError("runtime shard_count differs from frozen config")
    if not 0 <= shard_index < shard_count:
        raise SweepInputError("shard_index is outside shard_count")
    specs = matrix_specs(config)
    existing: dict[int, dict[str, Any]] = {}
    if output_path.exists():
        for row in _read_csv(output_path):
            index = int(row["matrix_index"])
            if index in existing:
                raise SweepInputError("duplicate matrix index in resumable shard")
            if index % shard_count != shard_index:
                raise SweepInputError("row stored in the wrong shard")
            _validate_retained_row(
                row,
                specs[index],
                index,
                generator_commit,
                generator_digest,
            )
            existing[index] = row
    checkpoint_every = config["execution"]["checkpoint_every_rows"]
    completed_since_checkpoint = 0
    for index, spec in enumerate(specs):
        if index % shard_count != shard_index or index in existing:
            continue
        if max_new_rows is not None and completed_since_checkpoint >= max_new_rows:
            break
        existing[index] = run_matrix_row(
            index,
            spec,
            generator_commit=generator_commit,
            generator_digest=generator_digest,
        )
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= checkpoint_every:
            _write_csv(output_path, [existing[key] for key in sorted(existing)], CSV_FIELDS)
            if max_new_rows is None:
                completed_since_checkpoint = 0
    _write_csv(output_path, [existing[key] for key in sorted(existing)], CSV_FIELDS)
    expected = sum(1 for index in range(len(specs)) if index % shard_count == shard_index)
    complete = len(existing) == expected
    if max_new_rows is None and not complete:
        raise SweepInputError("shard did not complete its frozen rows")
    return {
        "shard_index": shard_index,
        "rows": len(existing),
        "expected": expected,
        "complete": complete,
    }


def _number(row: dict[str, str], name: str) -> float:
    return float(row[name])


def _integer(row: dict[str, str], name: str) -> int:
    return int(row[name])


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise SweepInputError("cannot summarize an empty metric")
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "stdev": stdev,
        "ci95_half_width": 1.96 * stdev / math.sqrt(len(values)),
        "min": min(values),
        "max": max(values),
    }


def _axis_value(row: dict[str, str], name: str) -> Any:
    if name in {"learners", "s_max", "seed"}:
        return _integer(row, name)
    if name == "speed_model":
        return row[name]
    return _number(row, name)


def _profile_aggregates(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    profile_axes = tuple(name for name in AXIS_ORDER if name != "seed")
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(_axis_value(row, name) for name in profile_axes)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        members = groups[key]
        record = dict(zip(profile_axes, key, strict=True))
        record["seed_count"] = len(members)
        record["visibility_delay_over_h"] = _number(members[0], "visibility_delay_over_h")
        for metric in METRIC_FIELDS:
            values = [_number(row, metric) for row in members]
            summary = _stats(values)
            for stat_name, value in summary.items():
                record[f"{metric}_{stat_name}"] = value
        output.append(record)
    return output


def _recovery_records(rows: Sequence[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    pair_axes = tuple(name for name in AXIS_ORDER if name != "s_max")
    paired: dict[tuple[Any, ...], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = tuple(_axis_value(row, name) for name in pair_axes)
        paired[key][_integer(row, "s_max")] = row
    output: list[dict[str, Any]] = []
    for key in sorted(paired):
        arms = paired[key]
        if set(arms) != {0, 1, 2}:
            raise SweepInputError("S_max comparison is incomplete")
        record = dict(zip(pair_axes, key, strict=True))
        fresh = _number(arms[0], "accepted_token_efficiency")
        record["gain_smax1_vs_smax0"] = _number(arms[1], "accepted_token_efficiency") - fresh
        record["gain_smax2_vs_smax0"] = _number(arms[2], "accepted_token_efficiency") - fresh
        output.append(record)
    expected = config["execution"]["expected_rows"] // 3
    if len(output) != expected:
        raise SweepInputError("recovery comparison cardinality mismatch")
    return output


def _classification(gain_fraction: float, config: dict[str, Any]) -> str:
    thresholds = config["recommendation_policy"]["paper_classification_percentage_points"]
    percentage_points = 100.0 * gain_fraction
    if percentage_points >= thresholds["full_second_contribution_min"]:
        return "full_second_contribution"
    if percentage_points >= thresholds["compressed_section_min"]:
        return "compressed_section"
    return "ablation_or_discussion"


def _candidate_rankings(
    rows: Sequence[dict[str, str]], recovery: Sequence[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    policy = config["recommendation_policy"]["stage4_profile_b_candidates"]
    region = policy["ranking_region"]
    rankings: list[dict[str, Any]] = []
    for q_fraction in policy["quorum_fraction"]:
        for grace in policy["grace_fraction_of_h"]:
            selected_rows = [
                row
                for row in rows
                if _integer(row, "learners") == policy["learners"]
                and _number(row, "quorum_fraction") == q_fraction
                and _number(row, "grace_fraction_of_h") == grace
                and _integer(row, "s_max") == policy["s_max"]
                and _number(row, "heterogeneity_ratio") in region["heterogeneity_ratio"]
                and _number(row, "visibility_delay_seconds") in region["visibility_delay_seconds"]
            ]
            selected_recovery = [
                item
                for item in recovery
                if item["learners"] == policy["learners"]
                and item["quorum_fraction"] == q_fraction
                and item["grace_fraction_of_h"] == grace
                and item["heterogeneity_ratio"] in region["heterogeneity_ratio"]
                and item["visibility_delay_seconds"] in region["visibility_delay_seconds"]
            ]
            rankings.append(
                {
                    "quorum_fraction": q_fraction,
                    "q": int(policy["learners"] * q_fraction),
                    "grace_fraction_of_h": grace,
                    "mean_recovery_gain": statistics.fmean(
                        item["gain_smax1_vs_smax0"] for item in selected_recovery
                    ),
                    "mean_accepted_token_efficiency": statistics.fmean(
                        _number(row, "accepted_token_efficiency") for row in selected_rows
                    ),
                    "mean_update_interval": statistics.fmean(
                        _number(row, "update_interval_mean") for row in selected_rows
                    ),
                    "rows": len(selected_rows),
                    "paired_rows": len(selected_recovery),
                }
            )
    rankings.sort(
        key=lambda item: (
            -item["mean_recovery_gain"],
            -item["mean_accepted_token_efficiency"],
            item["mean_update_interval"],
            item["grace_fraction_of_h"],
            -item["quorum_fraction"],
        )
    )
    for rank, item in enumerate(rankings, start=1):
        item["rank"] = rank
    return rankings


def _metric_summary(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    return {metric: _stats([_number(row, metric) for row in rows]) for metric in METRIC_FIELDS}


def _build_summary(
    rows: Sequence[dict[str, str]],
    aggregates: Sequence[dict[str, Any]],
    recovery: Sequence[dict[str, Any]],
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    marginals: dict[str, list[dict[str, Any]]] = {}
    for axis in AXIS_ORDER:
        axis_groups: dict[Any, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            axis_groups[_axis_value(row, axis)].append(row)
        marginals[axis] = [
            {"value": value, "rows": len(axis_groups[value]), "metrics": _metric_summary(axis_groups[value])}
            for value in sorted(axis_groups, key=str)
        ]
    recovery_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    recovery_axes = tuple(name for name in AXIS_ORDER if name not in {"s_max", "seed"})
    for item in recovery:
        key = tuple(item[name] for name in recovery_axes)
        recovery_groups[key].append(item["gain_smax1_vs_smax0"])
    classified = Counter()
    group_means: list[float] = []
    for values in recovery_groups.values():
        mean = statistics.fmean(values)
        group_means.append(mean)
        classified[_classification(mean, config)] += 1
    rankings = _candidate_rankings(rows, recovery, config)
    chosen = rankings[0]
    anchor_rows = [
        row
        for row in rows
        if _integer(row, "learners") == 8
        and _number(row, "quorum_fraction") == chosen["quorum_fraction"]
        and _number(row, "grace_fraction_of_h") == chosen["grace_fraction_of_h"]
        and _integer(row, "s_max") == 1
        and _number(row, "heterogeneity_ratio") == 2.0
        and _number(row, "visibility_delay_seconds") == 1.0
        and row["speed_model"] == "constant_ratio"
    ]
    anchor_recovery = [
        item
        for item in recovery
        if item["learners"] == 8
        and item["quorum_fraction"] == chosen["quorum_fraction"]
        and item["grace_fraction_of_h"] == chosen["grace_fraction_of_h"]
        and item["heterogeneity_ratio"] == 2.0
        and item["visibility_delay_seconds"] == 1.0
        and item["speed_model"] == "constant_ratio"
    ]
    symmetric = [
        row
        for row in rows
        if _number(row, "quorum_fraction") == 1.0
        and _number(row, "heterogeneity_ratio") == 1.0
        and _number(row, "grace_fraction_of_h") == 0.0
        and _number(row, "visibility_delay_seconds") == 0.0
        and _integer(row, "s_max") == 0
        and row["speed_model"] == "constant_ratio"
    ]
    return {
        "schema_version": 1,
        "profile_id": config["profile_id"],
        "evidence_run_id": run_id,
        "claims_boundary": config["claims_boundary"],
        "matrix": {
            "complete": len(rows) == config["execution"]["expected_rows"],
            "expected_rows": config["execution"]["expected_rows"],
            "actual_rows": len(rows),
            "aggregate_profile_rows": len(aggregates),
            "paired_recovery_rows": len(recovery),
            "seed_count": len(config["axes"]["seed"]),
        },
        "overall_metrics": _metric_summary(rows),
        "marginals": marginals,
        "recovery": {
            "unit": "accepted-token efficiency fraction; multiply by 100 for percentage points",
            "paired_seed_level": _stats([item["gain_smax1_vs_smax0"] for item in recovery]),
            "profile_mean": _stats(group_means),
            "paper_classification_counts": dict(sorted(classified.items())),
        },
        "symmetric_fresh_sanity": {
            "rows": len(symmetric),
            "expected_update_interval": config["fixed"]["h_steps"],
            "metrics": _metric_summary(symmetric),
        },
        "stage4_candidate_rankings": rankings,
        "stage4_prediction_anchor": {
            "profile": {
                "learners": 8,
                "q": chosen["q"],
                "quorum_fraction": chosen["quorum_fraction"],
                "q_fresh": 1,
                "s_max": 1,
                "lambda_s": config["fixed"]["lambda_s"],
                "grace_fraction_of_h": chosen["grace_fraction_of_h"],
                "h_steps": config["fixed"]["h_steps"],
                "fragments": config["fixed"]["fragments"],
                "heterogeneity_ratio": 2.0,
                "visibility_delay_seconds": 1.0,
                "visibility_delay_over_h": 1.0 / config["fixed"]["h_steps"],
                "nominal_fastest_step_seconds": config["fixed"][
                    "nominal_fastest_step_seconds"
                ],
                "speed_model": "constant_ratio",
            },
            "seed_count": len(anchor_rows),
            "metrics": _metric_summary(anchor_rows),
            "recovery_gain": _stats(
                [item["gain_smax1_vs_smax0"] for item in anchor_recovery]
            ),
            "paper_classification": _classification(
                statistics.fmean(item["gain_smax1_vs_smax0"] for item in anchor_recovery),
                config,
            ),
        },
        "stage1_profile_a": config["recommendation_policy"]["stage1_profile_a"],
        "stage5_visibility_delay_seconds": config["recommendation_policy"][
            "stage5_visibility_delay_seconds"
        ],
    }


def _recommendation_markdown(summary: dict[str, Any]) -> str:
    anchor = summary["stage4_prediction_anchor"]
    profile = anchor["profile"]
    recovery = anchor["recovery_gain"]
    stale = anchor["metrics"]["stale_acceptance_rate"]
    efficiency = anchor["metrics"]["accepted_token_efficiency"]
    stage1 = summary["stage1_profile_a"]
    return f"""# Stage 0-A Simulation Recommendation

## Scope

This report is a parameter-selection and later-measurement anchor. It is not
runtime correctness, model-quality, or a reason to cancel Stage 4.

## Matrix and uncertainty

- Complete SIM-03 extended matrix: `{summary['matrix']['actual_rows']}` / `{summary['matrix']['expected_rows']}` raw rows.
- Three seeds per profile; reported uncertainty is sample standard deviation and a normal 95% CI half-width.
- Full seed-level CSV and `{summary['matrix']['aggregate_profile_rows']}` profile aggregates remain in the immutable evidence run `{summary['evidence_run_id']}`.
- Recovery is paired `accepted_token_efficiency(S_max=1) - accepted_token_efficiency(S_max=0)` in fraction units; percentage points are `100 × fraction`.

## Stage 1 Profile A recommendation

Freeze the fresh reference at `M={stage1['learners']}`, `Q=M`, `Q_fresh=M`,
`S_max=0`, `grace=0`, `lambda_s={stage1['lambda_s']}`, `F=4`, and evenly
staggered `H=50` steps. The sweep evaluates `Q_fresh=1`, which is behaviorally
equivalent in this fresh-only `Q=M` arm because every eligible contributor is fresh.
Stage 0-B must still validate or mitigate the storage-dependent H/visibility budget.

## Stage 4 Profile B prediction anchor

The preregistered ranking selects `M={profile['learners']}`, `Q={profile['q']}`
(`Q/M={profile['quorum_fraction']}`), `Q_fresh=1`, `S_max=1`,
`lambda_s={profile['lambda_s']}`, grace `{profile['grace_fraction_of_h']}H`,
`F={profile['fragments']}`, and `H={profile['h_steps']}` steps. The validation
anchor injects constant heterogeneity `{profile['heterogeneity_ratio']}×` and
visibility `{profile['visibility_delay_seconds']}s` (`delay/H={profile['visibility_delay_over_h']:.6f}`).
Here `H` in seconds uses the simulator's explicit nominal-fastest
`{profile['nominal_fastest_step_seconds']}s/step` reference; real training must
recompute the ratio from its measured step time.

- predicted recovery: `{100.0 * recovery['mean']:.4f}` percentage points, 95% CI half-width `{100.0 * recovery['ci95_half_width']:.4f}` pp;
- stale accepted-token rate: `{100.0 * stale['mean']:.4f}%` (95% CI half-width `{100.0 * stale['ci95_half_width']:.4f}` pp);
- accepted-token efficiency: `{100.0 * efficiency['mean']:.4f}%`;
- paper classification: `{anchor['paper_classification']}`.

Stage 4 remains mandatory. Its measured stale acceptance and recovery must be
reported against this exact anchor and the matched `S_max=0` control.

## Stage 5 delay recommendation

Retain the preregistered visibility points `0/1/5/30s`, always reporting both
seconds and `visibility delay / H`. These span the observed no-delay reference,
low-delay operating region, intermediate degradation, and high-delay stress point.
Do not infer storage capability from the simulation; use Stage 0-B measurements.
"""


def _selected_traces(config: dict[str, Any]) -> dict[str, Any]:
    specs_by_axes = {
        tuple(spec[name] for name in AXIS_ORDER): spec for spec in matrix_specs(config)
    }
    traces = []
    for requested in config["selected_trace_profiles"]:
        key = tuple(requested[name] for name in AXIS_ORDER)
        spec = specs_by_axes[key]
        result = simulate(_simulation_config(spec, max_trace_events=128))
        traces.append(
            {
                "requested_profile": requested,
                "config_digest": result.config_digest,
                "global_cycle": result.global_cycle,
                "accepted_token_efficiency": result.accepted_token_efficiency,
                "stale_acceptance_rate": result.stale_acceptance_rate,
                "trace": list(result.trace),
            }
        )
    return {"schema_version": 1, "traces": traces}


def aggregate_shards(
    config_path: Path,
    shard_paths: Sequence[Path],
    output_dir: Path,
    publish_root: Path,
    run_id: str,
    code_commit: str,
    evidence_config_digest: str,
    pbs_job_id: str,
    hostname: str,
    initial_pbs_job_id: str,
    session_count: int,
) -> dict[str, Any]:
    config = load_sweep_config(config_path)
    specs = matrix_specs(config)
    rows: list[dict[str, str]] = []
    for shard_path in shard_paths:
        rows.extend(_read_csv(shard_path))
    rows.sort(key=lambda row: int(row["matrix_index"]))
    if len(rows) != len(specs):
        raise SweepInputError("combined shard cardinality is incomplete")
    seen: set[str] = set()
    for index, (row, spec) in enumerate(zip(rows, specs, strict=True)):
        _validate_retained_row(
            row,
            spec,
            index,
            code_commit,
            evidence_config_digest,
        )
        if row["row_key"] in seen:
            raise SweepInputError("duplicate matrix row key")
        seen.add(row["row_key"])

    combined_path = output_dir / "simulation_raw.csv"
    _write_csv(combined_path, rows, CSV_FIELDS)
    aggregates = _profile_aggregates(rows)
    aggregate_fields = tuple(aggregates[0])
    aggregate_path = output_dir / "simulation_aggregates.csv"
    _write_csv(aggregate_path, aggregates, aggregate_fields)
    recovery = _recovery_records(rows, config)
    summary = _build_summary(rows, aggregates, recovery, config, run_id)
    summary_path = output_dir / "simulation_summary.json"
    _atomic_write_json(summary_path, summary)
    recommendation = _recommendation_markdown(summary)
    recommendation_path = output_dir / "simulation_recommendation.md"
    _atomic_write_text(recommendation_path, recommendation)
    traces_path = output_dir / "selected_event_traces.json"
    _atomic_write_json(traces_path, _selected_traces(config))

    manifest = {
        "schema_version": 1,
        "profile_id": config["profile_id"],
        "run_id": run_id,
        "code_commit": code_commit,
        "pbs_job_id": pbs_job_id,
        "initial_pbs_job_id": initial_pbs_job_id,
        "execution_sessions": session_count,
        "resumed_from_checkpoint": session_count > 1,
        "compute_hostname": hostname,
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "config_canonical_sha256": _sha256_bytes(_canonical_json(config).encode("utf-8")),
        "evidence_config_digest": evidence_config_digest,
        "axes": config["axes"],
        "fixed": config["fixed"],
        "matrix": summary["matrix"],
        "shards": [
            {"path": str(path), "sha256": file_sha256(path), "rows": len(_read_csv(path))}
            for path in shard_paths
        ],
        "artifacts": {
            "raw_csv": {"path": str(combined_path), "sha256": file_sha256(combined_path)},
            "aggregate_csv": {"path": str(aggregate_path), "sha256": file_sha256(aggregate_path)},
            "summary": {"path": str(summary_path), "sha256": file_sha256(summary_path)},
            "recommendation": {
                "path": str(recommendation_path),
                "sha256": file_sha256(recommendation_path),
            },
            "selected_traces": {"path": str(traces_path), "sha256": file_sha256(traces_path)},
        },
        "claims_boundary": config["claims_boundary"],
    }
    manifest_path = output_dir / "simulation_sweep_manifest.json"
    _atomic_write_json(manifest_path, manifest)

    report_root = publish_root / "reports" / "stage0"
    _atomic_write_text(
        report_root / "simulation_sweep_manifest.json",
        manifest_path.read_text(encoding="utf-8"),
    )
    _atomic_write_text(
        report_root / "simulation_summary.json",
        summary_path.read_text(encoding="utf-8"),
    )
    _atomic_write_text(report_root / "simulation_recommendation.md", recommendation)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("run-shard")
    shard.add_argument("--config", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, required=True)
    shard.add_argument("--generator-commit", required=True)
    shard.add_argument("--generator-digest", required=True)
    shard.add_argument("--max-new-rows", type=int)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument("--shard", type=Path, action="append", required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--publish-root", type=Path, required=True)
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument("--code-commit", required=True)
    aggregate.add_argument("--evidence-config-digest", required=True)
    aggregate.add_argument("--pbs-job-id", required=True)
    aggregate.add_argument("--hostname", required=True)
    aggregate.add_argument("--initial-pbs-job-id", required=True)
    aggregate.add_argument("--session-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run-shard":
        result = run_shard(
            args.config,
            args.output,
            args.shard_index,
            args.shard_count,
            args.generator_commit,
            args.generator_digest,
            args.max_new_rows,
        )
        print(_canonical_json(result))
        return 0
    manifest = aggregate_shards(
        args.config,
        args.shard,
        args.output_dir,
        args.publish_root,
        args.run_id,
        args.code_commit,
        args.evidence_config_digest,
        args.pbs_job_id,
        args.hostname,
        args.initial_pbs_job_id,
        args.session_count,
    )
    print(_canonical_json({"rows": manifest["matrix"]["actual_rows"], "complete": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
