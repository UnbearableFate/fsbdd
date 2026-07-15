from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from fsbdd.diloco.protocol.storage import (
    PosixStorageBackend,
    PublicationSpec,
)


class CapacityPreflightError(RuntimeError):
    """The S1-13 exact-size publication preflight is not admissible."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapacityPreflightError(f"cannot read JSON contract: {path}") from error
    if not isinstance(value, dict):
        raise CapacityPreflightError("capacity contract must be a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with path.open("xb") as stream:
            if stream.write(content) != len(content):
                raise CapacityPreflightError("capacity result write was incomplete")
    except OSError as error:
        raise CapacityPreflightError(f"cannot write capacity result: {path}") from error


def _require_positive_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CapacityPreflightError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_float(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise CapacityPreflightError(f"{field} must be nonnegative")
    return float(value)


def _validated_contract(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    actual_sha256 = _hash_file(path)
    if actual_sha256 != expected_sha256:
        raise CapacityPreflightError("capacity contract SHA-256 mismatch")
    contract = _read_json(path)
    if (
        contract.get("schema_version") != 1
        or contract.get("kind") != "s1_13_exact_size_publication_capacity_preflight"
    ):
        raise CapacityPreflightError("capacity contract identity is invalid")
    sizes = contract.get("payload_bytes")
    if not isinstance(sizes, list) or not sizes:
        raise CapacityPreflightError("capacity contract omits payload sizes")
    for index, size in enumerate(sizes):
        _require_positive_integer(size, f"payload_bytes[{index}]")
    _require_positive_integer(contract.get("repetitions"), "repetitions")
    thresholds = contract.get("thresholds")
    if not isinstance(thresholds, dict):
        raise CapacityPreflightError("capacity contract omits thresholds")
    for field in (
        "maximum_source_to_readback_median_ratio",
        "minimum_mean_seconds_saved",
        "maximum_predicted_update_mean_seconds",
    ):
        _require_nonnegative_float(thresholds.get(field), f"thresholds.{field}")
    baseline = contract.get("failed_smoke_baseline")
    if not isinstance(baseline, dict):
        raise CapacityPreflightError("capacity contract omits the failed baseline")
    _require_nonnegative_float(
        baseline.get("mean_update_latency_seconds"),
        "failed_smoke_baseline.mean_update_latency_seconds",
    )
    return contract, actual_sha256


def _publication_spec(run_id: str, *, sequence: int, size: int) -> PublicationSpec:
    return PublicationSpec(
        run_identity=run_id,
        fragment_map_identity="a" * 64,
        fragment_identity=f"capacity-payload-{size}",
        version=sequence,
        sequence=sequence,
        dtype="uint8",
        shape=(size,),
        base_content_identity="b" * 64,
    )


def _run_trial(
    *,
    root: Path,
    run_id: str,
    payload: bytes,
    size: int,
    repetition: int,
    readback: bool,
) -> dict[str, Any]:
    mode = "post_write_readback" if readback else "source_sha256_complete_write"
    backend = PosixStorageBackend(
        root / mode, verify_payload_readback=readback
    )
    if backend.publication_verification_mode != mode:
        raise CapacityPreflightError("backend verification mode differs")
    sequence = repetition + 1
    started_ns = time.monotonic_ns()
    record = backend.publish(
        "current",
        payload,
        _publication_spec(run_id, sequence=sequence, size=size),
    )
    completed_ns = time.monotonic_ns()
    elapsed_seconds = (completed_ns - started_ns) / 1_000_000_000
    if elapsed_seconds <= 0:
        raise CapacityPreflightError("publication duration must be positive")
    observed = backend.read_bound_record(record)
    if observed.payload != payload:
        raise CapacityPreflightError("published payload validation differs")
    return {
        "mode": mode,
        "payload_bytes": size,
        "repetition": repetition,
        "elapsed_seconds": elapsed_seconds,
        "bytes_per_second": size / elapsed_seconds,
        "payload_sha256": record.payload_sha256,
        "payload_relative_path": record.payload_relative_path,
        "read_validation": "pass",
    }


def run_preflight(
    *,
    shared_root: Path,
    output: Path,
    run_id: str,
    contract_path: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    contract, actual_contract_sha256 = _validated_contract(
        contract_path, contract_sha256
    )
    if shared_root.exists():
        raise CapacityPreflightError("capacity shared root already exists")
    shared_root.mkdir(parents=True)
    sizes = tuple(int(value) for value in contract["payload_bytes"])
    repetitions = int(contract["repetitions"])
    trials: list[dict[str, Any]] = []
    for size in sizes:
        payload = bytes(size)
        for repetition in range(repetitions):
            # Alternate order so a mode cannot win only because it always runs
            # first or second in the shared-filesystem cache/load sequence.
            modes = (True, False) if repetition % 2 == 0 else (False, True)
            for readback in modes:
                trials.append(
                    _run_trial(
                        root=shared_root / f"payload-{size}",
                        run_id=run_id,
                        payload=payload,
                        size=size,
                        repetition=repetition,
                        readback=readback,
                    )
                )
        del payload

    summaries: list[dict[str, Any]] = []
    seconds_saved: list[float] = []
    ratios: list[float] = []
    for size in sizes:
        readback_values = [
            float(row["elapsed_seconds"])
            for row in trials
            if row["payload_bytes"] == size
            and row["mode"] == "post_write_readback"
        ]
        source_values = [
            float(row["elapsed_seconds"])
            for row in trials
            if row["payload_bytes"] == size
            and row["mode"] == "source_sha256_complete_write"
        ]
        if len(readback_values) != repetitions or len(source_values) != repetitions:
            raise CapacityPreflightError("capacity trial matrix is incomplete")
        readback_median = statistics.median(readback_values)
        source_median = statistics.median(source_values)
        ratio = source_median / readback_median
        saved = readback_median - source_median
        ratios.append(ratio)
        seconds_saved.append(saved)
        summaries.append(
            {
                "payload_bytes": size,
                "post_write_readback_median_seconds": readback_median,
                "source_complete_write_median_seconds": source_median,
                "source_to_readback_ratio": ratio,
                "seconds_saved": saved,
            }
        )

    mean_seconds_saved = statistics.fmean(seconds_saved)
    maximum_ratio = max(ratios)
    baseline_mean = float(
        contract["failed_smoke_baseline"]["mean_update_latency_seconds"]
    )
    predicted_update_mean = baseline_mean - mean_seconds_saved
    thresholds = contract["thresholds"]
    checks = {
        "trial_matrix_complete": len(trials)
        == len(sizes) * repetitions * 2,
        "all_payload_reads_validated": all(
            row["read_validation"] == "pass" for row in trials
        ),
        "source_mode_faster_for_every_size": all(value > 0 for value in seconds_saved),
        "maximum_source_to_readback_ratio": maximum_ratio
        <= float(thresholds["maximum_source_to_readback_median_ratio"]),
        "minimum_mean_seconds_saved": mean_seconds_saved
        >= float(thresholds["minimum_mean_seconds_saved"]),
        "predicted_update_mean": predicted_update_mean
        <= float(thresholds["maximum_predicted_update_mean_seconds"]),
        "torch_not_imported": "torch" not in sys.modules,
    }
    result = {
        "schema_version": 1,
        "kind": "s1_13_exact_size_publication_capacity_preflight_result",
        "status": "pass" if all(checks.values()) else "fail",
        "complete": True,
        "run_id": run_id,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "hostname": os.uname().nodename,
        "code_commit": os.environ.get("EXPECTED_COMMIT"),
        "contract_path": str(contract_path),
        "contract_sha256": actual_contract_sha256,
        "verification_modes": [
            "post_write_readback",
            "source_sha256_complete_write",
        ],
        "trials": trials,
        "summaries": summaries,
        "aggregate": {
            "mean_seconds_saved": mean_seconds_saved,
            "maximum_source_to_readback_ratio": maximum_ratio,
            "failed_smoke_mean_update_latency_seconds": baseline_mean,
            "predicted_corrected_update_mean_seconds": predicted_update_mean,
        },
        "thresholds": thresholds,
        "checks": checks,
        "forbidden_runtime": {
            "torch_module_imported": "torch" in sys.modules,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    _write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    arguments = parser.parse_args()
    try:
        result = run_preflight(
            shared_root=arguments.shared_root,
            output=arguments.output,
            run_id=arguments.run_id,
            contract_path=arguments.contract,
            contract_sha256=arguments.contract_sha256,
        )
    except Exception as error:
        if not arguments.output.exists():
            _write_json(
                arguments.output,
                {
                    "schema_version": 1,
                    "kind": "s1_13_exact_size_publication_capacity_preflight_result",
                    "status": "fail",
                    "complete": False,
                    "run_id": arguments.run_id,
                    "pbs_job_id": os.environ.get("PBS_JOBID"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        return 1
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
