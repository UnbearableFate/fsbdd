from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import socket
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentFailureError(RuntimeError):
    pass


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_CODE_IDENTITY = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def _required_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentFailureError(f"{field} must be nonempty text")
    return value


def _canonical_output_root(path: Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise ExperimentFailureError("output_root must be absolute")
    canonical = raw.resolve(strict=False)
    if raw != canonical:
        raise ExperimentFailureError("output_root must be a canonical absolute path")
    return canonical


def _safe_component(value: str) -> str:
    normalized = _SAFE_COMPONENT.sub("-", value).strip("-.")
    return normalized or "unknown"


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise ExperimentFailureError("failure record write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())


def _utc_timestamp(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ExperimentFailureError(f"{field} must be an explicit UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ExperimentFailureError(f"{field} is not a valid timestamp") from error
    if parsed.utcoffset() != dt.timedelta(0):  # pragma: no cover - fixed suffix
        raise ExperimentFailureError(f"{field} must be UTC")
    return value


def capture_failure(
    *,
    output_root: Path,
    experiment_id: str,
    exit_code: int,
    line_number: int,
    command: str,
    job_id: str,
    role_id: str | None = None,
    code_commit: str | None = None,
    timestamp_utc: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = _canonical_output_root(output_root)
    experiment = _required_text(experiment_id, "experiment_id")
    job = _required_text(job_id, "job_id")
    failed_command = _required_text(command, "command")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code <= 0:
        raise ExperimentFailureError("exit_code must be a positive integer")
    if isinstance(line_number, bool) or not isinstance(line_number, int) or line_number <= 0:
        raise ExperimentFailureError("line_number must be a positive integer")
    if role_id is not None:
        _required_text(role_id, "role_id")
    if code_commit is not None and (
        not isinstance(code_commit, str) or not _CODE_IDENTITY.fullmatch(code_commit)
    ):
        raise ExperimentFailureError(
            "code_commit must be a lowercase SHA-1 or SHA-256 identity"
        )
    captured_at = timestamp_utc or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    _utc_timestamp(captured_at, "timestamp_utc")

    value: dict[str, Any] = {
        "schema_version": 1,
        "record_kind": "experiment_failure_capture",
        "analysis_status": "pending_review",
        "experiment_id": experiment,
        "job_id": job,
        "role_id": role_id,
        "captured_at_utc": captured_at,
        "hostname": socket.gethostname(),
        "code_commit": code_commit,
        "output_root": str(root),
        "failure": {
            "summary": "experiment command exited with a nonzero status",
            "exit_code": exit_code,
            "line_number": line_number,
            "command": failed_command,
        },
        "cause": {
            "status": "pending_review",
            "immediate_cause": "the recorded shell command returned a nonzero status",
            "root_cause": None,
        },
        "expected_solution": {
            "status": "review_required",
            "action": (
                "preserve the failed run, determine the evidence-backed root cause, "
                "write a reviewed failure analysis, and do not submit the next "
                "experiment before that analysis is complete"
            ),
        },
    }
    failure_root = root / "failures"
    failure_root.mkdir(parents=True, exist_ok=True)
    stem_parts = [job]
    if role_id is not None:
        stem_parts.append(role_id)
    stem_parts.append(str(os.getpid()))
    path = failure_root / (
        "raw-" + "-".join(_safe_component(part) for part in stem_parts) + ".json"
    )
    _write_new_json(path, value)
    return path, value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture an immutable raw failure record for an experiment"
    )
    parser.add_argument("capture", choices=("capture",))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--line-number", type=int, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--role-id")
    parser.add_argument("--code-commit")
    parser.add_argument("--timestamp-utc")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    path, value = capture_failure(
        output_root=arguments.output_root,
        experiment_id=arguments.experiment_id,
        exit_code=arguments.exit_code,
        line_number=arguments.line_number,
        command=arguments.command,
        job_id=arguments.job_id,
        role_id=arguments.role_id,
        code_commit=arguments.code_commit,
        timestamp_utc=arguments.timestamp_utc,
    )
    print(json.dumps({"path": str(path), "record": value}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
