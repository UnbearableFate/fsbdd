from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class Stage1SubmissionError(RuntimeError):
    pass


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_root(path: Path, field: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise Stage1SubmissionError(f"{field} must be absolute")
    canonical = raw.resolve(strict=False)
    if raw != canonical:
        raise Stage1SubmissionError(f"{field} must be a canonical absolute path")
    if not canonical.parent.is_dir():
        raise Stage1SubmissionError(f"{field} parent does not exist")
    return canonical


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(value: str, field: str) -> str:
    if not _SHA256.fullmatch(value):
        raise Stage1SubmissionError(f"{field} must be a lowercase SHA-256 identity")
    return value


def _utc_timestamp(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Stage1SubmissionError(f"{field} must be UTC with a Z suffix")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise Stage1SubmissionError(f"invalid {field}") from error
    if parsed.utcoffset() != dt.timedelta(0):  # pragma: no cover - fixed suffix
        raise Stage1SubmissionError(f"{field} must be UTC")
    return value


def _marker(
    *,
    shared_root: Path,
    result_root: Path,
    evidence_root: Path,
    run_id: str,
    submission_utc: str,
    code_commit: str,
    config_sha256: str,
    asset_marker_sha256: str,
    gate_contract_sha256: str,
) -> dict[str, Any]:
    if not run_id or any(character.isspace() for character in run_id):
        raise Stage1SubmissionError("run_id must be nonempty and contain no whitespace")
    _utc_timestamp(submission_utc, "submission_utc")
    return {
        "schema_version": 1,
        "complete": True,
        "kind": "s1_13_nine_node_exclusive_submission_roots",
        "workload": "nine_node",
        "run_id": run_id,
        "submission_utc": submission_utc,
        "code_commit": _identity(code_commit, "code_commit"),
        "config_sha256": _identity(config_sha256, "config_sha256"),
        "asset_marker_sha256": _identity(asset_marker_sha256, "asset_marker_sha256"),
        "gate_contract_sha256": _identity(gate_contract_sha256, "gate_contract_sha256"),
        "shared_root": str(shared_root),
        "result_root": str(result_root),
        "evidence_root": str(evidence_root),
        "creation_semantics": "each raw root created by one exclusive mkdir before qsub",
        "fail_if_exists": True,
    }


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise Stage1SubmissionError("submission marker write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())


def prepare_nine_node_roots(
    *,
    shared_root: Path,
    result_root: Path,
    evidence_root: Path,
    run_id: str,
    submission_utc: str,
    code_commit: str,
    config_sha256: str,
    asset_marker_sha256: str,
    gate_contract_sha256: str,
) -> dict[str, Any]:
    shared = _canonical_root(shared_root, "shared_root")
    result = _canonical_root(result_root, "result_root")
    evidence = _canonical_root(evidence_root, "evidence_root")
    if len({shared, result, evidence}) != 3:
        raise Stage1SubmissionError("shared, result, and evidence roots must differ")
    if len({shared, result, evidence}) != 3:
        raise Stage1SubmissionError("shared, result, and evidence roots must differ")
    for path in (shared, result, evidence):
        if path.exists():
            raise Stage1SubmissionError(f"refusing to reuse submission root: {path}")
    value = _marker(
        shared_root=shared,
        result_root=result,
        evidence_root=evidence,
        run_id=run_id,
        submission_utc=submission_utc,
        code_commit=code_commit,
        config_sha256=config_sha256,
        asset_marker_sha256=asset_marker_sha256,
        gate_contract_sha256=gate_contract_sha256,
    )
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    created: list[Path] = []
    try:
        shared.mkdir(mode=0o750, exist_ok=False)
        created.append(shared)
        result.mkdir(mode=0o750, exist_ok=False)
        created.append(result)
        _write_new(shared / "submission-root.json", payload)
        _write_new(result / "submission-root.json", payload)
    except BaseException:
        for root in reversed(created):
            marker = root / "submission-root.json"
            if marker.is_file():
                marker.unlink()
            try:
                root.rmdir()
            except OSError:
                pass
        raise
    marker_sha256 = hashlib.sha256(payload).hexdigest()
    return {**value, "submission_marker_sha256": marker_sha256}


def validate_nine_node_roots(
    *,
    shared_root: Path,
    result_root: Path,
    evidence_root: Path,
    run_id: str,
    submission_utc: str,
    code_commit: str,
    config_sha256: str,
    asset_marker_sha256: str,
    gate_contract_sha256: str,
    submission_marker_sha256: str,
) -> dict[str, Any]:
    shared = _canonical_root(shared_root, "shared_root")
    result = _canonical_root(result_root, "result_root")
    evidence = _canonical_root(evidence_root, "evidence_root")
    expected = _marker(
        shared_root=shared,
        result_root=result,
        evidence_root=evidence,
        run_id=run_id,
        submission_utc=submission_utc,
        code_commit=code_commit,
        config_sha256=config_sha256,
        asset_marker_sha256=asset_marker_sha256,
        gate_contract_sha256=gate_contract_sha256,
    )
    expected_sha256 = _identity(submission_marker_sha256, "submission_marker_sha256")
    for root in (shared, result):
        marker = root / "submission-root.json"
        if not marker.is_file() or marker.is_symlink():
            raise Stage1SubmissionError(
                f"submission marker is absent or unsafe: {marker}"
            )
        if _sha256(marker) != expected_sha256:
            raise Stage1SubmissionError(
                f"submission marker checksum mismatch: {marker}"
            )
        try:
            actual = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise Stage1SubmissionError(
                f"invalid submission marker: {marker}"
            ) from error
        if actual != expected:
            raise Stage1SubmissionError(
                f"submission marker identity mismatch: {marker}"
            )
    if (shared / "submission-root.json").read_bytes() != (
        result / "submission-root.json"
    ).read_bytes():
        raise Stage1SubmissionError("shared and result submission markers differ")
    return {**expected, "submission_marker_sha256": expected_sha256}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or validate exclusive S1-13 nine-node submission roots"
    )
    parser.add_argument("action", choices=("prepare-nine", "validate-nine"))
    for name in ("shared-root", "result-root", "evidence-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "run-id",
        "submission-utc",
        "code-commit",
        "config-sha256",
        "asset-marker-sha256",
        "gate-contract-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--submission-marker-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    common = {
        "shared_root": arguments.shared_root,
        "result_root": arguments.result_root,
        "evidence_root": arguments.evidence_root,
        "run_id": arguments.run_id,
        "submission_utc": arguments.submission_utc,
        "code_commit": arguments.code_commit,
        "config_sha256": arguments.config_sha256,
        "asset_marker_sha256": arguments.asset_marker_sha256,
        "gate_contract_sha256": arguments.gate_contract_sha256,
    }
    if arguments.action == "prepare-nine":
        if arguments.submission_marker_sha256 is not None:
            raise Stage1SubmissionError(
                "prepare-nine does not accept submission_marker_sha256"
            )
        result = prepare_nine_node_roots(**common)
    else:
        if arguments.submission_marker_sha256 is None:
            raise Stage1SubmissionError(
                "validate-nine requires submission_marker_sha256"
            )
        result = validate_nine_node_roots(
            **common,
            submission_marker_sha256=arguments.submission_marker_sha256,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
