from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from fsbdd.auxiliary.contracts.evidence import EvidencePackage
from fsbdd.auxiliary.contracts.manifest import build_manifest
from fsbdd.auxiliary.stage1.gate import analyze
from fsbdd.diloco.common.identity import file_digest
from fsbdd.diloco.protocol.storage import PublicationError, decode_publication_record


class Stage1PackageError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage1PackageError(f"JSON object required: {path}")
    return value


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise Stage1PackageError(f"required evidence file is absent: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise Stage1PackageError(f"refusing to overwrite evidence file: {destination}")
    shutil.copy2(source, destination)


def _reject_placeholders(value: Any, *, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_placeholders(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_placeholders(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and (
        value == "asset_stage"
        or "<submission-nonce>" in value
        or "<pbs-job-id>" in value
    ):
        raise Stage1PackageError(f"unresolved placeholder in {path}")


def _validate_submission_marker(arguments: argparse.Namespace) -> None:
    marker_path = arguments.submission_marker
    if marker_path is None or not marker_path.is_file() or marker_path.is_symlink():
        raise Stage1PackageError("nine-node package requires a safe submission marker")
    marker = _read(marker_path)
    expected = {
        "complete": True,
        "kind": "s1_13_nine_node_exclusive_submission_roots",
        "workload": "nine_node",
        "run_id": arguments.run_id,
        "submission_utc": arguments.timestamp_utc,
        "code_commit": arguments.commit,
        "config_sha256": file_digest(arguments.config),
        "asset_marker_sha256": file_digest(arguments.asset_root / "complete.json"),
        "gate_contract_sha256": file_digest(arguments.gate_contract),
        "shared_root": str(arguments.shared_root.resolve()),
        "result_root": str(arguments.result_root.resolve()),
        "evidence_root": str(arguments.evidence_root.resolve()),
        "creation_semantics": "each raw root created by one exclusive mkdir before qsub",
        "fail_if_exists": True,
        "schema_version": 1,
    }
    if marker != expected:
        raise Stage1PackageError("submission marker differs from package identities")
    shared_marker = arguments.shared_root / "submission-root.json"
    if (
        not shared_marker.is_file()
        or shared_marker.is_symlink()
        or shared_marker.read_bytes() != marker_path.read_bytes()
    ):
        raise Stage1PackageError("shared and result submission markers differ")


def _capture_current_protocol_samples(
    shared_root: Path,
    evidence: EvidencePackage,
) -> dict[str, Any]:
    """Retain fixed-slot metadata plus small, independently readable byte samples."""

    summary: dict[str, Any] = {
        "schema_version": 1,
        "sample_bytes_per_edge": 4096,
        "backends": {},
    }
    for backend_name in ("global", "proposals"):
        backend_root = shared_root / "protocol" / backend_name
        visibility_root = backend_root / "visibility"
        raw_payload_root = backend_root / "payloads"
        payload_root = raw_payload_root.resolve()
        if (
            not visibility_root.is_dir()
            or visibility_root.is_symlink()
            or not raw_payload_root.is_dir()
            or raw_payload_root.is_symlink()
        ):
            raise Stage1PackageError(
                f"raw protocol backend is incomplete: {backend_root}"
            )
        rows = []
        for source in sorted(visibility_root.glob("*.json")):
            if source.is_symlink():
                raise Stage1PackageError(f"unsafe current visibility record: {source}")
            try:
                record_value = decode_publication_record(source.read_bytes())
            except (OSError, PublicationError) as error:
                raise Stage1PackageError(
                    f"invalid current visibility record: {source}"
                ) from error
            record = record_value.to_dict()
            relative = record_value.payload_relative_path
            raw_payload = backend_root / relative
            payload = raw_payload.resolve()
            if (
                raw_payload.is_symlink()
                or payload.parent != payload_root
                or not payload.is_file()
            ):
                raise Stage1PackageError(f"unsafe or absent current payload: {payload}")
            payload_bytes = record_value.payload_bytes
            if payload.stat().st_size != payload_bytes:
                raise Stage1PackageError(f"current payload size mismatch: {payload}")
            with payload.open("rb") as stream:
                prefix = stream.read(min(4096, payload_bytes))
                stream.seek(max(0, payload_bytes - 4096))
                suffix = stream.read(min(4096, payload_bytes))
            _copy_file(
                source,
                evidence.root
                / "raw-metadata"
                / backend_name
                / "visibility"
                / source.name,
            )
            sample = {
                "schema_version": 1,
                "slot": source.stem,
                "payload_relative_path": relative,
                "payload_bytes": payload_bytes,
                "payload_sha256": record["payload_sha256"],
                "dtype": record["dtype"],
                "shape": record["shape"],
                "encoding": "hex",
                "prefix_offset": 0,
                "prefix_hex": prefix.hex(),
                "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                "suffix_offset": payload_bytes - len(suffix),
                "suffix_hex": suffix.hex(),
                "suffix_sha256": hashlib.sha256(suffix).hexdigest(),
            }
            evidence.write_json(
                f"raw-metadata/{backend_name}/samples/{source.stem}.json",
                sample,
            )
            rows.append(
                {
                    "slot": source.stem,
                    "record_sha256": file_digest(source),
                    "payload_bytes": payload_bytes,
                    "payload_sha256": record["payload_sha256"],
                    "sample_path": f"raw-metadata/{backend_name}/samples/{source.stem}.json",
                }
            )
        summary["backends"][backend_name] = {
            "current_visibility_records": len(rows),
            "current_slots": rows,
        }
    evidence.write_json("raw-metadata/classified-current-inventory.json", summary)
    return summary


def package(arguments: argparse.Namespace) -> dict[str, Any]:
    evidence = EvidencePackage.create(arguments.evidence_root)
    resolved_config = _read(arguments.config)
    _reject_placeholders(resolved_config)
    workload = resolved_config["selected_workload"]
    expected_count = 8 if workload == "nine_node" else 4
    roles = [
        _read(arguments.result_root / "roles" / f"learner-{index:02d}.json")
        for index in range(expected_count)
    ]
    syncer = _read(arguments.result_root / "roles" / "syncer.json")
    learner_hosts = [str(item["identity"]["hostname"]) for item in roles]
    syncer_host = str(syncer["identity"]["hostname"])
    role_map = {"learners": learner_hosts, "syncer": [syncer_host]}
    for relative in (
        "reports/stage1/S1-13-evidence-contract.json",
        "reports/stage1/S1-13-experiment-failures.json",
        "reports/stage1/S1-13-long-schedule-adr.md",
        "reports/stage1/S1-13-requirement-matrix.csv",
        "plans/FS_DILOCO_CODEX_EXECUTION_PLAN/loops/stage1/S1-13.md",
    ):
        _copy_file(
            arguments.project_root / relative,
            arguments.evidence_root / "contracts" / Path(relative).name,
        )
    _copy_file(
        arguments.gate_contract,
        arguments.evidence_root / "contracts" / "S1-13-gate-contract.json",
    )
    if workload == "nine_node":
        if arguments.submission_marker is None:
            raise Stage1PackageError("nine-node package requires a submission marker")
        _validate_submission_marker(arguments)
        _copy_file(
            arguments.submission_marker,
            arguments.evidence_root / "contracts" / "submission-root.json",
        )
    _copy_file(
        arguments.config, arguments.evidence_root / "configs" / "resolved-config.json"
    )
    _copy_file(
        arguments.asset_root / "manifest.json",
        arguments.evidence_root / "assets" / "manifest.json",
    )
    _copy_file(
        arguments.asset_root / "complete.json",
        arguments.evidence_root / "assets" / "complete.json",
    )
    _copy_file(
        arguments.asset_root / "workloads" / workload / "manifest.json",
        arguments.evidence_root / "assets" / "workload-manifest.json",
    )
    for index in range(expected_count):
        learner = f"learner-{index:02d}"
        _copy_file(
            arguments.result_root / "roles" / f"{learner}.json",
            arguments.evidence_root / "runtime" / "roles" / f"{learner}.json",
        )
        _copy_file(
            arguments.result_root / "logs" / f"{learner}.jsonl",
            arguments.evidence_root / "runtime" / "logs" / f"{learner}.jsonl",
        )
    _copy_file(
        arguments.result_root / "roles" / "syncer.json",
        arguments.evidence_root / "runtime" / "roles" / "syncer.json",
    )
    _copy_file(
        arguments.result_root / "logs" / "syncer.jsonl",
        arguments.evidence_root / "runtime" / "logs" / "syncer.jsonl",
    )
    protocol_samples = _capture_current_protocol_samples(
        arguments.shared_root, evidence
    )
    expected_current = {"global": 4, "proposals": expected_count * 4}
    if any(
        protocol_samples["backends"][name]["current_visibility_records"] != count
        for name, count in expected_current.items()
    ):
        raise Stage1PackageError("current raw protocol slot inventory is incomplete")
    if workload == "nine_node":
        for phase in ("initial", "final"):
            _copy_file(
                arguments.shared_root / "evaluation" / phase / "manifest.json",
                arguments.evidence_root / "evaluation" / phase / "manifest.json",
            )
    if arguments.env_root.is_dir():
        for source in sorted(arguments.env_root.rglob("*")):
            if source.is_file():
                _copy_file(
                    source,
                    arguments.evidence_root
                    / "env"
                    / source.relative_to(arguments.env_root),
                )
    if arguments.stdout_root.is_dir():
        for source in sorted(arguments.stdout_root.rglob("*")):
            if source.is_file():
                _copy_file(
                    source,
                    arguments.evidence_root
                    / "stdout"
                    / source.relative_to(arguments.stdout_root),
                )
    if arguments.stderr_root.is_dir():
        for source in sorted(arguments.stderr_root.rglob("*")):
            if source.is_file():
                if source.stat().st_size:
                    raise Stage1PackageError(
                        f"formal role produced non-empty stderr: {source}"
                    )
                _copy_file(
                    source,
                    arguments.evidence_root
                    / "stderr"
                    / source.relative_to(arguments.stderr_root),
                )
    analyze(
        arguments.result_root,
        arguments.config,
        arguments.gate_contract,
        arguments.evidence_root / "analysis",
    )
    runtime_source = (
        arguments.project_root / "src/fsbdd/auxiliary/stage1/close.py"
    ).read_text(encoding="utf-8")
    forbidden_api_tokens = (
        "init_process_group(",
        "DistributedDataParallel(",
        "init_rpc(",
        "torchrun",
        "nccl",
    )
    matches = [token for token in forbidden_api_tokens if token in runtime_source]
    evidence.write_json(
        "analysis/static-no-network.json",
        {
            "schema_version": 1,
            "status": "pass" if not matches else "fail",
            "scope": "src/fsbdd/auxiliary/stage1/close.py",
            "forbidden_application_data_plane_api_matches": matches,
            "mpi_usage": "launcher_only_in_five_node_long_run",
            "algorithm_coordination": "shared_filesystem_only",
        },
    )
    if matches:
        raise Stage1PackageError(
            "formal runtime source contains a forbidden network data-plane API"
        )
    requirements = json.loads(arguments.requirements.read_text(encoding="utf-8"))
    evidence.write_json("analysis/requirements.json", requirements)
    nodefile = arguments.evidence_root / "env" / "manifest-role-hosts.txt"
    nodefile.parent.mkdir(parents=True, exist_ok=True)
    nodefile.write_text(
        "".join(f"{host}\n" for host in [*learner_hosts, syncer_host]), encoding="utf-8"
    )
    identities = {
        "code": {
            "repository": arguments.repository,
            "branch": arguments.branch,
            "commit": arguments.commit,
            "dirty": False,
        },
        "config": {
            "sha256": file_digest(arguments.config),
            "gate_contract_sha256": file_digest(arguments.gate_contract),
        },
        "source": {
            "research_plan_sha256": arguments.research_sha256,
            "stage0_4_spec_sha256": arguments.spec_sha256,
        },
        "skill": {
            "repository": arguments.skill_repository,
            "commit": arguments.skill_commit,
        },
        "execution": {
            "identity": arguments.run_id,
            "initial_hostname": arguments.initial_hostname,
            "compute_hostname": syncer_host,
            "workflow": (
                "nine-independent-single-node-pbs-array-roles-filesystem-only"
                if workload == "nine_node"
                else "five-node-coallocated-four-learners-one-cpu-syncer-filesystem-only"
            ),
        },
        "roles": {"declared": role_map, "actual": role_map},
        "paths": {
            "project_root": str(arguments.project_root.resolve()),
            "evidence_root": str(arguments.evidence_root.resolve()),
        },
    }
    manifest = build_manifest(
        "S1-13",
        "L3" if workload == "nine_node" else "L4",
        arguments.timestamp_utc,
        arguments.time_source,
        identities,
        scheduler={
            "job_id": arguments.job_id,
            "qtime_utc": arguments.scheduler_qtime_utc,
            "queue": arguments.queue,
            "group": "xg24i002",
            "nodefile_sha256": file_digest(nodefile),
            "modules": ["nv-hpcx/25.9"],
        },
    )
    manifest["stage1_closure"] = {
        "workload": workload,
        "asset_marker_sha256": file_digest(arguments.asset_root / "complete.json"),
        "gate_contract_sha256": file_digest(arguments.gate_contract),
        "submission_marker_sha256": (
            None
            if arguments.submission_marker is None
            else file_digest(arguments.submission_marker)
        ),
        "hf_cli": {"skill": "hugging-face:hf-cli", "version": "1.23.0"},
        "raw_protocol_root": str(arguments.shared_root.resolve()),
        "role_scheduler_records": [
            {
                "role": item["learner_id"],
                "job_id": item["identity"]["pbs_job_id"],
                "qtime_utc": item["identity"]["pbs_qtime_utc"],
                "hostname": item["identity"]["hostname"],
                "gpu_uuid": item["identity"]["gpu"]["gpu_uuid"],
            }
            for item in roles
        ]
        + [
            {
                "role": "syncer",
                "job_id": syncer["identity"]["pbs_job_id"],
                "qtime_utc": syncer["identity"]["pbs_qtime_utc"],
                "hostname": syncer_host,
                "gpu_uuid": None,
            }
        ],
    }
    return evidence.finalize(manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an admissible S1-13 formal evidence package"
    )
    parser.add_argument(
        "--time-source", choices=("submission_utc", "pbs_qtime"), required=True
    )
    for name in (
        "run-id",
        "repository",
        "branch",
        "commit",
        "research-sha256",
        "spec-sha256",
        "skill-repository",
        "skill-commit",
        "initial-hostname",
        "job-id",
        "timestamp-utc",
        "scheduler-qtime-utc",
        "queue",
    ):
        parser.add_argument(f"--{name}", required=True)
    for name in (
        "project-root",
        "evidence-root",
        "result-root",
        "shared-root",
        "asset-root",
        "config",
        "requirements",
        "env-root",
        "stdout-root",
        "stderr-root",
        "gate-contract",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--submission-marker", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(json.dumps(package(arguments), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
