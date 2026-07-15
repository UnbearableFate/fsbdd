from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from .evidence import EvidencePackage
from .identity import file_digest
from .manifest import build_manifest
from .stage1_gate import analyze


class Stage1PackageError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage1PackageError(f"JSON object required: {path}")
    return value


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise Stage1PackageError(f"required evidence file is absent: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise Stage1PackageError(f"refusing to overwrite evidence file: {destination}")
    shutil.copy2(source, destination)


def package(arguments: argparse.Namespace) -> dict[str, Any]:
    evidence = EvidencePackage.create(arguments.evidence_root)
    workload = _read(arguments.config)["selected_workload"]
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
        "reports/stage1/S1-13-gate-contract.json",
        "reports/stage1/S1-13-requirement-matrix.csv",
        "plans/FS_DILOCO_CODEX_EXECUTION_PLAN/loops/stage1/S1-13.md",
    ):
        _copy_file(arguments.project_root / relative, arguments.evidence_root / "contracts" / Path(relative).name)
    _copy_file(arguments.config, arguments.evidence_root / "configs" / "resolved-config.json")
    _copy_file(arguments.asset_root / "manifest.json", arguments.evidence_root / "assets" / "manifest.json")
    _copy_file(arguments.asset_root / "complete.json", arguments.evidence_root / "assets" / "complete.json")
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
    if workload == "nine_node":
        for phase in ("initial", "final"):
            _copy_file(
                arguments.shared_root / "evaluation" / phase / "manifest.json",
                arguments.evidence_root / "evaluation" / phase / "manifest.json",
            )
    if arguments.env_root.is_dir():
        for source in sorted(arguments.env_root.rglob("*")):
            if source.is_file():
                _copy_file(source, arguments.evidence_root / "env" / source.relative_to(arguments.env_root))
    if arguments.stdout_root.is_dir():
        for source in sorted(arguments.stdout_root.rglob("*")):
            if source.is_file():
                _copy_file(source, arguments.evidence_root / "stdout" / source.relative_to(arguments.stdout_root))
    if arguments.stderr_root.is_dir():
        for source in sorted(arguments.stderr_root.rglob("*")):
            if source.is_file():
                if source.stat().st_size:
                    raise Stage1PackageError(f"formal role produced non-empty stderr: {source}")
                _copy_file(source, arguments.evidence_root / "stderr" / source.relative_to(arguments.stderr_root))
    analyze(arguments.result_root, arguments.config, arguments.evidence_root / "analysis")
    requirements = json.loads(arguments.requirements.read_text(encoding="utf-8"))
    evidence.write_json("analysis/requirements.json", requirements)
    nodefile = arguments.evidence_root / "env" / "manifest-role-hosts.txt"
    nodefile.parent.mkdir(parents=True, exist_ok=True)
    nodefile.write_text("".join(f"{host}\n" for host in [*learner_hosts, syncer_host]), encoding="utf-8")
    identities = {
        "code": {
            "repository": arguments.repository,
            "branch": arguments.branch,
            "commit": arguments.commit,
            "dirty": False,
        },
        "config": {"sha256": file_digest(arguments.config)},
        "source": {
            "research_plan_sha256": arguments.research_sha256,
            "stage0_4_spec_sha256": arguments.spec_sha256,
        },
        "skill": {"repository": arguments.skill_repository, "commit": arguments.skill_commit},
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
    parser = argparse.ArgumentParser(description="Build an admissible S1-13 formal evidence package")
    parser.add_argument("--time-source", choices=("submission_utc", "pbs_qtime"), required=True)
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
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(json.dumps(package(arguments), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
