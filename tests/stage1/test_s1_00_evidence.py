from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsbdd.evidence import EvidenceError, EvidencePackage, validate_package
from fsbdd.manifest import ManifestError, build_manifest


def identities() -> dict[str, object]:
    return {
        "code": {"repository": "https://example.invalid/repo", "branch": "test", "commit": "a" * 40, "dirty": False},
        "config": {"sha256": "b" * 64},
        "source": {"research_plan_sha256": "c" * 64, "stage0_4_spec_sha256": "d" * 64},
        "skill": {"repository": "https://github.com/UnbearableFate/miyabi-development", "commit": "e" * 40},
        "execution": {"identity": "local-nonce", "initial_hostname": "host", "workflow": "local"},
        "roles": {"declared": {"checker": ["host"]}, "actual": {"checker": ["host"]}},
        "paths": {"project_root": "/repo", "evidence_root": "/evidence"},
    }


def test_dual_manifest_profiles_and_qtime_semantics() -> None:
    local = build_manifest("S1-00", "L1", "2026-07-15T00:00:00Z", "submission_utc", identities())
    assert "scheduler" not in local
    with pytest.raises(ManifestError):
        build_manifest("S1-00", "L2", "2026-07-15T00:00:00Z", "pbs_qtime", identities())
    pbs = build_manifest(
        "S1-00", "L2", "2026-07-15T00:00:00Z", "pbs_qtime", identities(),
        scheduler={"job_id": "1.pbs", "qtime_utc": "2026-07-15T00:00:00Z", "queue": "q", "group": "g", "nodefile_sha256": "f" * 64, "modules": []},
    )
    assert pbs["scheduler"]["qtime_utc"] == pbs["run_identity"]["timestamp_utc"]


def test_package_finalization_and_mutation_matrix(tmp_path: Path) -> None:
    root = tmp_path / "package"
    package = EvidencePackage.create(root)
    package.write_json("tests/result.json", {"passed": True})
    package.write_text("commands.log", "pytest\n")
    manifest = build_manifest("S1-00", "L1", "2026-07-15T00:00:00Z", "submission_utc", identities())
    summary = package.finalize(manifest)
    assert summary["status"] == "admissible"
    assert validate_package(root)["status"] == "admissible"
    with pytest.raises(EvidenceError, match="exists"):
        EvidencePackage.create(root)

    (root / "extra.txt").write_text("extra", encoding="utf-8")
    assert validate_package(root)["status"] == "rejected"
    (root / "extra.txt").unlink()
    (root / "tests/result.json").write_text("truncated", encoding="utf-8")
    assert validate_package(root)["status"] == "rejected"


def test_dirty_placeholder_role_and_resume_identity_rejected(tmp_path: Path) -> None:
    bad = identities()
    bad["code"]["dirty"] = True
    with pytest.raises(ManifestError, match="dirty"):
        build_manifest("S1-00", "L1", "2026-07-15T00:00:00Z", "submission_utc", bad)
    bad = identities()
    bad["roles"]["actual"] = {"checker": ["other"]}
    with pytest.raises(ManifestError, match="role"):
        build_manifest("S1-00", "L1", "2026-07-15T00:00:00Z", "submission_utc", bad)
    bad = identities()
    bad["execution"]["identity"] = "<placeholder>"
    with pytest.raises(ManifestError, match="placeholder"):
        build_manifest("S1-00", "L1", "2026-07-15T00:00:00Z", "submission_utc", bad)

    package = EvidencePackage.create(tmp_path / "resume")
    package.bind_resume_identity({"code": "a", "config": "b", "generator": "c"})
    with pytest.raises(EvidenceError, match="resume"):
        package.verify_resume_identity({"code": "a", "config": "changed", "generator": "c"})
