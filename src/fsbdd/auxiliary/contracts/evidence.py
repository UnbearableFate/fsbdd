from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from fsbdd.diloco.common.identity import canonical_digest, file_digest
from fsbdd.auxiliary.contracts.manifest import ManifestError, validate_manifest


class EvidenceError(RuntimeError):
    pass


_CONTROL = {"manifest.json", "checksums.sha256", "validation-summary.json"}


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise EvidenceError(f"unsafe evidence path: {value}")
    return str(path)


def _actual_payloads(root: Path) -> list[str]:
    files: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise EvidenceError(f"symlink forbidden in evidence package: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in _CONTROL:
                files.append(relative)
    return sorted(files)


class EvidencePackage:
    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(cls, root: Path) -> "EvidencePackage":
        try:
            root.mkdir(parents=True, mode=0o700, exist_ok=False)
        except FileExistsError as error:
            raise EvidenceError(f"evidence root already exists: {root}") from error
        return cls(root)

    def _path(self, relative: str) -> Path:
        path = self.root / _relative(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_text(self, relative: str, value: str) -> None:
        path = self._path(relative)
        if path.exists():
            raise EvidenceError(f"refusing to overwrite evidence file: {relative}")
        path.write_text(value, encoding="utf-8")

    def write_json(self, relative: str, value: Any) -> None:
        self.write_text(relative, json.dumps(value, sort_keys=True, indent=2) + "\n")

    def bind_resume_identity(self, identity: dict[str, str]) -> None:
        needed = {"code", "config", "generator"}
        if identity.keys() != needed or any(not value for value in identity.values()):
            raise EvidenceError(
                "resume identity requires exact code/config/generator fields"
            )
        self.write_json("resume-identity.json", identity)

    def verify_resume_identity(self, identity: dict[str, str]) -> None:
        path = self.root / "resume-identity.json"
        if (
            not path.is_file()
            or json.loads(path.read_text(encoding="utf-8")) != identity
        ):
            raise EvidenceError("resume identity mismatch")

    def finalize(self, manifest: dict[str, Any]) -> dict[str, Any]:
        if any((self.root / name).exists() for name in _CONTROL):
            raise EvidenceError("package already finalized or partially finalized")
        files = _actual_payloads(self.root)
        manifest = json.loads(json.dumps(manifest))
        manifest["evidence"] = {
            "files": files,
            "inventory_sha256": canonical_digest(files),
        }
        manifest["status"] = "finalized"
        validate_manifest(manifest)
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        checksum_paths = ["manifest.json", *files]
        checksum_text = "".join(
            f"{file_digest(self.root / relative)}  {relative}\n"
            for relative in checksum_paths
        )
        (self.root / "checksums.sha256").write_text(checksum_text, encoding="utf-8")
        summary = validate_package(self.root)
        if summary["status"] != "admissible":
            raise EvidenceError(f"final package rejected: {summary['errors']}")
        (self.root / "validation-summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return summary


def _parse_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise EvidenceError("malformed checksums file") from error
        relative = _relative(relative)
        if relative in entries or len(digest) != 64:
            raise EvidenceError("invalid or duplicate checksum entry")
        entries[relative] = digest
    return entries


def validate_package(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest: dict[str, Any] = {}
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        validate_manifest(manifest)
    except (OSError, json.JSONDecodeError, ManifestError, TypeError) as error:
        errors.append(f"manifest: {error}")
    try:
        actual = _actual_payloads(root)
    except (OSError, EvidenceError) as error:
        actual = []
        errors.append(f"inventory: {error}")
    listed = (
        manifest.get("evidence", {}).get("files", [])
        if isinstance(manifest, dict)
        else []
    )
    if listed != actual:
        errors.append(f"listed/actual mismatch: listed={listed!r} actual={actual!r}")
    try:
        checksums = _parse_checksums(root / "checksums.sha256")
        expected = ["manifest.json", *actual]
        if sorted(checksums) != sorted(expected):
            errors.append("checksum inventory mismatch")
        for relative, digest in checksums.items():
            path = root / relative
            if not path.is_file() or file_digest(path) != digest:
                errors.append(f"checksum mismatch: {relative}")
    except (OSError, EvidenceError) as error:
        errors.append(f"checksums: {error}")
    return {
        "schema_version": 1,
        "status": "admissible" if not errors else "rejected",
        "errors": errors,
        "listed_files": len(listed),
        "actual_files": len(actual),
        "manifest_sha256": file_digest(root / "manifest.json")
        if (root / "manifest.json").is_file()
        else None,
    }
