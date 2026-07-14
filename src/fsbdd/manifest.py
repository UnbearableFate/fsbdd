from __future__ import annotations

import copy
import datetime as dt
import re
from typing import Any

from .identity import canonical_digest


class ManifestError(ValueError):
    pass


_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")
_LEVELS = {"L0", "L1", "L2", "L3", "L4"}
_PLACEHOLDERS = ("<placeholder", "<replace", "<sha", "todo", "tbd")


def _parse_utc(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ManifestError(f"{field} must be UTC with Z suffix")
    try:
        dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ManifestError(f"invalid {field}") from error
    return value


def _find_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        lower = value.lower()
        return any(marker in lower for marker in _PLACEHOLDERS)
    if isinstance(value, dict):
        return any(_find_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_find_placeholder(item) for item in value)
    return False


def _validate_identities(identities: dict[str, Any]) -> None:
    required = {"code", "config", "source", "skill", "execution", "roles", "paths"}
    missing = required - identities.keys()
    if missing:
        raise ManifestError(f"missing identities: {sorted(missing)}")
    if _find_placeholder(identities):
        raise ManifestError("placeholder in manifest identity")
    code = identities["code"]
    if code.get("dirty") is not False:
        raise ManifestError("dirty code cannot produce formal evidence")
    if not _SHA40.fullmatch(str(code.get("commit", ""))):
        raise ManifestError("invalid code commit")
    source = identities["source"]
    for field in ("research_plan_sha256", "stage0_4_spec_sha256"):
        if not _SHA64.fullmatch(str(source.get(field, ""))):
            raise ManifestError(f"missing or invalid source hash: {field}")
    if not _SHA40.fullmatch(str(identities["skill"].get("commit", ""))):
        raise ManifestError("missing or invalid skill commit")
    if not _SHA64.fullmatch(str(identities["config"].get("sha256", ""))):
        raise ManifestError("missing or invalid config digest")
    if identities["roles"].get("declared") != identities["roles"].get("actual"):
        raise ManifestError("declared/actual role map mismatch")
    if not identities["execution"].get("identity"):
        raise ManifestError("missing execution identity")


def build_manifest(
    loop_id: str,
    resource_level: str,
    timestamp_utc: str,
    time_source: str,
    identities: dict[str, Any],
    *,
    scheduler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if resource_level not in _LEVELS:
        raise ManifestError("unknown resource level")
    _validate_identities(identities)
    timestamp = _parse_utc(timestamp_utc, "timestamp_utc")
    if time_source not in {"submission_utc", "pbs_qtime"}:
        raise ManifestError("invalid time_source")
    requires_pbs = resource_level in {"L2", "L3", "L4"}
    if requires_pbs and scheduler is None:
        raise ManifestError("PBS profile requires scheduler identity")
    if not requires_pbs and scheduler is not None:
        raise ManifestError("local profile must omit scheduler identity")
    if time_source == "pbs_qtime":
        if scheduler is None:
            raise ManifestError("pbs_qtime requires scheduler identity")
        if _parse_utc(scheduler.get("qtime_utc"), "scheduler.qtime_utc") != timestamp:
            raise ManifestError("timestamp and qtime mismatch")
    manifest = {
        "schema_version": 1,
        "loop_id": loop_id,
        "resource_level": resource_level,
        "run_identity": {"timestamp_utc": timestamp, "time_source": time_source, **copy.deepcopy(identities["execution"])},
        "identities": {key: copy.deepcopy(value) for key, value in identities.items() if key != "execution"},
        "evidence": {"files": [], "inventory_sha256": None},
        "status": "building",
    }
    if scheduler is not None:
        needed = {"job_id", "qtime_utc", "queue", "group", "nodefile_sha256", "modules"}
        if needed - scheduler.keys():
            raise ManifestError(f"scheduler identity missing fields: {sorted(needed - scheduler.keys())}")
        manifest["scheduler"] = copy.deepcopy(scheduler)
    validate_manifest(manifest, finalized=False)
    return manifest


def validate_manifest(manifest: dict[str, Any], *, finalized: bool = True) -> None:
    if manifest.get("schema_version") != 1:
        raise ManifestError("unsupported manifest schema")
    if _find_placeholder(manifest):
        raise ManifestError("placeholder in manifest")
    level = manifest.get("resource_level")
    if level not in _LEVELS:
        raise ManifestError("invalid resource level")
    execution = manifest.get("run_identity", {})
    timestamp = _parse_utc(execution.get("timestamp_utc"), "timestamp_utc")
    time_source = execution.get("time_source")
    identities = dict(manifest.get("identities", {}))
    identities["execution"] = {key: value for key, value in execution.items() if key not in {"timestamp_utc", "time_source"}}
    _validate_identities(identities)
    scheduler = manifest.get("scheduler")
    if level in {"L2", "L3", "L4"} and not isinstance(scheduler, dict):
        raise ManifestError("PBS profile requires scheduler identity")
    if level in {"L0", "L1"} and scheduler is not None:
        raise ManifestError("local profile must omit scheduler identity")
    if time_source == "pbs_qtime":
        if scheduler is None or scheduler.get("qtime_utc") != timestamp:
            raise ManifestError("timestamp and qtime mismatch")
    if finalized:
        evidence = manifest.get("evidence", {})
        files = evidence.get("files")
        if not isinstance(files, list) or files != sorted(files) or len(files) != len(set(files)):
            raise ManifestError("invalid evidence inventory")
        if evidence.get("inventory_sha256") != canonical_digest(files):
            raise ManifestError("evidence inventory digest mismatch")
        if manifest.get("status") != "finalized":
            raise ManifestError("manifest is not finalized")
