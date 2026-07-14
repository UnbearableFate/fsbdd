"""Deliberately wrong S1-00 surrogate; RED must reach semantic assertions."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path


def naive_identity(config: dict[str, object]) -> str:
    # Historical defect: fragment/model identities were omitted.
    reduced = {"run": config["run"], "algorithm": config["algorithm"]}
    return hashlib.sha256(json.dumps(reduced, sort_keys=True).encode()).hexdigest()


def naive_validate(root: Path, listed: list[str]) -> bool:
    # Historical defect: only listed files are checked, extras are invisible.
    return all((root / relative).is_file() for relative in listed)


failures: list[str] = []
base = {
    "run": "r1",
    "algorithm": {"S_max": 0, "Q": 8},
    "model": {"identity": "m1"},
    "fragment_map": {"identity": "f1"},
}
mutated = {**base, "fragment_map": {"identity": "f2"}}
if naive_identity(base) == naive_identity(mutated):
    failures.append("PROP-01: fragment identity mutation was ignored")

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "listed.txt").write_text("ok", encoding="utf-8")
    (root / "unlisted.txt").write_text("must reject", encoding="utf-8")
    if naive_validate(root, ["listed.txt"]):
        failures.append("FS-07: unlisted extra file was accepted")

# Each independent semantic case executes even when an earlier one fails.
if {"schema_version": 1, "unknown": True}.get("schema_version") == 1:
    failures.append("DISC-03: naive schema accepted an unknown field")
if {"time_source": "invalid", "scheduler": {}}.get("scheduler") is not None:
    failures.append("FS-07: naive manifest accepted invalid time source and empty scheduler")

if failures:
    raise AssertionError("\n".join(failures))
