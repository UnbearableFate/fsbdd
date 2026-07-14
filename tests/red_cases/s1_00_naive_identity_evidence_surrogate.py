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


base = {
    "run": "r1",
    "algorithm": {"S_max": 0, "Q": 8},
    "model": {"identity": "m1"},
    "fragment_map": {"identity": "f1"},
}
mutated = {**base, "fragment_map": {"identity": "f2"}}
assert naive_identity(base) != naive_identity(mutated), "PROP-01: fragment identity mutation was ignored"

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "listed.txt").write_text("ok", encoding="utf-8")
    (root / "unlisted.txt").write_text("must reject", encoding="utf-8")
    assert not naive_validate(root, ["listed.txt"]), "FS-07: unlisted extra file was accepted"
