"""Deliberately wrong S1-04 bootstrap used only to prove the semantic RED."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path


class NaiveBootstrap:
    """Wrong by construction: split files, shared head, overwrite, and scans."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.read_calls = 0

    def bootstrap(self, fragments: dict[str, bytes]) -> None:
        history = self.root / "history"
        history.mkdir(parents=True, exist_ok=True)
        for fragment, parameters in fragments.items():
            target = history / fragment
            target.mkdir(exist_ok=True)
            # Wrong: replace stable names independently and silently overwrite.
            (target / "parameters.bin").write_bytes(parameters)
            (target / "outer-state.bin").write_bytes(b"outer")
        # Wrong: one all-fragment head is required for authority.
        (self.root / "global-head.json").write_text(
            json.dumps({"fragments": sorted(fragments)}), encoding="utf-8"
        )

    def load(self) -> dict[str, tuple[bytes, bytes]]:
        self.read_calls = 0
        result: dict[str, tuple[bytes, bytes]] = {}
        # Wrong: restart discovery scans every historical object.
        for path in sorted((self.root / "history").rglob("*")):
            self.read_calls += 1
            if path.name != "parameters.bin":
                continue
            outer = path.with_name("outer-state.bin")
            # Wrong: missing compound state is accepted as empty outer state.
            result[path.parent.name] = (path.read_bytes(), outer.read_bytes() if outer.exists() else b"")
        return result


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="s1-04-red-") as directory:
        root = Path(directory)
        naive = NaiveBootstrap(root)
        initial = {"fragment-0": b"alpha", "fragment-1": b"beta"}
        naive.bootstrap(initial)

        # Partial compound publication must never be accepted.
        (root / "history/fragment-1/outer-state.bin").unlink()
        if naive.load()["fragment-1"][1] == b"":
            failures.append("GLOBAL-06/FS-03: partial fragment state was accepted")

        # Same identity with different content must not overwrite version zero.
        naive.bootstrap({"fragment-0": b"different", "fragment-1": b"beta"})
        if (root / "history/fragment-0/parameters.bin").read_bytes() == b"different":
            failures.append("GLOBAL-03/GLOBAL-06: conflicting bootstrap overwrote version zero")

        # Per-fragment authority must not depend on a shared global head.
        if (root / "global-head.json").exists():
            failures.append("GLOBAL-01/FS-04: bootstrap created a shared global head")

        baseline = naive.load()
        baseline_reads = naive.read_calls
        junk = root / "history/junk"
        junk.mkdir()
        for index in range(10_000):
            (junk / f"old-{index:05d}.bin").write_bytes(b"")
        after = naive.load()
        if naive.read_calls > baseline_reads or after != baseline:
            failures.append("FS-10: restart I/O grew with 10000 historical objects")

        payload_identities = [
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "history").rglob("parameters.bin")
        ]
        live_candidates = sum(1 for path in (root / "history").rglob("*") if path.is_file())
        if len(payload_identities) != len(initial) or live_candidates > 2 * len(initial):
            failures.append("SYNC-02/FS-06: storage has no bounded authoritative live-set contract")

    if not failures:
        raise SystemExit("surrogate unexpectedly satisfied S1-04")
    for failure in failures:
        print(failure)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
