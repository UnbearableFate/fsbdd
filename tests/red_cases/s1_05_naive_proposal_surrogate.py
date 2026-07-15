"""Deliberately wrong S1-05 proposal protocol used only for semantic RED."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


class NaiveProposalStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        learner: str,
        fragment: int,
        sequence: int,
        metadata: dict,
        payload: bytes | None,
    ) -> None:
        history = self.root / f"{learner}-{fragment}-{sequence}.json"
        history.write_text(
            json.dumps({"sequence": sequence, "metadata": metadata}), encoding="utf-8"
        )
        if payload is not None:
            (self.root / f"{learner}-{fragment}-{sequence}.bin").write_bytes(payload)
        # Wrong: completion order overwrites latest even when sequence regresses.
        (self.root / f"latest-{learner}-{fragment}.json").write_text(
            history.read_text(), encoding="utf-8"
        )

    def discover(self) -> list[dict]:
        # Wrong: normal discovery scans and orders the complete history.
        return [
            json.loads(path.read_text()) for path in sorted(self.root.glob("*.json"))
        ]


def naive_eligible(
    proposals: list[dict],
    *,
    current_version: int,
    last_sequences: dict[str, int],
    q: int,
) -> list[dict]:
    # Wrong: fresh-only branch; no base-content/progress checks, consumed-base
    # frontier, or one-per-learner reduction.
    eligible = [
        item
        for item in proposals
        if item["base_version"] == current_version
        and item["sequence"] > last_sequences.get(item["learner"], -1)
    ]
    return eligible if len(eligible) >= q else []


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as temporary:
        store = NaiveProposalStore(Path(temporary))
        store.publish("a", 0, 1, {"base_version": 0}, None)
        if store.discover():
            failures.append(
                "PROP-01/A-PROP-01: payload-only partial proposal became discoverable"
            )
        store.publish("a", 0, 3, {"base_version": 0}, b"new")
        store.publish("a", 0, 2, {"base_version": 0}, b"old-delayed")
        latest = json.loads((Path(temporary) / "latest-a-0.json").read_text())
        if latest["sequence"] != 3:
            failures.append("PROP-04: delayed old completion regressed latest sequence")
        for index in range(10_000):
            (Path(temporary) / f"history-{index:05d}.json").write_text(
                "{}", encoding="utf-8"
            )
        if len(store.discover()) > 1:
            failures.append("PROP-03: discovery cost grew with 10000 history objects")

    duplicates = [
        {"learner": "a", "sequence": 1, "base_version": 4},
        {"learner": "a", "sequence": 2, "base_version": 4},
    ]
    if len(naive_eligible(duplicates, current_version=4, last_sequences={}, q=2)) == 2:
        failures.append("PROP-05/A-PROP-02: duplicate learner proposals formed quorum")
    same_base = [{"learner": "a", "sequence": 9, "base_version": 4}]
    if naive_eligible(same_base, current_version=4, last_sequences={"a": 5}, q=1):
        failures.append(
            "PROP-06/A-PROP-03: higher sequence on consumed base was accepted"
        )
    stale = [{"learner": "b", "sequence": 1, "base_version": 3}]
    if not naive_eligible(stale, current_version=4, last_sequences={}, q=1):
        failures.append(
            "DISC-01/STALE-01: generic S_max eligibility was replaced by fresh-only equality"
        )
    failures.append(
        "DISC-02/LEARN-07/PROP-07/STALE-05: schema lacks base identity progress bounds and shared weighting"
    )
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
