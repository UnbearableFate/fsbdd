"""Retained RED: three tempting snapshot/publish implementations are invalid."""

from __future__ import annotations

import json


def main() -> int:
    failures: list[dict[str, str]] = []

    # Naive modulo code gives every fragment the same phase and creates one
    # full-model-sized burst instead of distributing byte work over H.
    fragment_bytes = (154_533_888, 170_108_928, 170_115_072, 154_533_888)
    offsets = tuple(0 for _ in fragment_bytes)
    if len(set(offsets)) == 1:
        failures.append(
            {
                "counterexample": "all_fragments_due_same_step",
                "violation": "offset scheduler creates one full-byte burst",
            }
        )

    # A view into a live parameter is not a snapshot: the later optimizer
    # mutation silently changes the value that would be published.
    parameter = bytearray(b"old-fragment")
    live_view = memoryview(parameter)
    before = bytes(live_view)
    parameter[:] = b"new-fragment"
    if bytes(live_view) != before:
        failures.append(
            {
                "counterexample": "snapshot_during_optimizer_mutation",
                "violation": "snapshot aliases mutable model storage",
            }
        )

    # A publication task that consults the learner's current base at write
    # time rewrites an already-started old-base proposal after adoption.
    current_base = {"version": 3, "identity": "a" * 64}
    task = {"base": current_base}
    started = dict(task["base"])
    current_base.update(version=4, identity="b" * 64)
    if task["base"] != started:
        failures.append(
            {
                "counterexample": "in_flight_base_metadata_rewrite",
                "violation": "publication metadata follows later adoption",
            }
        )

    print(
        json.dumps(
            {
                "status": "semantic_red",
                "expected_failures": 3,
                "observed_failures": len(failures),
                "failures": failures,
            },
            sort_keys=True,
        )
    )
    return 1 if len(failures) == 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
