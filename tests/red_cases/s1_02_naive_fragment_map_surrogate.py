"""Deliberately wrong fragment mapper; all S1-02 RED cases execute."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys


def naive_greedy(weights: tuple[int, ...], fragment_count: int) -> tuple[int, ...]:
    cuts: list[int] = []
    start = 0
    for fragment in range(fragment_count - 1):
        remaining_fragments = fragment_count - fragment
        target = sum(weights[start:]) / remaining_fragments
        subtotal = 0
        end = start
        latest_end = len(weights) - (remaining_fragments - 1)
        while end < latest_end:
            candidate = weights[end]
            if end > start and abs(subtotal - target) <= abs(
                subtotal + candidate - target
            ):
                break
            subtotal += candidate
            end += 1
        cuts.append(end)
        start = end
    return tuple(cuts)


def maximum_bytes(weights: tuple[int, ...], cuts: tuple[int, ...]) -> int:
    starts = (0, *cuts)
    ends = (*cuts, len(weights))
    return max(sum(weights[start:end]) for start, end in zip(starts, ends, strict=True))


def unstable_digest() -> str:
    cut = int(next(iter({"2", "3"})))
    body = {"weights": [1, 1, 1, 1, 1], "fragment_count": 2, "cuts": [cut]}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


if os.environ.get("S1_02_DIGEST_CHILD") == "1":
    print(unstable_digest())
    raise SystemExit(0)


failures: list[str] = []
weights = (1, 5, 1, 4)
greedy_cuts = naive_greedy(weights, 3)
if maximum_bytes(weights, greedy_cuts) > maximum_bytes(weights, (1, 2)):
    failures.append("FRAG-04/05/A-FRAG-03: greedy target cuts are not minimax optimal")

child_env = {**os.environ, "S1_02_DIGEST_CHILD": "1"}
digests = []
for seed in ("1", "2"):
    child_env["PYTHONHASHSEED"] = seed
    result = subprocess.run(
        [sys.executable, __file__],
        check=True,
        capture_output=True,
        text=True,
        env=child_env,
    )
    digests.append(result.stdout.strip())
if len(set(digests)) != 1:
    failures.append(
        "FRAG-05/07: unordered tie selection changes the map digest across processes"
    )

invalid_counts_accepted = [0, 1, len(weights), len(weights) + 1]
if invalid_counts_accepted:
    failures.append(
        "FRAG-02/03: naive mapper has no production bounds or nonempty-interval validation"
    )

naive_parameter_buckets = ({"p0", "p1"}, {"p1", "p2"})
flattened = [identity for bucket in naive_parameter_buckets for identity in bucket]
if len(flattened) != len(set(flattened)) or set(flattened) != {"p0", "p1", "p2", "p3"}:
    failures.append("FRAG-01/08/A-FRAG-01/02: naive map duplicates p1 and omits p3")

if failures:
    raise AssertionError("\n".join(failures))
