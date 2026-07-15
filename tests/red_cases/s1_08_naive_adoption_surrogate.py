"""Retained RED: three tempting adoption implementations violate S1-08."""

from __future__ import annotations

import hashlib
import json


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    failures: list[dict[str, object]] = []

    # A naive full checkpoint load overwrites an unrelated fragment even when
    # only fragment zero has a newly published value.
    local = [b"local-f0", b"local-f1"]
    global_checkpoint = [b"global-f0-v3", b"stale-global-f1"]
    before = [_digest(value) for value in local]
    local[:] = global_checkpoint
    after = [_digest(value) for value in local]
    if after[1] != before[1]:
        failures.append(
            {
                "counterexample": "full_model_overwrite_for_target_adoption",
                "violation": "adopting fragment zero changed fragment one",
                "before": before,
                "after": after,
            }
        )

    # A vector-alignment guard rejects the valid mixed state [3, 0], so an
    # independently current fragment can never be adopted or trained.
    version_vector = [3, 0]
    alignment_guard_allows_training = len(set(version_vector)) == 1
    if not alignment_guard_allows_training:
        failures.append(
            {
                "counterexample": "full_version_vector_alignment_guard",
                "violation": "valid mixed-version training is blocked",
                "version_vector": version_vector,
            }
        )

    # Clearing Adam state on parameter overwrite destroys the retained inner
    # trajectory even though moments are not part of the global fragment.
    moments = {"step": b"7", "exp_avg": b"first", "exp_avg_sq": b"second"}
    before_moments = _digest(b"".join(moments[key] for key in sorted(moments)))
    moments.clear()
    after_moments = _digest(b"".join(moments[key] for key in sorted(moments)))
    if after_moments != before_moments:
        failures.append(
            {
                "counterexample": "default_optimizer_moment_reset",
                "violation": "target inner optimizer state changed during adoption",
                "before": before_moments,
                "after": after_moments,
            }
        )

    print(
        json.dumps(
            {
                "status": "semantic_red",
                "seed": 1808,
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
