from __future__ import annotations

import math


def naive_accepts(run: dict[str, object]) -> bool:
    return bool(run.get("status") == "complete")


def main() -> int:
    counterexamples = {
        "duplicate_hosts": {"status": "complete", "hosts": ["mg0001"] * 9},
        "missing_gpu_identity": {"status": "complete", "learner_gpus": [None] * 8},
        "nonfinite_loss": {"status": "complete", "losses": [4.0, math.nan]},
        "history_growth": {"status": "complete", "payload_counts": [10, 100, 1000]},
        "full_model_transfer": {"status": "complete", "full_model_operations": 40},
    }
    failures = 0
    for name, fixture in counterexamples.items():
        if naive_accepts(fixture):
            failures += 1
            print(f"RED {name}: naive analyzer accepted an invalid Stage 1 closure run")
    return 1 if failures == len(counterexamples) else 2


if __name__ == "__main__":
    raise SystemExit(main())
