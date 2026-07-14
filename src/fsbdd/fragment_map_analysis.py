from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from .fragment_map import build_fragment_map, partition_layer_bytes
from .identity import canonical_digest
from .model_registry import Coverage, LogicalLayer, LogicalLayerRegistry, ParameterRecord


def _brute_objective(weights: tuple[int, ...], fragment_count: int) -> tuple[int, int, tuple[int, ...]]:
    total = sum(weights)
    candidates: list[tuple[int, int, tuple[int, ...]]] = []
    for cuts in itertools.combinations(range(1, len(weights)), fragment_count - 1):
        boundaries = (0, *cuts, len(weights))
        sizes = tuple(sum(weights[start:end]) for start, end in zip(boundaries, boundaries[1:]))
        candidates.append((max(sizes), sum(abs(fragment_count * size - total) for size in sizes), cuts))
    return min(candidates)


def _registry(weights: tuple[int, ...]) -> LogicalLayerRegistry:
    parameters = tuple(
        ParameterRecord(
            identity=f"probe-parameter-{index}",
            owner_layer=f"probe-layer-{index}",
            owner_name=f"probe.{index}.weight",
            aliases=(f"probe.{index}.weight",),
            shape=(weight,),
            dtype="probe",
            numel=weight,
            sync_bytes=weight,
        )
        for index, weight in enumerate(weights)
    )
    layers = tuple(
        LogicalLayer(
            index=index,
            name=f"probe-layer-{index}",
            kind="block",
            module_path=f"probe.{index}",
            parameter_identities=(parameters[index].identity,),
            parameter_count=weight,
            sync_bytes=weight,
        )
        for index, weight in enumerate(weights)
    )
    coverage = Coverage(len(parameters), len(parameters), 0, 0, 0, 0, 0)
    body = {
        "schema_version": 1,
        "family": "fragment-map-probe",
        "sync_dtype_bytes": 1,
        "layers": [layer.to_dict() for layer in layers],
        "parameters": [parameter.to_dict() for parameter in parameters],
        "buffers": [],
        "tied_identities": [],
        "misc_assignments": [],
        "coverage": coverage.to_dict(),
    }
    return LogicalLayerRegistry(
        schema_version=1,
        family="fragment-map-probe",
        sync_dtype_bytes=1,
        layers=layers,
        parameters=parameters,
        buffers=(),
        tied_identities=(),
        misc_assignments=(),
        coverage=coverage,
        digest=canonical_digest(body),
    )


def _objective_tuple(weights: tuple[int, ...], fragment_count: int) -> tuple[int, int, tuple[int, ...]]:
    objective = partition_layer_bytes(weights, fragment_count)
    return objective.maximum_sync_bytes, objective.total_deviation_scaled, objective.cut_indices


def analyze() -> dict[str, Any]:
    exhaustive_cases = 0
    for layer_count in (3, 4, 5, 6):
        for weights in itertools.product(range(6), repeat=layer_count):
            if sum(weights) == 0:
                continue
            for fragment_count in range(2, layer_count):
                exhaustive_cases += 1
                actual = _objective_tuple(weights, fragment_count)
                expected = _brute_objective(weights, fragment_count)
                if actual != expected:
                    raise AssertionError(
                        f"exhaustive mismatch weights={weights} F={fragment_count}: {actual} != {expected}"
                    )

    generator = random.Random(20260715)
    random_cases = 500
    for _ in range(random_cases):
        layer_count = generator.randint(3, 12)
        fragment_count = generator.randint(2, layer_count - 1)
        weights = tuple(generator.randint(0, 1_000_000) for _ in range(layer_count))
        if sum(weights) == 0:
            weights = (1, *weights[1:])
        actual = _objective_tuple(weights, fragment_count)
        expected = _brute_objective(weights, fragment_count)
        if actual != expected:
            raise AssertionError(f"random mismatch weights={weights} F={fragment_count}: {actual} != {expected}")
    return {
        "schema_version": 1,
        "status": "pass",
        "objective": ["minimum_maximum", "minimum_scaled_deviation", "earliest_cuts"],
        "exhaustive": {
            "layer_counts": [3, 4, 5, 6],
            "byte_values": [0, 1, 2, 3, 4, 5],
            "case_count": exhaustive_cases,
            "mismatches": 0,
        },
        "random": {
            "seed": 20260715,
            "case_count": random_cases,
            "maximum_layers": 12,
            "maximum_layer_bytes": 1_000_000,
            "mismatches": 0,
        },
    }


def _digest_repetitions() -> dict[str, Any]:
    digests: list[str] = []
    for seed in ("1", "2", "7", "31"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-m", "fsbdd.fragment_map_analysis", "--probe"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        digests.append(result.stdout.strip())
    if len(set(digests)) != 1:
        raise AssertionError(f"map digest changed across processes: {digests}")
    return {
        "schema_version": 1,
        "status": "pass",
        "process_count": len(digests),
        "python_hash_seeds": [1, 2, 7, 31],
        "digests": digests,
        "unique_digest_count": 1,
    }


def _write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fsbdd.fragment_map_analysis")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--digest-output", type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args(argv)
    if args.probe:
        print(build_fragment_map(_registry((7, 0, 11, 3, 5, 2, 13)), 4).digest)
        return 0
    if args.output is None or args.digest_output is None:
        parser.error("--output and --digest-output are required unless --probe is used")
    summary = analyze()
    repetitions = _digest_repetitions()
    _write_new(args.output, summary)
    _write_new(args.digest_output, repetitions)
    print(
        json.dumps(
            {
                "status": "pass",
                "exhaustive_cases": summary["exhaustive"]["case_count"],
                "random_cases": summary["random"]["case_count"],
                "unique_digest_count": repetitions["unique_digest_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
