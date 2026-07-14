from __future__ import annotations

import dataclasses
from itertools import combinations

import pytest

from fsbdd.fragment_map import (
    FragmentMapError,
    build_fragment_map,
    partition_layer_bytes,
    validate_fragment_map,
)
from fsbdd.identity import canonical_digest
from fsbdd.model_registry import Coverage, LogicalLayer, LogicalLayerRegistry, ParameterRecord


def registry_from_bytes(weights: tuple[int, ...]) -> LogicalLayerRegistry:
    parameters = tuple(
        ParameterRecord(
            identity=f"parameter-{index}",
            owner_layer=f"layer-{index}",
            owner_name=f"layer.{index}.weight",
            aliases=(f"layer.{index}.weight",),
            shape=(weight,),
            dtype="test",
            numel=weight,
            sync_bytes=weight,
        )
        for index, weight in enumerate(weights)
    )
    layers = tuple(
        LogicalLayer(
            index=index,
            name=f"layer-{index}",
            kind="block",
            module_path=f"layer.{index}",
            parameter_identities=(parameters[index].identity,),
            parameter_count=weight,
            sync_bytes=weight,
        )
        for index, weight in enumerate(weights)
    )
    coverage = Coverage(len(parameters), len(parameters), 0, 0, 0, 0, 0)
    body = {
        "schema_version": 1,
        "family": "synthetic",
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
        family="synthetic",
        sync_dtype_bytes=1,
        layers=layers,
        parameters=parameters,
        buffers=(),
        tied_identities=(),
        misc_assignments=(),
        coverage=coverage,
        digest=canonical_digest(body),
    )


def brute_objective(weights: tuple[int, ...], fragment_count: int) -> tuple[int, int, tuple[int, ...]]:
    total = sum(weights)
    candidates = []
    for cuts in combinations(range(1, len(weights)), fragment_count - 1):
        boundaries = (0, *cuts, len(weights))
        sizes = tuple(sum(weights[start:end]) for start, end in zip(boundaries, boundaries[1:]))
        candidates.append((max(sizes), sum(abs(fragment_count * size - total) for size in sizes), cuts))
    return min(candidates)


def test_minimax_beats_naive_greedy_counterexample() -> None:
    registry = registry_from_bytes((1, 5, 1, 4))
    fragment_map = build_fragment_map(registry, 3)
    assert fragment_map.objective.cut_indices == (1, 2)
    assert fragment_map.objective.maximum_sync_bytes == 5
    assert tuple(fragment.sync_bytes for fragment in fragment_map.fragments) == (1, 5, 5)


def test_secondary_objective_and_front_cut_match_brute_force() -> None:
    for weights, fragment_count in (((1, 1, 1, 1, 1), 2), ((3, 1, 2, 2, 1), 3)):
        objective = partition_layer_bytes(weights, fragment_count)
        assert (
            objective.maximum_sync_bytes,
            objective.total_deviation_scaled,
            objective.cut_indices,
        ) == brute_objective(weights, fragment_count)
    assert partition_layer_bytes((1, 1, 1, 1, 1), 2).cut_indices == (2,)


@pytest.mark.parametrize("fragment_count", [0, 1, 4, 5, True])
def test_production_fragment_count_fails_closed(fragment_count: int) -> None:
    with pytest.raises(FragmentMapError, match="fragment_count|1 < F < L"):
        build_fragment_map(registry_from_bytes((1, 2, 3, 4)), fragment_count)


def test_nonproduction_oracle_mode_only_allows_one_fragment() -> None:
    registry = registry_from_bytes((1, 2, 3, 4))
    fragment_map = build_fragment_map(registry, 1, production=False)
    assert fragment_map.production_mode is False
    assert fragment_map.fragments[0].layer_indices == (0, 1, 2, 3)
    with pytest.raises(FragmentMapError, match="only F = 1"):
        build_fragment_map(registry, 2, production=False)


def test_oversized_layer_stays_whole_and_singleton() -> None:
    fragment_map = build_fragment_map(registry_from_bytes((2, 2, 50, 2, 2)), 3)
    oversized = next(fragment for fragment in fragment_map.fragments if 2 in fragment.layer_indices)
    assert oversized.layer_indices == (2,)
    assert oversized.sync_bytes == 50


def test_map_is_frozen_and_repeated_builds_have_identical_digest() -> None:
    registry = registry_from_bytes((7, 2, 9, 1, 4))
    first = build_fragment_map(registry, 3)
    second = build_fragment_map(registry, 3)
    assert first == second
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.fragment_count = 2  # type: ignore[misc]


def test_parameter_partition_and_interval_validation_fail_closed() -> None:
    registry = registry_from_bytes((3, 4, 5, 6))
    fragment_map = build_fragment_map(registry, 2)
    duplicate = dataclasses.replace(
        fragment_map.fragments[1],
        parameter_identities=fragment_map.fragments[0].parameter_identities,
    )
    tampered = dataclasses.replace(fragment_map, fragments=(fragment_map.fragments[0], duplicate))
    tampered = dataclasses.replace(tampered, digest=canonical_digest(tampered._body()))
    with pytest.raises(FragmentMapError, match="parameter identities"):
        validate_fragment_map(tampered, registry)


def test_registry_digest_and_coverage_are_revalidated() -> None:
    registry = registry_from_bytes((3, 4, 5, 6))
    with pytest.raises(FragmentMapError, match="registry digest"):
        build_fragment_map(dataclasses.replace(registry, digest="0" * 64), 2)


def test_map_layer_summary_cannot_diverge_from_registry() -> None:
    registry = registry_from_bytes((3, 4, 5, 6))
    fragment_map = build_fragment_map(registry, 2)
    changed_layer = dataclasses.replace(fragment_map.layers[0], module_path="different.path")
    tampered = dataclasses.replace(fragment_map, layers=(changed_layer, *fragment_map.layers[1:]))
    tampered = dataclasses.replace(tampered, digest=canonical_digest(tampered._body()))
    with pytest.raises(FragmentMapError, match="layer summaries"):
        validate_fragment_map(tampered, registry)


def test_report_contains_exact_ownership_and_balance_summary(tmp_path) -> None:
    from fsbdd.fragment_map import write_fragment_map_report

    fragment_map = build_fragment_map(registry_from_bytes((3, 4, 5, 6)), 2)
    destination = tmp_path / "fragment-map.json"
    assert write_fragment_map_report(fragment_map, destination) == fragment_map.digest
    body = destination.read_text(encoding="utf-8")
    assert '"maximum_to_average_ratio"' in body
    assert '"parameter_identities"' in body
    with pytest.raises(FragmentMapError, match="overwrite"):
        write_fragment_map_report(fragment_map, destination)
